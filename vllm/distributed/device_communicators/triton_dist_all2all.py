# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton-distributed AllGather/ReduceScatter All2All manager for DP+EP.

Replaces NCCL-based AllGatherv/ReduceScatterv with Triton-distributed's
NVSHMEM-based implementations. Fits into vLLM's modular prepare/finalize
framework — expert computation still uses vLLM's standard Triton/CUTLASS
kernels.

Key difference from the monolithic triton_distributed backend:
- Monolithic: fuses comm + GEMM in one mega-kernel (NVSHMEM + Group GEMM)
- This AG/RS: only replaces the communication, expert compute is standard

Usage: --all2all-backend triton_distributed_ag_rs
"""

from __future__ import annotations

import atexit
import math

import torch

from triton_dist.kernels.nvidia.reduce_scatter import (
    ReduceScatter2DContext,
    create_reduce_scater_2d_ctx,
    reduce_scatter_2d_op,
)
from triton_dist.utils import (
    NVSHMEM_SIGNAL_DTYPE,
    init_nvshmem_by_torch_process_group,
    is_shmem_initialized,
    nvshmem_barrier_all_on_stream,
    nvshmem_create_tensor,
)

from vllm.distributed import get_dp_group, get_ep_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

from .base_device_communicator import All2AllManagerBase

logger = init_logger(__name__)


class TritonDistAll2AllManager(All2AllManagerBase):
    """
    AllGather/ReduceScatter for DP+EP using Triton-distributed NVSHMEM.

    Dispatch (AllGather): copies local tokens into a pre-allocated NVSHMEM
    symmetric buffer, then each rank reads the full gathered tensor.

    Combine (ReduceScatter): uses Triton-distributed's optimized
    reduce_scatter_2d_op which overlaps reduction with NVSHMEM communication.
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)

        self._initialized = False
        # Will be set during lazy init
        self._ag_symm_buf: torch.Tensor | None = None  # [max_total_tokens, max_hidden]
        self._ag_signal: torch.Tensor | None = None
        self._rs_ctx: ReduceScatter2DContext | None = None
        self._max_total_tokens: int = 0
        self._max_hidden: int = 0

    def _ensure_nvshmem(self):
        """Ensure NVSHMEM runtime is initialized."""
        if not is_shmem_initialized():
            # Use the EP group's underlying process group for NVSHMEM init
            ep_group = get_ep_group()
            logger.info(
                "[Rank %d/%d] Initializing NVSHMEM for Triton-dist AG/RS.",
                self.rank,
                self.world_size,
            )
            init_nvshmem_by_torch_process_group(ep_group.device_group)

    def _ensure_buffers(self, total_tokens: int, hidden: int):
        """Lazily allocate NVSHMEM symmetric buffers sized for the workload."""
        if (
            self._initialized
            and total_tokens <= self._max_total_tokens
            and hidden <= self._max_hidden
        ):
            return

        self._ensure_nvshmem()

        # Round total_tokens up to multiple of world_size (RS requirement)
        total_tokens_padded = (
            math.ceil(total_tokens / self.world_size) * self.world_size
        )

        logger.info(
            "[Rank %d] Allocating Triton-dist AG/RS buffers: "
            "total_tokens=%d (padded=%d), hidden=%d",
            self.rank,
            total_tokens,
            total_tokens_padded,
            hidden,
        )

        # AllGather buffer: each rank writes its chunk, all ranks read full
        self._ag_symm_buf = nvshmem_create_tensor(
            (total_tokens_padded, hidden), torch.bfloat16
        )
        # Signal buffer for AG synchronization
        self._ag_signal = nvshmem_create_tensor(
            (self.world_size,), NVSHMEM_SIGNAL_DTYPE
        )

        # ReduceScatter context
        local_world_size = min(self.world_size, 8)  # intra-node
        self._rs_ctx = create_reduce_scater_2d_ctx(
            max_M=total_tokens_padded,
            N=hidden,
            rank=self.rank,
            world_size=self.world_size,
            local_world_size=local_world_size,
            dtype=torch.bfloat16,
        )

        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

        self._max_total_tokens = total_tokens_padded
        self._max_hidden = hidden
        self._initialized = True

        if not hasattr(self, "_atexit_registered"):
            atexit.register(self.destroy)
            self._atexit_registered = True

    def _allgather_nvshmem(
        self, tensors: list[torch.Tensor], sizes: list[int]
    ) -> list[torch.Tensor]:
        """AllGather tensors along dim=0 using NVSHMEM symmetric memory.

        Each rank copies its local data into the symmetric buffer at its
        designated offset, signals completion, then all ranks wait and
        read the full buffer.
        """
        total_tokens = sum(sizes)
        hidden = tensors[0].shape[1] if tensors[0].dim() > 1 else 1

        # Compute the max hidden across all tensors for buffer sizing
        max_hidden = max(t.shape[1] if t.dim() > 1 else 1 for t in tensors)
        self._ensure_buffers(total_tokens, max_hidden)

        # Compute this rank's offset in the gathered tensor
        dist_group = get_ep_group()
        my_rank = dist_group.rank_in_group
        offset = sum(sizes[:my_rank])

        results = []
        for tensor in tensors:
            t_hidden = tensor.shape[1] if tensor.dim() > 1 else 1

            # Use NVSHMEM symmetric buffer for allgather
            buf = self._ag_symm_buf[:total_tokens, :t_hidden]

            # Reset signal
            self._ag_signal.zero_()
            nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

            # Copy local data to our slot in the symmetric buffer
            buf[offset : offset + sizes[my_rank]].copy_(tensor)

            # Signal that our data is ready
            from triton_dist.kernels.nvidia.common_ops import _set_signal_cuda
            _set_signal_cuda(
                self._ag_signal[my_rank], 1, torch.cuda.current_stream()
            )

            # Wait for all ranks
            nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

            # Read the full gathered tensor
            gathered = buf[:total_tokens].clone()
            results.append(gathered)

        return results

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ):
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None
        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        assert sizes[dist_group.rank_in_group] == hidden_states.shape[0]

        tensors_to_gather = [hidden_states, router_logits]
        if extra_tensors is not None:
            tensors_to_gather.extend(extra_tensors)

        gathered = self._allgather_nvshmem(tensors_to_gather, sizes)

        if extra_tensors is not None:
            return gathered[0], gathered[1], gathered[2:]
        return gathered[0], gathered[1]

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ):
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None
        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        assert sizes[dist_group.rank_in_group] == hidden_states.shape[0]

        tensors_to_gather = [hidden_states, topk_weights, topk_ids]
        if extra_tensors is not None:
            tensors_to_gather.extend(extra_tensors)

        gathered = self._allgather_nvshmem(tensors_to_gather, sizes)

        if extra_tensors is None:
            return gathered[0], gathered[1], gathered[2]
        return gathered[0], gathered[1], gathered[2], gathered[3:]

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None

        total_tokens = hidden_states.shape[0]
        hidden = hidden_states.shape[1]

        # ReduceScatter requires M % world_size == 0; pad if needed
        padded_tokens = (
            math.ceil(total_tokens / self.world_size) * self.world_size
        )

        if padded_tokens != total_tokens:
            pad_size = padded_tokens - total_tokens
            hidden_states = torch.nn.functional.pad(
                hidden_states, (0, 0, 0, pad_size)
            )

        self._ensure_buffers(padded_tokens, hidden)

        # Use Triton-distributed reduce_scatter
        output = reduce_scatter_2d_op(hidden_states, self._rs_ctx)

        # Output is [padded_tokens // world_size, hidden]
        # Trim to our actual size
        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        my_size = sizes[dist_group.rank_in_group]
        return output[:my_size]

    def destroy(self):
        if self._rs_ctx is not None:
            try:
                from triton_dist.utils import nvshmem_free_tensor_sync
                if self._ag_symm_buf is not None:
                    nvshmem_free_tensor_sync(self._ag_symm_buf)
                if self._ag_signal is not None:
                    nvshmem_free_tensor_sync(self._ag_signal)
                self._rs_ctx = None
            except Exception:
                pass
        self._initialized = False
