# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton-distributed AllGather/ReduceScatter All2All manager for DP+EP.

Dispatch (AllGather): uses NCCL via vLLM's existing all_gatherv infrastructure.
Combine (ReduceScatter): uses Triton-distributed's NVSHMEM-based
reduce_scatter_2d_op which overlaps reduction with NVSHMEM communication.

This hybrid approach is intentional:
- AllGather carries routing metadata (small), NCCL handles it well and is
  CUDA-graph compatible.
- ReduceScatter carries expert output (large), NVSHMEM's overlap of
  reduction + scatter across streams provides the real performance gain.

Usage: --all2all-backend triton_distributed_ag_rs
"""

from __future__ import annotations

import atexit
import math

import torch

from triton_dist.kernels.nvidia.reduce_scatter import (
    ReduceScatter2DContext,
    create_reduce_scater_2d_ctx,
    reduce_scatter_for_each_node_ring,
    ring_reduce,
)
from triton_dist.utils import (
    init_nvshmem_by_torch_process_group,
    is_shmem_initialized,
    nvshmem_barrier_all_on_stream,
)

from vllm.config import get_current_vllm_config_or_none
from vllm.distributed import get_dp_group, get_ep_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

from .base_device_communicator import All2AllManagerBase

logger = init_logger(__name__)


class TritonDistAll2AllManager(All2AllManagerBase):
    """
    Hybrid AG(NCCL) + RS(NVSHMEM) manager for DP+EP.

    Dispatch uses NCCL AllGatherv (reliable, well-tested).
    Combine uses Triton-distributed ReduceScatter (NVSHMEM, overlapped).
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)

        self._rs_initialized = False
        self._rs_ctx: ReduceScatter2DContext | None = None
        self._rs_max_M: int = 0
        self._rs_N: int = 0
        self._local_world_size = self._resolve_local_world_size()

    def _resolve_local_world_size(self) -> int:
        """Resolve local world size from existing parallel config."""
        if not self.internode:
            local_world_size = self.world_size
        else:
            vllm_config = get_current_vllm_config_or_none()
            if vllm_config is None:
                raise RuntimeError(
                    "vLLM config is not available when initializing "
                    "triton_distributed_ag_rs."
                )
            local_world_size = vllm_config.parallel_config.data_parallel_size_local

        if local_world_size <= 0 or self.world_size % local_world_size != 0:
            raise RuntimeError(
                "Invalid Triton-dist RS topology: world_size=%d, "
                "local_world_size=%d"
                % (self.world_size, local_world_size)
            )

        logger.info(
            "[Rank %d/%d] Triton-dist RS topology: local_world_size=%d",
            self.rank,
            self.world_size,
            local_world_size,
        )
        return local_world_size

    def _ensure_nvshmem(self):
        """Ensure NVSHMEM runtime is initialized."""
        if not is_shmem_initialized():
            ep_group = get_ep_group()
            logger.info(
                "[Rank %d/%d] Initializing NVSHMEM for Triton-dist RS.",
                self.rank,
                self.world_size,
            )
            init_nvshmem_by_torch_process_group(ep_group.device_group)

    def _ensure_rs_ctx(self, total_tokens: int, hidden: int):
        """Lazily create ReduceScatter context sized for the workload."""
        ep_group = get_ep_group()
        target_device = getattr(ep_group, "device", None)
        if (
            isinstance(target_device, torch.device)
            and target_device.type == "cuda"
            and target_device.index is not None
            and torch.cuda.current_device() != target_device.index
        ):
            torch.cuda.set_device(target_device)

        # Pad to multiple of world_size (RS requirement)
        padded = math.ceil(total_tokens / self.world_size) * self.world_size

        if (
            self._rs_initialized
            and padded <= self._rs_max_M
            and hidden <= self._rs_N
        ):
            return

        self._ensure_nvshmem()

        logger.info(
            "[Rank %d] Creating Triton-dist ReduceScatter context: "
            "M=%d (padded=%d), N=%d",
            self.rank,
            total_tokens,
            padded,
            hidden,
        )

        local_world_size = self._local_world_size
        self._rs_ctx = create_reduce_scater_2d_ctx(
            max_M=padded,
            N=hidden,
            rank=self.rank,
            world_size=self.world_size,
            local_world_size=local_world_size,
            dtype=torch.bfloat16,
        )

        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

        self._rs_max_M = padded
        self._rs_N = hidden
        self._rs_initialized = True

        if not hasattr(self, "_atexit_registered"):
            atexit.register(self.destroy)
            self._atexit_registered = True

    # ------------------------------------------------------------------
    # Dispatch: NCCL AllGatherv (delegates to vLLM's existing infra)
    # ------------------------------------------------------------------

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

        gathered = dist_group.all_gatherv(tensors_to_gather, dim=0, sizes=sizes)

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

        gathered = dist_group.all_gatherv(tensors_to_gather, dim=0, sizes=sizes)

        if extra_tensors is None:
            return gathered[0], gathered[1], gathered[2]
        return gathered[0], gathered[1], gathered[2], gathered[3:]

    # ------------------------------------------------------------------
    # Combine: Triton-distributed NVSHMEM ReduceScatter
    # ------------------------------------------------------------------

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        if hidden_states.device.type == "cuda":
            expected_device = hidden_states.device
            if torch.cuda.current_device() != expected_device.index:
                torch.cuda.set_device(expected_device)

        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None
        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        total_tokens = hidden_states.shape[0]
        hidden = hidden_states.shape[1]

        # Triton-dist RS kernels use flat pointer arithmetic internally.
        # Keep the input contiguous to avoid invalid address calculations.
        hidden_states = hidden_states.contiguous()

        # RS requires M % world_size == 0; pad if needed
        padded_tokens = (
            math.ceil(total_tokens / self.world_size) * self.world_size
        )

        self._ensure_rs_ctx(padded_tokens, hidden)

        if padded_tokens != total_tokens:
            hidden_states = torch.nn.functional.pad(
                hidden_states, (0, 0, 0, padded_tokens - total_tokens)
            )

        # NOTE: Force ring RS path for stability in vLLM AG/RS backend.
        # The fullmesh path may hit CUDA_ERROR_INVALID_VALUE in _wait_eq_cuda
        # for this workload/topology.
        M_per_rank = hidden_states.shape[0] // self._rs_ctx.world_size
        current_stream = torch.cuda.current_stream()
        self._rs_ctx.reduction_stream.wait_stream(current_stream)
        output = torch.empty(
            (M_per_rank, hidden_states.shape[1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        out_each_node = output if self._rs_ctx.nnodes == 1 else None
        rs_result_per_node = reduce_scatter_for_each_node_ring(
            hidden_states,
            self._rs_ctx,
            out_each_node,
        )
        if self._rs_ctx.nnodes > 1:
            nvshmem_barrier_all_on_stream(current_stream)
            ring_reduce(
                rs_result_per_node,
                output,
                self._rs_ctx.node_id,
                self._rs_ctx.nnodes,
            )
        self._rs_ctx.reset_barriers()

        # output shape: [padded_tokens // world_size, hidden]
        # Trim to this rank's actual token count
        my_size = sizes[dist_group.rank_in_group]
        return output[:my_size]

    def destroy(self):
        self._rs_ctx = None
        self._rs_initialized = False
