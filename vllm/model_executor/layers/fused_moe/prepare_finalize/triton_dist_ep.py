# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton-Distributed EP MoE integration.

Uses ByteDance Seed's Triton-distributed mega-kernels that fuse
All-to-All communication with Group GEMM computation at the
Triton tile level via NVSHMEM, achieving fine-grained
communication-computation overlap within a single kernel launch.

Key APIs:
- EpAll2AllFusedOp.mega_dispatch_group_gemm(): fused A2A dispatch + W1 GEMM
- EpAll2AllFusedOp.mega_group_gemm_combine(): fused W2 GEMM + A2A combine
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
import triton

from triton_dist.function.nvidia.common import (
    TritonDistEpContext,
    get_moe_optim_config,
    get_triton_dist_moe_profile_enabled,
    init_triton_dist_ep_ctx,
    init_triton_dist_ep_op,
    triton_dist_ep_op_initialized,
)
from triton_dist.utils import (
    init_nvshmem_by_torch_process_group,
    is_shmem_initialized,
)
from triton_dist.kernels.nvidia.group_gemm import (
    GROUP_GEMM_BLOCK_SIZE_M,
    build_block_row_idx_info_kernel,
)
from triton_dist.kernels.nvidia.swiglu import swiglu_forward

from vllm.distributed import get_ep_group
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

logger = init_logger(__name__)


class TritonDistEPState:
    """
    Manages Triton-distributed EP state including NVSHMEM initialization
    and EpAll2AllFusedOp lifecycle.

    This is initialized lazily on first forward pass since model
    parameters (hidden_size, etc.) are only known after weight loading.
    """

    def __init__(self):
        self._initialized = False
        self._ep_ctx: TritonDistEpContext | None = None
        self._init_lock = threading.Lock()

    def _is_runtime_initialized(self) -> bool:
        return triton_dist_ep_op_initialized(ep_implementation="mega")

    def invalidate(self) -> None:
        """Invalidate local initialization state.

        This is used as a recovery path when runtime reports NVSHMEM is not
        initialized even though this helper believes it is initialized.
        """
        with self._init_lock:
            self._initialized = False
            self._ep_ctx = None

    def ensure_initialized(
        self,
        ep_group: torch.distributed.ProcessGroup,
        max_tokens_per_rank: int,
        hidden_size: int,
        top_k: int,
        num_experts: int,
        num_sm: int = 64,
        capacity: float = 4.0,
    ) -> None:
        with self._init_lock:
            # Guard against stale local state: only short-circuit when both
            # local state and Triton-dist runtime state are initialized.
            if self._initialized and self._is_runtime_initialized():
                return

            ep_rank = ep_group.rank()
            ep_size = ep_group.size()

            # Triton-distributed NVSHMEM tensors require the NVSHMEM runtime
            # to be initialized against the EP process group first.
            if not is_shmem_initialized():
                logger.info(
                    "[EP Rank %d/%d] Initializing NVSHMEM runtime for "
                    "Triton-distributed EP.",
                    ep_rank,
                    ep_size,
                )
                init_nvshmem_by_torch_process_group(ep_group)

            if not self._is_runtime_initialized():
                logger.info(
                    "[EP Rank %d/%d] Initializing Triton-distributed EP "
                    "with max_tokens=%d, hidden=%d, topk=%d, "
                    "num_experts=%d, num_sm=%d, capacity=%.1f",
                    ep_rank,
                    ep_size,
                    max_tokens_per_rank,
                    hidden_size,
                    top_k,
                    num_experts,
                    num_sm,
                    capacity,
                )
                init_triton_dist_ep_op(
                    ep_group=ep_group,
                    max_tokens_per_rank=max_tokens_per_rank,
                    hidden_size=hidden_size,
                    topk=top_k,
                    ep_rank=ep_rank,
                    num_experts=num_experts,
                    ep_size=ep_size,
                    dtype=torch.bfloat16,
                    weight_dtype=torch.float32,
                    num_sm=num_sm,
                    num_buffers=1,
                    capacity=capacity,
                )

            self._initialized = True

    def get_context(
        self,
        ep_group: torch.distributed.ProcessGroup,
        top_k: int,
        num_experts: int,
    ) -> TritonDistEpContext:
        """Get or create a TritonDistEpContext for the current forward pass."""
        return init_triton_dist_ep_ctx(
            ep_group=ep_group,
            topk=top_k,
            num_experts=num_experts,
            ep_implementation="mega",
        )


# Global state - NVSHMEM should only be initialized once per process
_triton_dist_state = TritonDistEPState()


def triton_dist_ep_forward(
    layer: FusedMoE,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Forward pass using Triton-distributed mega-kernels for EP MoE.

    Fuses the full MoE pipeline:
    1. Router: softmax + topk selection
    2. mega_dispatch_group_gemm: fused A2A dispatch + W1 (gate_up) GEMM
    3. SwiGLU activation with routing weight scaling
    4. mega_group_gemm_combine: fused W2 (down) GEMM + A2A combine

    Args:
        layer: The FusedMoE layer containing weights and config
        hidden_states: Input tensor [num_tokens, hidden_size]
        router_logits: Router output [num_tokens, num_experts]

    Returns:
        Output tensor [num_tokens, hidden_size]
    """
    ep_group = get_ep_group().device_group
    ep_rank = ep_group.rank()
    ep_size = ep_group.size()
    num_experts = layer.global_num_experts
    top_k = layer.top_k
    num_experts_per_rank = num_experts // ep_size
    hidden_size = hidden_states.shape[-1]

    # Ensure NVSHMEM and EpAll2AllFusedOp are initialized.
    # If runtime reports NVSHMEM is not initialized, invalidate local state and
    # retry exactly once.
    #
    # CRITICAL: max_tokens_per_rank must be the scheduler's max_num_batched_tokens,
    # NOT moe_config.max_num_tokens (which is the DP chunk size, default 256).
    # EpAll2AllFusedOp allocates NVSHMEM buffers sized [nnodes, max_tokens, topk]
    # at init time. If the actual batch exceeds this, preprocess() will crash with
    # a tensor shape mismatch.
    #
    # layer.vllm_config is captured at layer __init__ time (when config context
    # is available), so it is safe to read here at forward time.
    max_tokens_per_rank = (
        layer.vllm_config.scheduler_config.max_num_batched_tokens
    )
    retry_once = True
    while True:
        try:
            _triton_dist_state.ensure_initialized(
                ep_group=ep_group,
                max_tokens_per_rank=max_tokens_per_rank,
                hidden_size=hidden_size,
                top_k=top_k,
                num_experts=num_experts,
            )

            # Get context for this forward pass
            ctx = _triton_dist_state.get_context(
                ep_group=ep_group,
                top_k=top_k,
                num_experts=num_experts,
            )
            break
        except Exception as e:
            msg = str(e)
            if retry_once and "NVSHMEM Library is not initialized" in msg:
                logger.warning(
                    "Triton-distributed runtime reported uninitialized NVSHMEM. "
                    "Retrying initialization once."
                )
                _triton_dist_state.invalidate()
                retry_once = False
                continue
            raise

    ep_op = ctx.ep_op

    # --- Step 1: Routing ---
    # Currently supports standard TopK routing only.
    # GroupedTopK or custom routing functions are not yet supported.
    if layer.use_grouped_topk:
        raise NotImplementedError(
            "Triton-distributed EP backend does not yet support "
            "GroupedTopK routing. Use a different --all2all-backend."
        )

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(
        routing_weights, top_k, dim=-1
    )
    if layer.renormalize:
        routing_weights = routing_weights / routing_weights.sum(
            dim=-1, keepdim=True
        )
    selected_experts = selected_experts.to(torch.int32)

    # --- Step 2: Compute local scatter indices for A2A layout ---
    local_scatter_indices = (
        selected_experts.flatten()
        .argsort(stable=True)
        .argsort()
        .int()
        .view(selected_experts.shape)
    )

    # --- Step 3: Preprocess A2A layout ---
    ep_a2a_layout_desc = ep_op.preprocess(
        selected_experts, None, local_scatter_indices
    )
    token_splits_this_rank = ep_a2a_layout_desc.recv_buf_tokens_per_expert[
        ep_rank
    ]

    # --- Step 4: Build group GEMM metadata ---
    optim_config = get_moe_optim_config(use_mega=True)
    profile_config = get_triton_dist_moe_profile_enabled()

    build_block_row_idx_info_kernel[(optim_config.num_build_sms,)](
        token_splits_this_rank,
        ctx.split_size_cum_per_expert,
        ctx.expert_ids,
        ctx.split_size_cum,
        ctx.tile_num,
        ctx.tile_num_cum,
        ctx.expert_tile_offset,
        ctx.num_tiles_total,
        num_experts_per_rank,
        triton.next_power_of_2(num_experts_per_rank),
        GROUP_GEMM_BLOCK_SIZE_M,
        optim_config.num_build_sms,
    )

    # --- Step 5: Fused dispatch + W1 GEMM (mega kernel) ---
    w1 = layer.w13_weight  # [num_local_experts, intermediate*2, hidden]
    w2 = layer.w2_weight  # [num_local_experts, hidden, intermediate]

    (
        dispatch_output_local,
        dispatch_weight_in_buf,
        dispatch_layout_desc,
        fc1_output,
    ) = ep_op.mega_dispatch_group_gemm(
        # dispatch token
        input=hidden_states,
        exp_indices=selected_experts,
        ep_a2a_layout_desc=ep_a2a_layout_desc,
        # group gemm
        gemm_weight=w1,
        gemm_expert_ids=ctx.expert_ids,
        gemm_split_size=token_splits_this_rank,
        gemm_split_size_cum=ctx.split_size_cum,
        gemm_tile_num=ctx.tile_num,
        gemm_tile_num_cum=ctx.tile_num_cum,
        gemm_num_tiles_total=ctx.num_tiles_total,
        gemm_expert_offs=ctx.split_size_cum_per_expert,
        # dispatch token
        weight=routing_weights,
        with_cpy_flag=True,
        comm_buffer_id=0,
        optional_sm=optim_config.num_dispatch_sms,
        num_tail_sms=optim_config.num_tail_sms_in_dispatch,
        # group gemm
        gemm_input_reduce_last_dim=True,
        gemm_weight_reduce_last_dim=True,
        gemm_output_data=None,
        gemm_BLOCK_SIZE_N=ep_op.FWD_GEMM_BLOCK_SIZE_N,
        gemm_BLOCK_SIZE_K=64,
        gemm_GROUP_SIZE_M=1,
        gemm_num_stages=3,
        # common
        use_block_wise_barrier=optim_config.dispatch_use_block_wise_barrier,
        num_warps=optim_config.num_dispatch_warps,
        enable_profiler=profile_config["fwd_dispatch"],
    )

    # --- Step 6: SwiGLU activation ---
    swiglu_output, _ = swiglu_forward(
        fc1_output, scale=dispatch_weight_in_buf.view(-1)
    )

    # --- Step 7: Fused W2 GEMM + combine (mega kernel) ---
    combine_output = ep_op.mega_group_gemm_combine(
        # group gemm
        gemm_input_data=swiglu_output,
        gemm_weight=w2,
        gemm_expert_ids=ctx.expert_ids,
        gemm_split_size=token_splits_this_rank,
        gemm_split_size_cum=ctx.split_size_cum,
        gemm_tile_num=ctx.tile_num,
        gemm_tile_num_cum=ctx.tile_num_cum,
        gemm_num_tiles_total=ctx.num_tiles_total,
        # combine token
        ep_a2a_layout_desc=dispatch_layout_desc,
        # group gemm
        gemm_input_reduce_last_dim=True,
        gemm_weight_reduce_last_dim=True,
        gemm_BLOCK_SIZE_N=ep_op.FWD_GEMM_BLOCK_SIZE_N,
        gemm_BLOCK_SIZE_K=64,
        gemm_GROUP_SIZE_M=1,
        gemm_num_stages=3,
        # combine token
        gate_input=None,
        cp_flag=False,
        combine_output=None,
        output_gate=None,
        optional_sm=optim_config.num_combine_sms,
        num_reduce_sms=optim_config.num_reduce_sms_in_combine,
        optional_signal_tensor=None,
        num_warps=optim_config.num_combine_warps,
        combine_mode="fuse_scatter",
        enable_profiler=profile_config["fwd_combine"],
    )

    # The expected output shape is [num_tokens, hidden_size]. Some runtime
    # configurations can return [num_tokens * top_k, hidden_size]. Normalize
    # the latter back to token-major layout to keep vLLM downstream contracts.
    expected_tokens = hidden_states.shape[0]
    if combine_output.shape[0] != expected_tokens:
        if combine_output.shape[0] == expected_tokens * top_k:
            logger.warning_once(
                "Triton-distributed combine output has token-replica shape "
                "[%d, %d], folding back to [%d, %d] by top-k reduction.",
                combine_output.shape[0],
                combine_output.shape[1],
                expected_tokens,
                combine_output.shape[1],
                scope="global",
            )
            combine_output = combine_output.view(expected_tokens, top_k, -1).sum(dim=1)
        else:
            raise RuntimeError(
                "Unexpected Triton-distributed combine output shape "
                f"{tuple(combine_output.shape)} for input tokens={expected_tokens}, "
                f"top_k={top_k}."
            )

    return combine_output
