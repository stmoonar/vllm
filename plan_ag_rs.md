# Triton-Distributed AG/RS 后端实现文档

## 概述

`triton_distributed_ag_rs` 是一个 **模块化（modular）** 的 DP+EP 通信后端，用 Triton-distributed 的 NVSHMEM 实现替换 NCCL 的 AllGather/ReduceScatter，但保留 vLLM 标准的 expert 计算 kernel。

与已有的 `triton_distributed`（monolithic）后端的根本区别：

```
triton_distributed (monolithic):
  整个 MoE 层 = 一个函数调用
  dispatch + W1 GEMM + activation + W2 GEMM + combine 全部在 mega-kernel 内
  绕过 vLLM 的 modular prepare/finalize 框架

triton_distributed_ag_rs (modular):
  只替换通信层 (NCCL → NVSHMEM)
  expert 计算仍用 vLLM 标准 Triton/CUTLASS kernel
  完全融入 vLLM 的 modular prepare/finalize 框架
```

## 设计动机

| 维度 | monolithic (triton_distributed) | modular (triton_distributed_ag_rs) |
|------|------|------|
| 通信计算重叠 | tile 级别（单 kernel 内） | stream 级别（标准异步） |
| expert 量化支持 | 仅 BF16 | 自动继承 vLLM 全部量化（FP8、GPTQ 等） |
| CUDA Graph 兼容 | PIECEWISE（MoE 排除） | PIECEWISE（MoE 排除） |
| Shared experts 并行 | 不支持 | 支持（modular 框架原生支持） |
| 代码侵入性 | 高（需要 monolithic 路径） | 低（仅替换 All2AllManager） |
| 维护成本 | 需跟踪 Triton-dist mega-kernel API | 仅需跟踪 AG/RS 原语 |
| 最佳场景 | 大 batch prefill（通信可被 GEMM 隐藏） | 通用场景（兼容性优先） |

## 架构

### 数据流

```
  Rank 0: [M₀, K]     Rank 1: [M₁, K]     Rank 2: [M₂, K]     Rank 3: [M₃, K]
       │                    │                    │                    │
       └────────────────────┴────────────────────┴────────────────────┘
                                     │
                          NVSHMEM AllGather (dispatch)
                          每个 rank 写入 symmetric buffer 的自己的 slot
                          barrier 同步后，所有 rank 读取完整 buffer
                                     │
                                     ▼
                All Ranks: [M₀+M₁+M₂+M₃, K]  (gathered hidden_states)
                           [M₀+M₁+M₂+M₃, E]  (gathered router_logits)
                                     │
                          vLLM 标准 routing (softmax + topk)
                          vLLM 标准 expert compute (Triton/CUTLASS GEMM)
                                     │
                                     ▼
                All Ranks: [M₀+M₁+M₂+M₃, K]  (expert output)
                                     │
                          NVSHMEM ReduceScatter (combine)
                          Triton-dist reduce_scatter_2d_op:
                            intra-node scatter + inter-node p2p + reduction
                            通信与 reduction 在独立 stream 上重叠
                                     │
       ┌────────────────────┬────────────────────┬────────────────────┐
       │                    │                    │                    │
  Rank 0: [M₀, K]     Rank 1: [M₁, K]     Rank 2: [M₂, K]     Rank 3: [M₃, K]
```

### 在 vLLM 框架中的位置

```
DefaultMoERunner.forward_impl()
  │
  ├── _maybe_dispatch()  ──→ get_ep_group().dispatch()
  │                            └── All2AllManager.dispatch()
  │                                 └── TritonDistAll2AllManager.dispatch()
  │                                      └── _allgather_nvshmem()  ← NVSHMEM
  │
  ├── _apply_quant_method()
  │     ├── router.select_experts()  (标准 routing)
  │     └── kernel.apply()           (标准 Triton/CUTLASS expert GEMM)
  │
  └── _maybe_combine()  ──→ get_ep_group().combine()
                              └── All2AllManager.combine()
                                   └── TritonDistAll2AllManager.combine()
                                        └── reduce_scatter_2d_op()  ← NVSHMEM
```

关键：`TritonDistAll2AllManager` 是 `All2AllManagerBase` 的子类，与 `AgRsAll2AllManager`（NCCL）接口完全相同。vLLM 的 modular prepare/finalize 层完全不感知底层通信实现的变化。

## 实现细节

### 核心类：TritonDistAll2AllManager

**文件**: `vllm/distributed/device_communicators/triton_dist_all2all.py`

```python
class TritonDistAll2AllManager(All2AllManagerBase):
    # 继承自 vLLM 的标准 All2AllManager 接口
    # 实现 dispatch(), dispatch_router_logits(), combine()
```

#### NVSHMEM 内存管理

```
初始化（懒加载，首次 forward 时触发）：

1. init_nvshmem_by_torch_process_group(ep_group)
   └── 初始化 NVSHMEM runtime，建立 rank 间的 symmetric heap

2. nvshmem_create_tensor([max_total_tokens, hidden], bf16)
   └── AllGather 的 symmetric buffer
   └── 所有 rank 共享同一地址空间，每个 rank 写自己的 slot

3. create_reduce_scater_2d_ctx(max_M, N, rank, world_size, ...)
   └── ReduceScatter 上下文，预分配：
       - scatter_bufs: 节点内 scatter 缓冲区
       - rs_per_node_bufs: 节点间 reduce-scatter 缓冲区
       - signal_bufs: 同步信号
       - reduction_stream: 独立 CUDA stream 用于重叠 reduction
```

#### AllGather 实现 (_allgather_nvshmem)

```python
def _allgather_nvshmem(self, tensors, sizes):
    # 1. 计算本 rank 在 gathered tensor 中的 offset
    offset = sum(sizes[:my_rank])

    for tensor in tensors:
        # 2. 所有 rank 的 symmetric buffer 有相同的虚拟地址
        buf = self._ag_symm_buf[:total_tokens, :hidden]

        # 3. 重置同步信号
        self._ag_signal.zero_()
        nvshmem_barrier_all_on_stream(stream)

        # 4. 每个 rank 写入自己的 slot（本地写，但其他 rank 可见）
        buf[offset : offset + sizes[my_rank]].copy_(tensor)

        # 5. 通知其他 rank 数据已就绪
        _set_signal_cuda(self._ag_signal[my_rank], 1, stream)

        # 6. 等待所有 rank 完成写入
        nvshmem_barrier_all_on_stream(stream)

        # 7. 读取完整的 gathered tensor（零拷贝，直接从 symmetric heap 读）
        gathered = buf[:total_tokens].clone()
```

**与 NCCL AllGatherv 的区别**：
- NCCL：数据经过 NVLink/PCIe 拷贝到每个 rank 的独立 buffer
- NVSHMEM：数据写入共享 symmetric heap，其他 rank 直接读取（single-copy）

#### ReduceScatter 实现

```python
def combine(self, hidden_states, ...):
    # 1. Padding: RS 要求 M % world_size == 0
    padded_tokens = ceil(total_tokens / world_size) * world_size
    if padded_tokens != total_tokens:
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad_size))

    # 2. 调用 Triton-distributed 的优化 RS
    output = reduce_scatter_2d_op(hidden_states, self._rs_ctx)
    #   内部流程：
    #   a. nvshmem_barrier_all_on_stream (全局同步)
    #   b. intra-node scatter (通过 NVSHMEM put)
    #   c. inter-node p2p (如果多节点)
    #   d. local reduction (在独立 stream 上与通信重叠)
    #   e. reset barriers

    # 3. 去掉 padding
    return output[:my_size]
```

**与 NCCL ReduceScatterv 的区别**：
- NCCL：通过 ring/tree 算法在 NVLink 上传输 + 归约
- Triton-dist：SM 级别的 reduction 与 NVSHMEM scatter 在不同 stream 上重叠

### 生命周期管理

```
进程启动
    │
    ▼ (首次 forward)
_ensure_nvshmem()  →  init_nvshmem_by_torch_process_group()
_ensure_buffers()  →  nvshmem_create_tensor() + create_reduce_scater_2d_ctx()
                  →  atexit.register(destroy)
    │
    ▼ (steady state)
dispatch() / combine() 复用预分配的 NVSHMEM buffer
    │
    ▼ (进程退出)
atexit → destroy()  →  nvshmem_free_tensor_sync()
```

## 全部修改文件

| 文件 | 改动 | 说明 |
|------|------|------|
| **新建** `vllm/distributed/device_communicators/triton_dist_all2all.py` | 新建 | `TritonDistAll2AllManager` 核心实现 |
| `vllm/config/parallel.py` | 修改 | `All2AllBackend` 添加 `"triton_distributed_ag_rs"` + 验证逻辑 |
| `vllm/config/compilation.py` | 修改 | PIECEWISE CUDA Graph 支持（与 monolithic 共享逻辑） |
| `vllm/model_executor/layers/fused_moe/config.py` | 修改 | `use_ag_rs_all2all_kernels` 包含新后端；添加 `use_triton_dist_ag_rs_kernels` |
| `vllm/distributed/device_communicators/cuda_communicator.py` | 修改 | 注册 `TritonDistAll2AllManager` |

**无需修改的文件**（这是 modular 方案的核心优势）：
- `prepare_finalize/naive_dp_ep.py` — 已有的 AG/RS prepare/finalize **原封不动复用**
- `runner/default_moe_runner.py` — runner 逻辑不变
- `oracle/unquantized.py` — expert kernel 选择不变
- `unquantized_fused_moe_method.py` — 不需要 monolithic 路径

## 使用方式

```bash
# 4卡节点内 DP+EP
vllm serve <moe-model> \
  --tensor-parallel-size 1 \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --all2all-backend triton_distributed_ag_rs

# 8卡节点内 DP+EP
vllm serve <moe-model> \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --all2all-backend triton_distributed_ag_rs
```

## 与现有后端的对比

```
                          通信实现          expert计算     CUDA Graph    量化支持
                          ─────────        ──────────     ──────────   ────────
allgather_reducescatter   NCCL AG/RS       vLLM标准       FULL         全部
triton_distributed        NVSHMEM mega     融合GroupGEMM  PIECEWISE    仅BF16
triton_distributed_ag_rs  NVSHMEM AG/RS    vLLM标准       PIECEWISE    全部
deepep_high_throughput    DeepEP buffer    vLLM标准       NONE         FP8
deepep_low_latency        DeepEP LL        vLLM标准       FULL         FP8
```

## 当前限制

- 仅支持节点内 EP（<=8 GPUs via NVLink）
- AG buffer 使用 BF16 固定精度（不影响 expert 计算的量化）
- ReduceScatter 要求 `total_tokens % world_size == 0`（自动 padding 处理）
- AllGather 实现当前是 barrier-based（每个 tensor 两次 barrier），可优化为 signal-wait 模式
- 不兼容完整 CUDA Graph（使用 PIECEWISE 模式缓解）

## 后续优化方向

1. **AllGather 优化**：将 barrier-based 同步替换为 per-rank signal-wait，减少同步开销
2. **AG+GEMM 融合**：使用 `allgather_group_gemm.py` 的 `ag_group_gemm()` 将 AllGather 与 W1 GEMM 重叠
3. **GEMM+RS 融合**：使用 `moe_reduce_rs.py` 的 `run_moe_reduce_rs()` 将 W2 GEMM 与 ReduceScatter 重叠
4. **多 tensor 批量 AllGather**：当前对每个 tensor（hidden_states、router_logits）分别做 AG，可以打包成一次通信
5. **FP8 dispatch**：在 AllGather 前量化、传输后反量化，减少通信量
