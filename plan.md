# Triton-Distributed 细粒度通信计算重叠集成到 vLLM EP MoE

## 背景

vLLM 的 MoE EP 推理采用三阶段流水线：**Dispatch(All2All) → Expert Compute(GEMM) → Combine(All2All)**。现有后端（DeepEP、FlashInfer 等）将通信和计算作为独立 kernel 启动，依赖 CUDA stream 级别异步或 DBO 微批实现重叠。

Triton-distributed（ByteDance Seed）提供了 **kernel 级融合**——通信（NVSHMEM put/signal）和计算（Group GEMM）在同一个 Triton mega-kernel 中执行，实现 tile 级别的交错，消除 kernel 启动开销。

核心 API：
- `EpAll2AllFusedOp.mega_dispatch_group_gemm()` — 融合 A2A dispatch + W1 GEMM
- `EpAll2AllFusedOp.mega_group_gemm_combine()` — 融合 W2 GEMM + A2A combine

## 设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 集成架构 | Monolithic（非 Modular 拆分） | mega kernel 将通信和 GEMM 融合在同一 kernel 内，无法拆分到 prepare/experts/finalize |
| 部署规模 | 仅节点内 EP<=8（NVLink） | Triton-distributed 当前仅支持 intra-node |
| 精度 | BF16 先行 | 最快集成路径，FP8 需修改 kernel |
| DBO 兼容 | 不需要 | mega kernel 内部自带 SM 级重叠 |

## 架构概览

### 数据流

```
hidden_states [M, K] + router_logits [M, num_experts]
    │
    ▼ (路由: softmax + topk)
routing_weights [M, topk], selected_experts [M, topk]
    │
    ▼ (ep_op.preprocess: 计算 A2A 布局)
ep_a2a_layout_desc
    │
    ▼ (mega_dispatch_group_gemm: 融合 A2A dispatch + W1 GEMM)
    │   ┌──────────────────────────────────────────┐
    │   │ SM 分区: 80 SMs 做 dispatch 通信          │
    │   │          剩余 SMs 做 Group GEMM (W1)      │
    │   │ NVSHMEM putmem + signal 与 GEMM tiles 交错│
    │   └──────────────────────────────────────────┘
fc1_output [local_tokens, intermediate*2]
    │
    ▼ (SwiGLU activation + weight scaling)
swiglu_output [local_tokens, intermediate]
    │
    ▼ (mega_group_gemm_combine: 融合 W2 GEMM + A2A combine)
    │   ┌──────────────────────────────────────────┐
    │   │ Group GEMM (W2) + reduce + A2A combine   │
    │   │ SM 分区: GEMM SMs + reduce SMs + comm SMs│
    │   └──────────────────────────────────────────┘
output [M, K] (已 reduce，回到原始 token 布局)
```

### 集成路径

采用 vLLM 现有的 monolithic 模式（与 FlashInfer TRT-LLM 类似）：
- `UnquantizedFusedMoEMethod` 标记为 `is_monolithic=True`
- `apply_monolithic(layer, x, router_logits)` 直接调用 `triton_dist_ep_forward()`
- 整个 routing → dispatch+W1 → activation → W2+combine 在一个函数内完成
- Runner 的 `_maybe_dispatch`/`_maybe_combine`（naive AllGather/ReduceScatter）被跳过

## 全部修改清单

### 1. 配置注册

**[vllm/config/parallel.py](vllm/config/parallel.py)**
- `All2AllBackend` 类型添加 `"triton_distributed"`
- `use_sequence_parallel_moe` 属性添加 `"triton_distributed"`
- 添加验证：`triton_distributed` 要求 `--enable-expert-parallel` 且 EP<=8

**[vllm/model_executor/layers/fused_moe/config.py](vllm/model_executor/layers/fused_moe/config.py)**
- `FusedMoEParallelConfig` 添加 `use_triton_dist_kernels` 属性
- `FusedMoEConfig` 添加同名代理属性

**[vllm/utils/import_utils.py](vllm/utils/import_utils.py)**
- 添加 `has_triton_dist()` 检测函数

**[vllm/config/compilation.py](vllm/config/compilation.py)**
- 禁用 CUDA Graphs（NVSHMEM 不兼容）

### 2. 核心实现

**[vllm/model_executor/layers/fused_moe/prepare_finalize/triton_dist_ep.py](vllm/model_executor/layers/fused_moe/prepare_finalize/triton_dist_ep.py)** (新建)

核心文件，包含：

- `TritonDistEPState` 类：
  - NVSHMEM 懒初始化管理（`threading.Lock` 保护）
  - 显式 `init_nvshmem_by_torch_process_group()` 确保 NVSHMEM runtime 先于 EpAll2AllFusedOp 初始化
  - NVSHMEM 初始化失败的 retry 逻辑（`invalidate()` + 单次重试）
  - `atexit` 注册 `shutdown()` 清理 NVSHMEM 资源，防止进程退出时 implicit free 报错
  - `shutdown()` 调用 `deinit_triton_dist_ep_op()` + `finalize_distributed()`

- `triton_dist_ep_forward()` 函数：
  - 完整融合流水线：routing → `mega_dispatch_group_gemm` → SwiGLU → `mega_group_gemm_combine`
  - combine output shape 归一化（处理 `[M*topk, K]` → `[M, K]` 的情况）

### 3. Oracle + Method 集成

**[vllm/model_executor/layers/fused_moe/oracle/unquantized.py](vllm/model_executor/layers/fused_moe/oracle/unquantized.py)**
- `UnquantizedMoeBackend` 枚举添加 `TRITON_DISTRIBUTED`
- 加入 `UNSUPPORTED_BACKEND` 列表（绕过 modular kernel 框架）
- `select_unquantized_moe_backend()` 优先检测 `use_triton_dist_kernels`

**[vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py](vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py)**
- `_is_monolithic` 条件添加 `TRITON_DISTRIBUTED`
- `_select_monolithic()` 返回 `forward_monolithic_triton_dist`
- 新增 `forward_monolithic_triton_dist()` 方法
- `maybe_make_prepare_finalize()` 对 `TRITON_DISTRIBUTED` 返回 `None`
- **关键**：`supports_internal_mk` 对 `TRITON_DISTRIBUTED` 返回 `True`，阻止 runner 做 naive AllGather/ReduceScatter

### 4. EP 通信管理

**[vllm/distributed/device_communicators/cuda_communicator.py](vllm/distributed/device_communicators/cuda_communicator.py)**
- 添加 `triton_distributed` 分支，创建 `NaiveAll2AllManager` 占位
- EP communicator 初始化不会因缺少 all2all_manager 而失败

## 踩过的坑与修复

### Bug 1: max_tokens_per_rank 缓冲区溢出

**现象**：`RuntimeError: The size of tensor a (256) must match the size of tensor b (8192)`

**根因**：`EpAll2AllFusedOp` 按 `max_tokens` 分配 NVSHMEM buffer `[nnodes, max_tokens, topk]`。最初用了 `moe_config.max_num_tokens`（= `VLLM_MOE_DP_CHUNK_SIZE` = 256），实际 batch 远超此值。

**修复**：改用 `layer.vllm_config.scheduler_config.max_num_batched_tokens`。

### Bug 2: get_current_vllm_config() 上下文不可用

**现象**：`AssertionError: Current vLLM config is not set`

**根因**：`get_current_vllm_config()` 依赖线程局部的 config context，在 Worker 进程的 forward 阶段此 context 不一定被设置。

**修复**：改用 `layer.vllm_config`（在 `FusedMoE.__init__()` 时已保存到实例属性）。

### Bug 3: Runner 做了多余的 AllGather 导致 token 数膨胀

**现象**：修复 Bug 1 后仍报 `tensor a (2048) must match tensor b (8192)`

**根因**：Runner 的 `do_naive_dispatch_combine = True`（因为 `supports_internal_mk=False`），在 monolithic forward 之前做了 AllGather，把 4 个 DP rank 的 tokens 合在一起（2048×4=8192）。然后 Triton-distributed 又试图做自己的 A2A dispatch——双重通信且 buffer 溢出。

**修复**：让 `UnquantizedFusedMoEMethod.supports_internal_mk` 对 `TRITON_DISTRIBUTED` 返回 `True`，阻止 runner 做 naive dispatch/combine。Triton-distributed 的 mega kernel 自己处理 EP All2All 通信。

### Bug 4: CudaCommunicator 缺少 all2all_manager

**现象**：EP group 初始化失败

**根因**：`CudaCommunicator` 没有 `triton_distributed` 的分支来创建 all2all_manager。

**修复**：添加分支创建 `NaiveAll2AllManager` 占位。

### Bug 5: NVSHMEM 未预先初始化

**现象**：`NVSHMEM Library is not initialized` 或 buffer 分配时段错误

**根因**：`init_triton_dist_ep_op()` 内部创建 `EpAll2AllFusedOp`，其 NVSHMEM tensor 需要 NVSHMEM runtime 已初始化。

**修复**：在 `init_triton_dist_ep_op()` 之前显式调用 `init_nvshmem_by_torch_process_group(ep_group)`。

### Bug 6: Ctrl+C 后 NVSHMEM 内存泄漏

**现象**：进程退出时大量 `NvshmemError: Buffer freed implicitly`，重启可能异常

**根因**：Python weakref finalizer 在进程退出时尝试释放 NVSHMEM buffer，但 CUDA context 已被破坏。

**修复**：`atexit.register(self.shutdown)` 在解释器退出前主动调用 `deinit_triton_dist_ep_op()` + `finalize_distributed()`，优雅释放 NVSHMEM 资源。

## 使用方式

```bash
# 8卡节点内 EP（DP=8, EP=8）
vllm serve <moe-model> \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --all2all-backend triton_distributed

# 4卡节点内 EP
vllm serve <moe-model> \
  --tensor-parallel-size 1 \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --all2all-backend triton_distributed
```

环境要求：
- `triton_dist` 包已安装（从 `Triton-distributed/` 目录 `pip install -e .`）
- `triton >= 3.1.0`（vLLM 自身的 Triton kernel 需要）
- NVSHMEM runtime（由 `triton_dist` 依赖自动安装）
- `NVSHMEM_SYMMETRIC_SIZE` 环境变量会被自动设置（`init_triton_dist_ep_op` 内部处理）

## 当前限制

- 仅支持节点内 EP（<=8 GPUs via NVLink）
- 仅支持 BF16 精度（不支持 FP8 量化 dispatch）
- 仅支持标准 TopK routing（不支持 GroupedTopK）
- 不兼容 CUDA Graphs
- 不兼容 DBO（Dual Batch Overlap）
- 不兼容 EPLB（Expert Parallelism Load Balancing）
- 激活函数硬编码为 SwiGLU（覆盖 Qwen3 MoE、DeepSeek 等主流模型）

## 关键文件索引

| 文件 | 职责 |
|------|------|
| `vllm/config/parallel.py` | backend 类型 + 验证 |
| `vllm/config/compilation.py` | 禁用 CUDA Graphs |
| `vllm/model_executor/layers/fused_moe/config.py` | `use_triton_dist_kernels` 属性 |
| `vllm/utils/import_utils.py` | `has_triton_dist()` |
| `vllm/model_executor/layers/fused_moe/oracle/unquantized.py` | backend 枚举 + 选择逻辑 |
| `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py` | monolithic 方法 + `supports_internal_mk` |
| **`vllm/model_executor/layers/fused_moe/prepare_finalize/triton_dist_ep.py`** | **核心：forward + NVSHMEM 生命周期** |
| `vllm/distributed/device_communicators/cuda_communicator.py` | EP communicator 占位 |

**Triton-distributed 侧关键引用：**

| 文件 | 用途 |
|------|------|
| `Triton-distributed/.../ep_a2a_fused_layer.py` | `EpAll2AllFusedOp` 类 |
| `Triton-distributed/.../common.py` | `init_triton_dist_ep_op()`, `TritonDistEpContext` |
| `Triton-distributed/.../ep_moe_fused.py` | 参考实现：`TritonDistFusedEpMoeFunction` |
| `Triton-distributed/.../group_gemm.py` | `build_block_row_idx_info_kernel` |
| `Triton-distributed/.../swiglu.py` | `swiglu_forward` |
| `Triton-distributed/.../utils.py` | `init_nvshmem_by_torch_process_group`, `finalize_distributed` |
