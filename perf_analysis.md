# Triton-Distributed EP 性能分析

## 测试配置

- **Baseline**：vLLM 默认 `--all2all-backend allgather_reducescatter`（DP+EP，Triton kernel + NCCL AllGather/ReduceScatter）
- **实验组**：`--all2all-backend triton_distributed`（Triton-distributed mega kernel + NVSHMEM）
- **共同**：CUDA Graphs 在 baseline 中**开启**，在 triton_distributed 中**被强制关闭**

## 性能差距来源分析（按影响大小排序）

---

### P0: CUDA Graphs 被关闭 — 预估影响 30-50%+ (decode)

**这是最大的性能杀手，尤其对 decode 吞吐量。**

```python
# compilation.py
if all2all_backend == "triton_distributed":
    self.cudagraph_mode = CUDAGraphMode.NONE
```

**影响机制**：
- vLLM baseline 对 decode（小 batch）使用 CUDA Graph 捕获整个 forward pass
- CUDA Graph 消除了每个 kernel 的 CPU launch overhead（~5-10us/kernel）
- 一个 transformer layer 有 ~10-20 个 kernel launch，28 层 = 280-560 次
- 不用 CUDA Graph 时，每次 decode step 额外 ~1-5ms 纯 CPU overhead
- Decode 本身单步只需 ~5-15ms（取决于 batch size），所以 CPU overhead 占比极高

**对比**：Baseline 用 `allgather_reducescatter` 时 CUDA Graph 正常开启。NCCL collective 可以被 CUDA Graph 捕获，但 NVSHMEM 的 `putmem_nbi`/`signal_op` 不能。

**优化方向**：
- 短期：对 Triton-distributed 使用 PIECEWISE CUDA Graph——只把 MoE 层排除在 graph 之外，attention 等其他层仍然捕获
- 长期：研究 NVSHMEM ops 是否可以被 CUDA Graph 捕获（可能需要 NVSHMEM 侧支持）

---

### P1: 每次 forward 重新创建 TritonDistEpContext — 预估影响 5-15%

```python
# triton_dist_ep_forward() — 每次 forward 调用
ctx = _triton_dist_state.get_context(ep_group, top_k, num_experts)

# init_triton_dist_ep_ctx() 内部：
triton_dist_ep_ctx = TritonDistEpContext(ep_group, triton_dist_ep_op, ...)
# TritonDistEpContext.__init__() 分配 7 个 GPU tensor：
self.split_size_cum_per_expert = torch.empty([num_experts_per_rank], ...)
self.expert_ids = torch.empty([max_num_tiles], ...)
self.split_size_cum = torch.empty([max_num_tiles], ...)
self.tile_num = torch.empty([max_num_tiles], ...)
self.tile_num_cum = torch.empty([max_num_tiles], ...)
self.expert_tile_offset = torch.empty([num_experts_per_rank], ...)
self.num_tiles_total = torch.empty([1], ...)
```

**每次 forward 都分配 7 个 GPU tensor**，然后下一次 forward 又被 GC。这产生：
- CUDA malloc/free 开销（每个 ~1-5us，×7 = 7-35us）
- 更严重的是 **PyTorch CUDA memory allocator 碎片化** 和 **GC 压力**
- 28 层 × 每层 7 个 tensor = 每个 forward step 分配/释放 196 个临时 tensor

**对比**：Baseline 的 modular kernel 在初始化时预分配 workspace，forward 时复用。

**修复**：缓存 `TritonDistEpContext`，只创建一次，每次 forward 复用：
```python
def get_context(self, ep_group, top_k, num_experts):
    if self._ep_ctx is None:
        self._ep_ctx = init_triton_dist_ep_ctx(...)
    return self._ep_ctx
```

---

### P2: preprocess() 中的 CPU-GPU 同步和 AllGather — 预估影响 5-20%

```python
# ep_a2a_fused_layer.py preprocess()
# 1. NVSHMEM AllGather on expert indices (cross-GPU sync!)
get_ag_splits_and_recv_offset_for_dispatch(...)
# 内部调用 nvshmem_barrier_all_on_stream，强制所有 GPU 同步
```

**影响机制**：
- `preprocess()` 在每次 forward 中对 expert indices 做跨 GPU 的 AllGather
- 这是一个 **同步点** — 所有 GPU 必须等待最慢的那个完成
- Baseline 的 `allgather_reducescatter` 也有类似的 AllGather，但它通过 NCCL 做，与 CUDA Graph 配合更好
- 关键区别：Triton-distributed 的 `preprocess()` 还包含多次 `bincount`、`copy_` 等小 kernel，每个都有 launch overhead

---

### P3: Python 侧 routing 逻辑 — 预估影响 3-10%

Baseline 的 routing 在 C++/Triton kernel 内完成（`select_experts` 调用），我们的实现在 Python 中做：

```python
# triton_dist_ep_forward() — 纯 Python/PyTorch ops
routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
selected_experts = selected_experts.to(torch.int32)

# scatter indices 计算 — 两次 argsort！
local_scatter_indices = (
    selected_experts.flatten()
    .argsort(stable=True)
    .argsort()
    .int()
    .view(selected_experts.shape)
)
```

**问题**：
- `argsort().argsort()` 是 O(N·K·log(N·K)) 的操作，且产生 2 次 kernel launch
- `softmax` + `topk` + `div` + `to` + `flatten` + `argsort` + `argsort` + `int` + `view` = **9 个独立 kernel launch**
- Baseline 的 `select_experts` 是一个融合的 Triton kernel

---

### P4: `get_moe_optim_config()` 每次调用 `torch.cuda.get_device_properties()` — 影响较小但可消除

```python
# common.py — 每次 forward 调用
def get_moe_optim_config(use_mega=True):
    max_sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    ...
```

`get_device_properties()` 不是零开销的。每层每个 forward step 调用一次。

**修复**：缓存结果。

---

### P5: combine output shape 归一化 — 影响极小

```python
if combine_output.shape[0] == expected_tokens * top_k:
    combine_output = combine_output.view(expected_tokens, top_k, -1).sum(dim=1)
```

如果这个分支被触发（warning_once 显示），那意味着 combine kernel **没有正确做 reduction**，我们在 Python 侧补了一个 `sum`。这本身应该在 kernel 内完成。

---

### P6: SwiGLU 作为独立 kernel — 可优化空间

```python
swiglu_output, _ = swiglu_forward(fc1_output, scale=dispatch_weight_in_buf.view(-1))
```

在 `mega_dispatch_group_gemm` 和 `mega_group_gemm_combine` 之间，SwiGLU 是一个独立的 kernel launch。理论上可以融合进 mega kernel。

**对比**：Baseline 的 Triton MoE kernel 将 activation 融合在 expert GEMM kernel 内。

---

### P7: Shared Experts 无法并行 — 影响因模型而异

Baseline 的 modular path 支持在独立 CUDA stream 上并行运行 shared experts：

```python
# default_moe_runner.py
if self.use_shared_experts_stream:
    self.shared_experts_stream.wait_stream(current_stream())
    with torch.cuda.stream(self.shared_experts_stream):
        shared_output = self.shared_experts(hidden_states)
```

Monolithic path 不支持这个优化。对 Qwen3-MoE、DeepSeek-V2 等有 shared experts 的模型，会有额外的串行等待。

---

## 总结：端到端性能差距来源

| 来源 | 预估影响 | 难度 | 状态 |
|------|---------|------|------|
| **CUDA Graphs 关闭** | 30-50%+ (decode) | 中 | 可用 piecewise CG 缓解 |
| **Context 每次重建（7 tensor alloc）** | 5-15% | 低 | 简单缓存即可 |
| **preprocess 跨 GPU 同步** | 5-20% | 高 | Triton-dist 架构限制 |
| **Python routing + 2x argsort** | 3-10% | 中 | 可移入 Triton kernel |
| **OptimConfig 重复查询** | <1% | 低 | 缓存 |
| **SwiGLU 独立 kernel** | 1-3% | 高 | 需改 Triton-dist |
| **Shared experts 无法并行** | 模型相关 | 中 | 需改架构 |

---

## 优先级排序的优化建议

### 立即可做（1-2 天）

1. **缓存 TritonDistEpContext** — `get_context()` 只创建一次，复用
2. **缓存 MoEOptimConfig** — 全局只计算一次
3. **尝试 PIECEWISE CUDA Graph** — 修改 compilation.py，不完全禁用 CG，只排除 MoE custom op

### 短期（1 周）

4. **将 routing 移入 preprocess** — 避免 Python 侧 9 个独立 kernel
5. **消除 combine output 的 shape 归一化** — 确保 combine kernel 直接输出 `[M, K]`

### 中期（2-4 周）

6. **研究 NVSHMEM + CUDA Graph 兼容性** — 是否可以在 graph capture 中包含 NVSHMEM ops
7. **Shared experts stream overlap 支持** — 在 monolithic 路径中恢复 stream overlap

---

## 补充：为什么 mega kernel 的融合优势没体现出来？

Triton-distributed 的核心卖点是 **通信和计算在同一 kernel 内 tile 级别重叠**。但在当前集成中，这个优势被以下因素抵消：

1. **Decode 场景 token 数极少（1-32）**：mega kernel 的优势在大 batch 时才明显（通信延迟可以被大量 GEMM tiles 隐藏）。小 batch 时通信延迟无法被有效隐藏，反而多了 NVSHMEM 的额外初始化和同步开销。

2. **CUDA Graph 的缺失放大了所有其他开销**：Baseline 的 "通信和计算分离" 方案虽然没有 tile 级别重叠，但 CUDA Graph 消除了所有 kernel launch overhead，使得端到端延迟非常紧凑。我们的方案虽然 kernel 内部更高效，但 kernel 外面的 Python/CPU overhead 把优势全吃了。

3. **Prefill 大 batch 场景才是 mega kernel 的主战场**：建议优先在 prefill-heavy workload（长 prompt、大 batch）下做 benchmark，而非 decode-heavy 的交互式场景。

---

## 推荐的 Benchmark 方案

```bash
# Prefill-heavy（mega kernel 优势场景）
python benchmarks/benchmark_throughput.py \
  --model <moe-model> \
  --input-len 4096 --output-len 16 \
  --num-prompts 64 \
  --enable-expert-parallel \
  --all2all-backend triton_distributed

# Decode-heavy（当前劣势场景）
python benchmarks/benchmark_throughput.py \
  --model <moe-model> \
  --input-len 128 --output-len 512 \
  --num-prompts 64 \
  --enable-expert-parallel \
  --all2all-backend triton_distributed

# 对比 baseline
python benchmarks/benchmark_throughput.py \
  --model <moe-model> \
  --input-len 4096 --output-len 16 \
  --num-prompts 64 \
  --enable-expert-parallel \
  --all2all-backend allgather_reducescatter
```

建议使用 `nsys profile` 抓 GPU trace 来精确定位各阶段耗时。
