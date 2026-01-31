# vLLM Sleep Mode 实现机制详解

## 概述

vLLM Sleep Mode 是一个用于**临时释放GPU显存**的功能，允许在不停止服务或卸载容器的情况下，释放大部分被模型占用的显存（包括模型权重和KV缓存）。这个功能特别适用于以下场景：

- **RLHF训练**：训练和推理交替进行，需要动态释放/恢复显存
- **多模型切换**：在同一GPU上运行多个不能同时加载的模型
- **成本优化**：在推理工作负载之间释放GPU资源

### 核心优势

1. **显存释放率高**：可释放90%+的GPU显存
2. **恢复速度快**：比冷启动快18-200倍
3. **支持分布式**：兼容TP（Tensor Parallelism）、PP（Pipeline Parallelism）、EP（Expert Parallelism）
4. **细粒度控制**：可选择性恢复权重或KV缓存

## 架构概览

Sleep Mode 的实现涉及多个层级的组件协作：

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户接口层                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   HTTP API      │  │   Python API    │  │   LLM Class     │ │
│  │ /sleep /wake_up │  │ engine.sleep()  │  │ llm.sleep()     │ │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘ │
└───────────┼────────────────────┼────────────────────┼───────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                        引擎层 (Engine Layer)                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ AsyncLLM / LLMEngine                                        ││
│  │   - sleep(level) → EngineCore.sleep()                       ││
│  │   - wake_up(tags) → EngineCore.wake_up()                    ││
│  │   - is_sleeping() → EngineCore.is_sleeping()                ││
│  └────────────────────────────┬────────────────────────────────┘│
└───────────────────────────────┼─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        执行器层 (Executor Layer)                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Executor (abstract.py:301-339)                              ││
│  │   - is_sleeping: bool                                       ││
│  │   - sleeping_tags: set[str]                                 ││
│  │   - sleep() → collective_rpc("sleep")                       ││
│  │   - wake_up() → collective_rpc("wake_up")                   ││
│  └────────────────────────────┬────────────────────────────────┘│
└───────────────────────────────┼─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Worker层 (Worker Layer)                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ GPU Worker (gpu_worker.py:113-159)                          ││
│  │   - sleep() → CuMemAllocator.sleep()                        ││
│  │   - wake_up() → CuMemAllocator.wake_up()                    ││
│  │   - _sleep_saved_buffers: dict (用于保存level 2的buffers)    ││
│  └────────────────────────────┬────────────────────────────────┘│
└───────────────────────────────┼─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     内存分配器层 (Memory Allocator)                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ CuMemAllocator (cumem.py:87-302)                            ││
│  │   - 基于CUDA虚拟内存管理                                      ││
│  │   - pointer_to_data: dict[int, AllocationData]              ││
│  │   - sleep() → cudaMemcpy to CPU + unmap_and_release         ││
│  │   - wake_up() → create_and_map + cudaMemcpy from CPU        ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## 两种睡眠级别

### Level 1: 权重卸载模式

**适用场景**：使用相同模型进行推理，中间需要释放显存给其他任务

**工作原理**：
- 将模型权重从GPU复制到CPU pinned memory
- 丢弃KV缓存内容
- 释放GPU显存

```
┌─────────────────────┐     Level 1 Sleep      ┌─────────────────────┐
│     GPU Memory      │ ──────────────────────►│     CPU Memory      │
│                     │                        │                     │
│  ┌───────────────┐  │    cudaMemcpy          │  ┌───────────────┐  │
│  │Model Weights  │──┼────────────────────────┼─►│ CPU Backup    │  │
│  └───────────────┘  │                        │  │ (pinned mem)  │  │
│                     │                        │  └───────────────┘  │
│  ┌───────────────┐  │    Discarded           │                     │
│  │  KV Cache     │──┼────────X               │                     │
│  └───────────────┘  │                        │                     │
└─────────────────────┘                        └─────────────────────┘
```

### Level 2: 完全丢弃模式

**适用场景**：加载不同模型或RLHF权重更新，不需要原有权重

**工作原理**：
- 丢弃模型权重（不备份到CPU）
- 丢弃KV缓存内容
- 仅保存模型的buffers（如RoPE缩放张量）到CPU
- 最大程度减少CPU内存压力

```
┌─────────────────────┐     Level 2 Sleep      ┌─────────────────────┐
│     GPU Memory      │ ──────────────────────►│     CPU Memory      │
│                     │                        │                     │
│  ┌───────────────┐  │    Discarded           │                     │
│  │Model Weights  │──┼────────X               │                     │
│  └───────────────┘  │                        │                     │
│                     │                        │  ┌───────────────┐  │
│  ┌───────────────┐  │    Saved (buffers)     │  │Model Buffers  │  │
│  │Model Buffers  │──┼────────────────────────┼─►│(RoPE scales)  │  │
│  └───────────────┘  │                        │  └───────────────┘  │
│                     │                        │                     │
│  ┌───────────────┐  │    Discarded           │                     │
│  │  KV Cache     │──┼────────X               │                     │
│  └───────────────┘  │                        │                     │
└─────────────────────┘                        └─────────────────────┘
```

## 核心组件实现详解

### 1. CuMemAllocator - 核心内存管理器

**文件位置**: `vllm/device_allocator/cumem.py`

这是sleep mode最核心的组件，使用PyTorch的可插拔分配器（Pluggable Allocator）机制来管理CUDA虚拟内存。

#### 关键数据结构

```python
# cumem.py:52-57
@dataclasses.dataclass
class AllocationData:
    handle: HandleType  # (device, alignedSize, d_mem, p_memHandle)
    tag: str            # 标签，如 "weights" 或 "kv_cache"
    cpu_backup_tensor: torch.Tensor | None = None  # CPU备份数据
```

#### 单例模式设计

```python
# cumem.py:87-125
class CuMemAllocator:
    """
    A singleton class that manages a memory pool for CUDA tensors.
    The memory in this pool can be offloaded or discarded when the
    allocator sleeps.
    """
    instance: "CuMemAllocator" = None
    default_tag: str = "default"

    @staticmethod
    def get_instance() -> "CuMemAllocator":
        if CuMemAllocator.instance is None:
            CuMemAllocator.instance = CuMemAllocator()
        return CuMemAllocator.instance
```

**为什么需要单例？** 当分配的tensor被垃圾回收时，PyTorch会调用free回调函数。C扩展使用全局变量存储此类实例的函数。如果创建多个实例，全局变量会被覆盖，free回调将无法正常工作。

#### Sleep 实现

```python
# cumem.py:175-223
def sleep(self, offload_tags: tuple[str, ...] | str | None = None) -> None:
    """
    Put the allocator in sleep mode.
    All data in the memory allocation with the specified tag will be
    offloaded to CPU memory, and others will be discarded.
    """
    if offload_tags is None:
        # 默认情况下，分配的张量在睡眠时会被卸载
        offload_tags = (CuMemAllocator.default_tag,)
    elif isinstance(offload_tags, str):
        offload_tags = (offload_tags,)

    total_bytes = 0
    backup_bytes = 0

    for ptr, data in self.pointer_to_data.items():
        handle = data.handle
        total_bytes += handle[1]
        if data.tag in offload_tags:
            # 备份到CPU
            backup_bytes += handle[1]
            size_in_bytes = handle[1]
            cpu_backup_tensor = torch.empty(
                size_in_bytes,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=is_pin_memory_available(),  # 使用pinned memory加速传输
            )
            cpu_ptr = cpu_backup_tensor.data_ptr()
            libcudart.cudaMemcpy(cpu_ptr, ptr, size_in_bytes)  # GPU → CPU
            data.cpu_backup_tensor = cpu_backup_tensor
        # 释放GPU虚拟内存映射
        unmap_and_release(handle)

    logger.info(
        "CuMemAllocator: sleep freed %.2f GiB memory in total, of which "
        "%.2f GiB is backed up in CPU and the rest %.2f GiB is discarded "
        "directly.",
        total_bytes / 1024**3,
        backup_bytes / 1024**3,
        (total_bytes - backup_bytes) / 1024**3,
    )

    gc.collect()
    torch.cuda.empty_cache()
```

**关键点**：
1. 使用`cudaMemcpy`直接复制数据到CPU pinned memory
2. 调用`unmap_and_release`释放GPU虚拟内存映射
3. 最后调用`gc.collect()`和`torch.cuda.empty_cache()`确保内存完全释放

#### Wake Up 实现

```python
# cumem.py:225-247
def wake_up(self, tags: list[str] | None = None) -> None:
    """
    Wake up the allocator from sleep mode.
    All data that is previously offloaded will be loaded back to GPU
    memory, and the rest of the data will have empty memory.
    """
    for ptr, data in self.pointer_to_data.items():
        if tags is None or data.tag in tags:
            handle = data.handle
            # 重新创建GPU内存映射
            create_and_map(handle)
            if data.cpu_backup_tensor is not None:
                cpu_backup_tensor = data.cpu_backup_tensor
                if cpu_backup_tensor is not None:
                    size_in_bytes = (
                        cpu_backup_tensor.numel() * cpu_backup_tensor.element_size()
                    )
                    cpu_ptr = cpu_backup_tensor.data_ptr()
                    # CPU → GPU
                    libcudart.cudaMemcpy(ptr, cpu_ptr, size_in_bytes)
                    data.cpu_backup_tensor = None  # 释放CPU备份
```

#### 内存池上下文管理器

```python
# cumem.py:249-291
@contextmanager
def use_memory_pool(self, tag: str | None = None):
    """
    A context manager to use the memory pool.
    All memory allocation created inside the context will be allocated
    in the memory pool, and has the specified tag.
    """
    if tag is None:
        tag = CuMemAllocator.default_tag

    old_tag = self.current_tag
    self.current_tag = tag
    with use_memory_pool_with_allocator(
        self.python_malloc_callback, self.python_free_callback
    ) as data:
        self.allocator_and_pools[tag] = data
        yield
        # ... 清理逻辑
        self.current_tag = old_tag
```

这个上下文管理器用于在加载模型权重和初始化KV缓存时，为分配的内存打上特定标签。

### 2. GPU Worker 实现

**文件位置**: `vllm/v1/worker/gpu_worker.py`

Worker层负责协调CuMemAllocator和模型之间的交互。

#### Sleep 方法

```python
# gpu_worker.py:113-135
def sleep(self, level: int = 1) -> None:
    from vllm.device_allocator.cumem import CuMemAllocator

    free_bytes_before_sleep = torch.cuda.mem_get_info()[0]

    # Level 2时保存buffers到CPU
    if level == 2:
        model = self.model_runner.model
        self._sleep_saved_buffers = {
            name: buffer.cpu().clone() for name, buffer in model.named_buffers()
        }

    allocator = CuMemAllocator.get_instance()
    # Level 1: 仅卸载权重 (offload_tags=("weights",))
    # Level 2: 全部丢弃 (offload_tags=tuple())
    allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())

    free_bytes_after_sleep, total = torch.cuda.mem_get_info()
    freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
    used_bytes = total - free_bytes_after_sleep

    logger.info(
        "Sleep mode freed %s GiB memory, %s GiB memory is still in use.",
        format_gib(freed_bytes),
        format_gib(used_bytes),
    )
```

#### Wake Up 方法

```python
# gpu_worker.py:137-159
def wake_up(self, tags: list[str] | None = None) -> None:
    from vllm.device_allocator.cumem import CuMemAllocator

    allocator = CuMemAllocator.get_instance()
    allocator.wake_up(tags)

    # Level 2唤醒后恢复buffers
    if len(self._sleep_saved_buffers):
        model = self.model_runner.model
        for name, buffer in model.named_buffers():
            if name in self._sleep_saved_buffers:
                buffer.data.copy_(self._sleep_saved_buffers[name].data)
        self._sleep_saved_buffers = {}

    # 如果KV缓存刚被唤醒，需要重置FP8缩放因子
    if (
        (tags is None or "kv_cache" in tags)
        and self.cache_config.cache_dtype.startswith("fp8")
        and hasattr(self.model_runner, "init_fp8_kv_scales")
    ):
        self.model_runner.init_fp8_kv_scales()
```

#### 内存池上下文使用

```python
# gpu_worker.py:161-172
def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
    if self.vllm_config.model_config.enable_sleep_mode:
        from vllm.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
        if tag == "weights":
            assert allocator.get_current_usage() == 0, (
                "Sleep mode can only be used for one instance per process."
            )
        return allocator.use_memory_pool(tag=tag)
    else:
        return nullcontext()
```

在模型加载和KV缓存初始化时使用：

```python
# gpu_worker.py:273-278 - 加载模型
def load_model(self) -> None:
    with self._maybe_get_memory_pool_context(tag="weights"):
        self.model_runner.load_model()

# gpu_worker.py:408-415 - 初始化KV缓存
def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
    if self.vllm_config.model_config.enable_sleep_mode:
        allocator = CuMemAllocator.get_instance()
        with allocator.use_memory_pool(tag="kv_cache"):
            self.model_runner.initialize_kv_cache(kv_cache_config)
    else:
        self.model_runner.initialize_kv_cache(kv_cache_config)
```

### 3. Executor 层实现

**文件位置**: `vllm/v1/executor/abstract.py`

Executor负责管理分布式环境下的sleep/wake_up协调。

```python
# abstract.py:101-103
def __init__(self, vllm_config: VllmConfig) -> None:
    # ...
    self.is_sleeping = False
    self.sleeping_tags: set[str] = set()

# abstract.py:301-339
def sleep(self, level: int = 1):
    if self.is_sleeping:
        logger.warning("Executor is already sleeping.")
        return
    time_before_sleep = time.perf_counter()
    # 通过RPC调用所有worker的sleep方法
    self.collective_rpc("sleep", kwargs=dict(level=level))
    time_after_sleep = time.perf_counter()
    self.sleeping_tags = {"weights", "kv_cache"}
    self.is_sleeping = True
    logger.info(
        "It took %.6f seconds to fall asleep.", time_after_sleep - time_before_sleep
    )

def wake_up(self, tags: list[str] | None = None):
    if not self.is_sleeping:
        logger.warning("Executor is not sleeping.")
        return
    if tags:
        for tag in tags:
            if tag not in self.sleeping_tags:
                logger.warning(
                    "Tag %s is not in sleeping tags %s", tag, self.sleeping_tags
                )
                return
    time_before_wakeup = time.perf_counter()
    # 通过RPC调用所有worker的wake_up方法
    self.collective_rpc("wake_up", kwargs=dict(tags=tags))
    time_after_wakeup = time.perf_counter()
    logger.info(
        "It took %.6f seconds to wake up tags %s.",
        time_after_wakeup - time_before_wakeup,
        tags if tags is not None else self.sleeping_tags,
    )
    # 更新sleeping_tags状态
    if tags:
        for tag in tags:
            self.sleeping_tags.remove(tag)
    else:
        self.sleeping_tags.clear()
    if not self.sleeping_tags:
        self.is_sleeping = False
```

### 4. HTTP API 实现

**文件位置**: `vllm/entrypoints/serve/sleep/api_router.py`

```python
router = APIRouter()

@router.post("/sleep")
async def sleep(raw_request: Request):
    level = raw_request.query_params.get("level", "1")
    await engine_client(raw_request).sleep(int(level))
    return Response(status_code=200)

@router.post("/wake_up")
async def wake_up(raw_request: Request):
    tags = raw_request.query_params.getlist("tags")
    if tags == []:
        tags = None  # 无tags时唤醒所有
    await engine_client(raw_request).wake_up(tags)
    return Response(status_code=200)

@router.get("/is_sleeping")
async def is_sleeping(raw_request: Request):
    is_sleeping = await engine_client(raw_request).is_sleeping()
    return JSONResponse(content={"is_sleeping": is_sleeping})

def attach_router(app: FastAPI):
    # 仅在开发模式下启用
    if not envs.VLLM_SERVER_DEV_MODE:
        return
    app.include_router(router)
```

### 5. Python API (LLM类)

**文件位置**: `vllm/entrypoints/llm.py`

```python
# llm.py:1600-1634
def sleep(self, level: int = 1):
    """
    Put the engine to sleep. The engine should not process any requests.

    Args:
        level: The sleep level.
            Level 1: offload weights to CPU RAM, discard KV cache
            Level 2: discard both weights and KV cache
    """
    self.reset_prefix_cache()  # 清除prefix cache
    self.llm_engine.sleep(level=level)

def wake_up(self, tags: list[str] | None = None):
    """
    Wake up the engine from sleep mode.

    Args:
        tags: Optional list of tags to reallocate.
              Values: "weights", "kv_cache"
              If None, all memory is reallocated.
    """
    self.llm_engine.wake_up(tags)
```

## 监控与指标

**文件位置**: `vllm/v1/metrics/loggers.py`

Sleep Mode提供Prometheus指标来监控睡眠状态：

```python
# loggers.py:1131-1148
def record_sleep_state(self, sleep: int = 0, level: int = 0):
    awake = 1
    discard_all = 0
    weights_offloaded = 0

    if sleep == 1:
        awake = 0
        if level == 1:
            weights_offloaded = 1
        elif level == 2:
            discard_all = 1

    for engine_idx in self.engine_indexes:
        self.gauge_engine_sleep_state["discard_all"][engine_idx].set(discard_all)
        self.gauge_engine_sleep_state["weights_offloaded"][engine_idx].set(weights_offloaded)
        self.gauge_engine_sleep_state["awake"][engine_idx].set(awake)
```

**Prometheus指标名**: `vllm:engine_sleep_state`

| 标签值 | 含义 |
|--------|------|
| `awake` | 1=清醒，0=睡眠 |
| `weights_offloaded` | 1=Level 1睡眠 |
| `discard_all` | 1=Level 2睡眠 |

## 数据流图

### Sleep 流程

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              Sleep 流程 (Level 1)                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  用户调用 llm.sleep(level=1)                                                  │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────┐                                                         │
│  │ LLM.sleep()     │                                                         │
│  │ reset_prefix_   │                                                         │
│  │ cache()         │                                                         │
│  └────────┬────────┘                                                         │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                         │
│  │ LLMEngine.      │                                                         │
│  │ sleep(level=1)  │                                                         │
│  └────────┬────────┘                                                         │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                         │
│  │ EngineCore.     │                                                         │
│  │ sleep(level=1)  │                                                         │
│  └────────┬────────┘                                                         │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐     collective_rpc      ┌─────────────────────────────┐│
│  │ Executor.sleep()│────────────────────────►│ Worker.sleep(level=1)       ││
│  │ is_sleeping=True│                         │                             ││
│  │ sleeping_tags=  │                         │  if level==2:               ││
│  │ {weights,       │                         │    save buffers to CPU      ││
│  │  kv_cache}      │                         │                             ││
│  └─────────────────┘                         │  CuMemAllocator.sleep(      ││
│                                              │    offload_tags=("weights",)││
│                                              │  )                          ││
│                                              └──────────────┬──────────────┘│
│                                                             │               │
│                                                             ▼               │
│                                              ┌─────────────────────────────┐│
│                                              │ CuMemAllocator.sleep()      ││
│                                              │                             ││
│                                              │ for each allocation:        ││
│                                              │   if tag in offload_tags:   ││
│                                              │     cudaMemcpy(CPU, GPU)    ││
│                                              │     save to cpu_backup      ││
│                                              │   unmap_and_release(handle) ││
│                                              │                             ││
│                                              │ gc.collect()                ││
│                                              │ torch.cuda.empty_cache()    ││
│                                              └─────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────────────┘
```

### Wake Up 流程

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              Wake Up 流程                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  用户调用 llm.wake_up(tags=["weights"])                                       │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────┐                                                         │
│  │ LLM.wake_up()   │                                                         │
│  └────────┬────────┘                                                         │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐     collective_rpc      ┌─────────────────────────────┐│
│  │ Executor.       │────────────────────────►│ Worker.wake_up(             ││
│  │ wake_up(tags)   │                         │   tags=["weights"])         ││
│  │                 │                         │                             ││
│  │ 更新sleeping_   │                         │ CuMemAllocator.wake_up(tags)││
│  │ tags状态        │                         │                             ││
│  └─────────────────┘                         │ 恢复保存的buffers (if any)  ││
│                                              │                             ││
│                                              │ 重置FP8 scales (if needed)  ││
│                                              └──────────────┬──────────────┘│
│                                                             │               │
│                                                             ▼               │
│                                              ┌─────────────────────────────┐│
│                                              │ CuMemAllocator.wake_up()    ││
│                                              │                             ││
│                                              │ for each allocation:        ││
│                                              │   if tag matches:           ││
│                                              │     create_and_map(handle)  ││
│                                              │     if has cpu_backup:      ││
│                                              │       cudaMemcpy(GPU, CPU)  ││
│                                              │       clear cpu_backup      ││
│                                              └─────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────────────┘
```

## 使用示例

### 离线推理

```python
from vllm import LLM

# 启用sleep mode
llm = LLM("Qwen/Qwen3-0.6B", enable_sleep_mode=True)

# 进行推理
output = llm.generate("Hello, world!")

# Level 1睡眠：卸载权重到CPU，丢弃KV缓存
llm.sleep(level=1)

# ... 执行其他GPU任务 ...

# 唤醒引擎
llm.wake_up()

# 继续推理
output = llm.generate("Hello again!")
```

### RLHF 权重更新场景

```python
from vllm import LLM

llm = LLM("my-model", enable_sleep_mode=True)

# Level 2睡眠：丢弃所有数据
llm.sleep(level=2)

# 仅唤醒权重内存（避免OOM）
llm.wake_up(tags=["weights"])

# 原地更新权重
llm.collective_rpc("reload_weights")

# 唤醒KV缓存
llm.wake_up(tags=["kv_cache"])

# 现在可以继续推理
```

### 在线服务

```bash
# 启动服务器（需要开发模式）
VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6B \
  --enable-sleep-mode \
  --port 8000

# 睡眠
curl -X POST 'http://localhost:8000/sleep?level=1'

# 检查状态
curl -X GET 'http://localhost:8000/is_sleeping'
# 返回: {"is_sleeping": true}

# 唤醒
curl -X POST 'http://localhost:8000/wake_up'
```

## 配置参数

| 参数 | 位置 | 说明 |
|------|------|------|
| `enable_sleep_mode` | `ModelConfig` / CLI `--enable-sleep-mode` | 启用sleep mode |
| `VLLM_SERVER_DEV_MODE` | 环境变量 | 启用开发模式HTTP端点 |
| `VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE` | 环境变量 | ROCm平台内存分块大小(MB)，默认256 |

## 平台支持

```python
# vllm/platforms/interface.py:176-181
def is_sleep_mode_available(self) -> bool:
    # 仅CUDA和ROCm平台支持sleep mode
    return current_platform.is_cuda() or current_platform.is_rocm()
```

## 限制与注意事项

1. **CPU内存要求**：Level 1需要足够的CPU内存来存储模型权重备份
2. **ROCm限制**：使用分块内存分配，可通过`VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE`调整
3. **进程限制**：每个进程只能有一个sleep mode实例
4. **不兼容expandable_segments**：需要禁用`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
5. **开发模式**：HTTP端点仅在`VLLM_SERVER_DEV_MODE=1`时可用

## 性能数据

根据官方博客，Sleep Mode相比冷启动的性能优势：

| 场景 | 冷启动时间 | Sleep Mode恢复时间 | 加速比 |
|------|-----------|-------------------|--------|
| 小模型 | ~30s | 1-2s | 15-30x |
| 大模型 | 100s+ | 5-10s | 10-20x |

加速的原因不仅是更快的权重加载，更重要的是**保留了冷启动时需要重建的昂贵基础设施**（如CUDA图、编译优化等）。

## 相关文件索引

| 文件 | 功能 |
|------|------|
| `vllm/device_allocator/cumem.py` | 核心内存分配器 |
| `vllm/v1/worker/gpu_worker.py` | GPU Worker sleep/wake_up实现 |
| `vllm/v1/executor/abstract.py` | Executor层协调 |
| `vllm/v1/engine/core.py` | EngineCore层接口 |
| `vllm/v1/engine/async_llm.py` | 异步LLM引擎 |
| `vllm/entrypoints/llm.py` | 用户Python API |
| `vllm/entrypoints/serve/sleep/api_router.py` | HTTP API路由 |
| `vllm/v1/metrics/loggers.py` | Prometheus指标 |
| `vllm/config/model.py` | enable_sleep_mode配置 |
| `tests/entrypoints/sleep/test_sleep.py` | 测试用例 |

## 参考资料

- [vLLM Sleep Mode 官方博客](https://blog.vllm.ai/2025/10/26/sleep-mode.html)
- [vLLM Sleep Mode 官方文档](https://docs.vllm.ai/en/latest/features/sleep_mode/)
- [Hugging Face - No GPU left behind: Unlocking Efficiency with Co-located vLLM in TRL](https://huggingface.co/blog/vllm-colocate)
