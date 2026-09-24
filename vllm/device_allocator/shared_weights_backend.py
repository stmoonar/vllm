# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sleep-mode backend that restores weights from a node-wide host snapshot.

Assumes weights never change after loading. The first suspend copies the
``weights`` pool once into a page-locked, NUMA-local tmpfs file; instances on
the same node that hold the same shard attach to that file instead of making
their own copy. Afterwards suspend only unmaps device memory and resume only
copies host to device.

A snapshot is shared only when every tensor reachable from the model (and the
draft model) sits at the same place in the pool and has the same checksum;
otherwise the instance falls back to a private snapshot.
"""

from __future__ import annotations

import atexit
import bisect
import contextlib
import ctypes
import fcntl
import gc
import hashlib
import json
import mmap
import os
import platform
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import regex as re
import torch

import vllm.envs as envs
from vllm.device_allocator.sleep_mode_backend import SleepModeBackend
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from torch import nn

    from vllm.config import VllmConfig
    from vllm.device_allocator import AllocationData, HandleType

logger = init_logger(__name__)

WEIGHTS_TAG = "weights"
_CHECKSUM_CHUNK = 1 << 22
_MPOL_PREFERRED = 1
_SYS_MBIND = {"x86_64": 237, "aarch64": 235}

# (name, allocation index, offset in allocation, nbytes, checksum)
TensorEntry = list[Any]


def _allocator():
    from vllm.device_allocator.cumem import CuMemAllocator

    return CuMemAllocator.get_instance()


def _unmap(handle: HandleType) -> None:
    from vllm.device_allocator.cumem import unmap_and_release

    unmap_and_release(handle)


def _memcpy(dst: int, src: int, nbytes: int) -> None:
    from vllm.device_allocator.cumem import libcudart

    libcudart.cudaMemcpy(dst, src, nbytes)


def _host_register(ptr: int, nbytes: int) -> None:
    from vllm.device_allocator.cumem import libcudart

    if libcudart.cudaHostRegister(ptr, nbytes, 0) != 0:
        libcudart.cudaGetLastError()
        logger.warning(
            "cudaHostRegister failed for the weight snapshot; wake-up will copy "
            "from pageable memory at reduced bandwidth."
        )


def gpu_numa_node(device: torch.device) -> int | None:
    props = torch.cuda.get_device_properties(device)
    pci = (
        f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
    )
    try:
        with open(f"/sys/bus/pci/devices/{pci}/numa_node") as f:
            node = int(f.read())
    except (OSError, ValueError):
        return None
    return node if node >= 0 else None


def bind_to_numa_node(addr: int, nbytes: int, node: int) -> bool:
    """Prefer ``node`` for pages first touched in ``[addr, addr + nbytes)``."""
    nr = _SYS_MBIND.get(platform.machine())
    if nr is None:
        return False
    nwords = node // 64 + 1
    mask = (ctypes.c_ulong * nwords)()
    mask[node // 64] = 1 << (node % 64)
    libc = ctypes.CDLL(None, use_errno=True)
    ret = libc.syscall(
        ctypes.c_long(nr),
        ctypes.c_void_p(addr),
        ctypes.c_ulong(nbytes),
        ctypes.c_int(_MPOL_PREFERRED),
        mask,
        ctypes.c_ulong(nwords * 64 + 1),
        ctypes.c_uint(0),
    )
    if ret != 0:
        logger.warning(
            "mbind to NUMA node %d failed: %s",
            node,
            os.strerror(ctypes.get_errno()),
        )
    return ret == 0


def checksum_weights(device: torch.device) -> torch.Tensor:
    weights = torch.arange(_CHECKSUM_CHUNK, dtype=torch.int64, device=device)
    return weights % 65521 + 1


def checksum(data: torch.Tensor, weights: torch.Tensor) -> str:
    """Checksum a 1-D uint8 tensor on its device. Integer-only, so the result
    does not depend on reduction order."""
    n4 = data.numel() // 4 * 4
    total = torch.zeros(2, dtype=torch.int64, device=data.device)
    chunks = list(data[:n4].view(torch.int32).split(_CHECKSUM_CHUNK))
    chunks.append(data[n4:])
    for chunk in chunks:
        x = chunk.to(torch.int64)
        total[0] += x.sum()
        total[1] += (x * weights[: x.numel()]).sum()
    return "{:x}:{:x}".format(*total.tolist())


def _module_tensors(module: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    for name, value in (
        *module._parameters.items(),
        *module._buffers.items(),
        *vars(module).items(),
    ):
        if isinstance(value, torch.Tensor):
            yield name, value


@contextlib.contextmanager
def _flock(path: str) -> Iterator[None]:
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


class HostSnapshot:
    """A page-locked tmpfs mapping holding one rank's weights pool."""

    def __init__(self, fd: int, nbytes: int) -> None:
        self.fd = fd
        self.mm = mmap.mmap(fd, nbytes)
        self.buffer = torch.frombuffer(self.mm, dtype=torch.uint8)
        self.path: str | None = None

    @classmethod
    def create(cls, path: str, nbytes: int, numa_node: int | None) -> HostSnapshot:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.ftruncate(fd, nbytes)
            snapshot = cls(fd, nbytes)
            # Set the policy before posix_fallocate first-touches the pages.
            if numa_node is not None:
                bind_to_numa_node(snapshot.buffer.data_ptr(), nbytes, numa_node)
            # Reserve tmpfs pages now: running out later would SIGBUS.
            os.posix_fallocate(fd, 0, nbytes)
        except OSError as e:
            os.close(fd)
            os.unlink(path)
            raise RuntimeError(
                f"Cannot allocate a {nbytes / 1024**3:.2f} GiB weight snapshot at "
                f"{path}. Enlarge the tmpfs (e.g. docker --shm-size) or point "
                "VLLM_SHARED_WEIGHTS_DIR elsewhere."
            ) from e
        _host_register(snapshot.buffer.data_ptr(), nbytes)
        return snapshot

    @classmethod
    def attach(cls, path: str, nbytes: int) -> HostSnapshot:
        fd = os.open(path, os.O_RDWR)
        if os.fstat(fd).st_size != nbytes:
            os.close(fd)
            raise RuntimeError(f"Weight snapshot {path} has an unexpected size.")
        snapshot = cls(fd, nbytes)
        _host_register(snapshot.buffer.data_ptr(), nbytes)
        return snapshot

    def share(self, path: str) -> None:
        """Hold a shared lock on the published file so the last user can
        unlink it on exit. Callers must hold the snapshot's ``.lock``."""
        fcntl.flock(self.fd, fcntl.LOCK_SH)
        self.path = path
        atexit.register(self.release)

    def release(self) -> None:
        """Drop this process's reference; unlink the files if it was the last."""
        if self.path is None:
            return
        base = self.path.removesuffix(".bin")
        with contextlib.suppress(OSError), _flock(base + ".lock"):
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            try:
                probe = os.open(self.path, os.O_RDONLY)
            except FileNotFoundError:
                return
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            finally:
                os.close(probe)
            for path in (base + ".json", self.path):
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(path)
        self.path = None


class SharedWeightsBackend(SleepModeBackend):
    """Level-1 sleep backed by a node-wide, read-only host weight snapshot."""

    def __init__(self) -> None:
        super().__init__()
        self._vllm_config: VllmConfig | None = None
        self._models: dict[str, nn.Module] = {}
        self._snapshot: HostSnapshot | None = None
        self._layout: list[tuple[int, int]] = []
        self._views: list[torch.Tensor] = []

    @classmethod
    def is_supported(cls) -> bool:
        return current_platform.is_cuda()

    @classmethod
    def preserves_communicators(cls) -> bool:
        return True

    @classmethod
    def verify_config(cls, vllm_config: VllmConfig) -> None:
        mutating = [
            feature
            for feature, enabled in (
                ("LoRA", vllm_config.lora_config is not None),
                ("EPLB", vllm_config.parallel_config.enable_eplb),
                ("weight transfer", vllm_config.weight_transfer_config is not None),
            )
            if enabled
        ]
        if mutating:
            raise ValueError(
                "Sleep-mode backend 'shared_weights' requires weights to stay "
                f"unchanged after loading, which {', '.join(mutating)} violates."
            )

    def bind(self, vllm_config: VllmConfig, models: dict[str, nn.Module]) -> None:
        self._vllm_config = vllm_config
        self._models = models

    def suspend(self, level: int = 1) -> None:
        if level != 1:
            raise ValueError(
                "Sleep-mode backend 'shared_weights' only supports level 1; "
                "level 2 implies the weights will be replaced."
            )
        self._release(None)

    def discard(self, tags: tuple[str, ...]) -> None:
        self._release(tags)

    def resume(self, tags: list[str] | None = None) -> None:
        self._state = "RESUMING"
        _allocator().wake_up(tags)
        self._state = "RUNNING"

    def _release(self, tags: tuple[str, ...] | None) -> None:
        allocator = _allocator()
        if tags is None or WEIGHTS_TAG in tags:
            weights = [
                data
                for data in allocator.pointer_to_data.values()
                if data.tag == WEIGHTS_TAG
            ]
            if self._snapshot is None:
                self._take_snapshot(weights)
            elif self._layout != [(d.handle[2], d.handle[1]) for d in weights]:
                raise RuntimeError("The weights pool changed after the snapshot.")
            for data, view in zip(weights, self._views):
                data.cpu_backup_tensor = view

        torch.accelerator.synchronize()
        freed = 0
        for data in allocator.pointer_to_data.values():
            if data.is_asleep or (tags is not None and data.tag not in tags):
                continue
            _unmap(data.handle)
            data.is_asleep = True
            freed += data.handle[1]
        self._state = "SUSPENDED"
        logger.info("shared_weights: released %.2f GiB.", freed / 1024**3)
        gc.collect()
        torch.accelerator.empty_cache()

    def _take_snapshot(self, weights: list[AllocationData]) -> None:
        if not weights:
            return
        if any(data.is_asleep for data in weights):
            raise RuntimeError("Cannot snapshot weights that are already released.")
        sizes = [data.handle[1] for data in weights]
        table = self._tensor_table(weights)
        total = sum(sizes)
        numa_node = self._numa_node()
        base = os.path.join(envs.VLLM_SHARED_WEIGHTS_DIR, self._key(sizes, numa_node))
        os.makedirs(envs.VLLM_SHARED_WEIGHTS_DIR, exist_ok=True)

        snapshot = None
        with _flock(base + ".lock"):
            manifest = self._read_manifest(base + ".json")
            if manifest is None:
                snapshot = self._fill(
                    HostSnapshot.create(base + ".tmp", total, numa_node), weights
                )
                os.rename(base + ".tmp", base + ".bin")
                snapshot.share(base + ".bin")
                self._write_manifest(base + ".json", table)
                logger.info(
                    "shared_weights: published a %.2f GiB snapshot at %s "
                    "(NUMA node %s).",
                    total / 1024**3,
                    base,
                    numa_node,
                )
            elif manifest == table:
                snapshot = HostSnapshot.attach(base + ".bin", total)
                snapshot.share(base + ".bin")
                logger.info(
                    "shared_weights: attached to the snapshot at %s after "
                    "verifying %.2f GiB of named tensors in the %.2f GiB pool.",
                    base,
                    sum(entry[3] for entry in table) / 1024**3,
                    total / 1024**3,
                )
            else:
                mismatched = [
                    ours[0] for ours, theirs in zip(table, manifest) if ours != theirs
                ]
                logger.warning(
                    "shared_weights: this instance's weights differ from the "
                    "snapshot at %s (first mismatches: %s); using a private "
                    "snapshot.",
                    base,
                    mismatched[:5] or "tensor count",
                )

        if snapshot is None:
            path = f"{base}.{os.getpid()}.private"
            snapshot = HostSnapshot.create(path, total, numa_node)
            os.unlink(path)
            self._fill(snapshot, weights)

        self._snapshot = snapshot
        self._layout = [(data.handle[2], data.handle[1]) for data in weights]
        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size)
        self._views = [
            snapshot.buffer[start:end] for start, end in zip(offsets, offsets[1:])
        ]

    @staticmethod
    def _fill(snapshot: HostSnapshot, weights: list[AllocationData]) -> HostSnapshot:
        host = snapshot.buffer.data_ptr()
        for data in weights:
            _memcpy(host, data.handle[2], data.handle[1])
            host += data.handle[1]
        return snapshot

    def _tensor_table(self, weights: list[AllocationData]) -> list[TensorEntry]:
        """Locate and checksum every storage in the pool reachable by name."""
        starts = [data.handle[2] for data in weights]
        order = sorted(range(len(weights)), key=starts.__getitem__)
        sorted_starts = [starts[i] for i in order]
        seen: set[int] = set()
        table: list[TensorEntry] = []
        weights_by_device: dict[torch.device, torch.Tensor] = {}
        for prefix, model in self._models.items():
            for module_name, module in model.named_modules():
                for attr, tensor in _module_tensors(module):
                    try:
                        storage = tensor.untyped_storage()
                        ptr, nbytes = storage.data_ptr(), storage.nbytes()
                    except (RuntimeError, NotImplementedError):
                        # Wrapper subclasses without their own storage.
                        continue
                    if nbytes == 0 or ptr in seen:
                        continue
                    pos = bisect.bisect_right(sorted_starts, ptr) - 1
                    if pos < 0:
                        continue
                    index = order[pos]
                    offset = ptr - starts[index]
                    if offset + nbytes > weights[index].handle[1]:
                        continue
                    seen.add(ptr)
                    data = torch.empty(0, dtype=torch.uint8, device=tensor.device)
                    data.set_(storage)
                    if tensor.device not in weights_by_device:
                        weights_by_device[tensor.device] = checksum_weights(
                            tensor.device
                        )
                    name = ".".join(filter(None, (prefix, module_name, attr)))
                    digest = checksum(data, weights_by_device[tensor.device])
                    table.append([name, index, offset, nbytes, digest])
        return table

    @staticmethod
    def _numa_node() -> int | None:
        return gpu_numa_node(torch.device("cuda", torch.cuda.current_device()))

    def _key(self, sizes: list[int], numa_node: int | None) -> str:
        from vllm import __version__
        from vllm.distributed.parallel_state import (
            get_pp_group,
            get_tensor_model_parallel_rank,
        )

        config = self._vllm_config
        assert config is not None, "bind() must be called before suspend()"
        parallel = config.parallel_config
        device = torch.cuda.current_device()
        factors = [
            __version__,
            torch.__version__,
            config.model_config.compute_hash(),
            config.load_config.compute_hash(),
            config.cache_config.cache_dtype,
            (
                config.speculative_config.compute_hash()
                if config.speculative_config is not None
                else None
            ),
            parallel.tensor_parallel_size,
            parallel.pipeline_parallel_size,
            get_tensor_model_parallel_rank(),
            get_pp_group().rank_in_group,
            parallel.enable_expert_parallel,
            (
                (parallel.data_parallel_size, parallel.data_parallel_rank)
                if parallel.enable_expert_parallel
                else None
            ),
            torch.cuda.get_device_name(device),
            torch.cuda.get_device_capability(device),
            numa_node,
            sizes,
        ]
        digest = hashlib.sha256(json.dumps(factors).encode()).hexdigest()[:32]
        name = os.path.basename(config.model_config.model.rstrip("/"))
        return re.sub(r"[^\w.-]", "_", name) + "-" + digest

    @staticmethod
    def _read_manifest(path: str) -> list[TensorEntry] | None:
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    @staticmethod
    def _write_manifest(path: str, table: list[TensorEntry]) -> None:
        with open(path + ".tmp", "w") as f:
            json.dump(table, f)
        os.rename(path + ".tmp", path)
