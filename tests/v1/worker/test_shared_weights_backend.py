# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the ``shared_weights`` sleep-mode backend.

Device memory is simulated with host memory: "unmapping" zeroes it and waking
copies the backup back, mirroring ``CuMemAllocator.wake_up``.
"""

import ctypes
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.device_allocator.shared_weights_backend as sw
from vllm.device_allocator import AllocationData
from vllm.device_allocator.shared_weights_backend import SharedWeightsBackend
from vllm.device_allocator.sleep_mode_backend import SleepModeBackendFactory

POOL_BYTES = 1 << 16


class FakeAllocator:
    def __init__(self) -> None:
        self.memory: list[torch.Tensor] = []
        self.pointer_to_data: dict[int, AllocationData] = {}

    def allocate(self, tag: str, nbytes: int) -> int:
        buf = torch.zeros(nbytes, dtype=torch.uint8)
        self.memory.append(buf)
        ptr = buf.data_ptr()
        self.pointer_to_data[ptr] = AllocationData((0, nbytes, ptr, 0), tag)
        return ptr

    def wake_up(self, tags: list[str] | None = None) -> None:
        for ptr, data in self.pointer_to_data.items():
            if data.is_asleep and (tags is None or data.tag in tags):
                data.is_asleep = False
                if data.cpu_backup_tensor is not None:
                    ctypes.memmove(
                        ptr, data.cpu_backup_tensor.data_ptr(), data.handle[1]
                    )
                    data.cpu_backup_tensor = None


def tensor_at(ptr: int, numel: int) -> torch.Tensor:
    """A float32 tensor with its own storage over existing memory."""
    raw = (ctypes.c_uint8 * (numel * 4)).from_address(ptr)
    return torch.frombuffer(raw, dtype=torch.float32)


class Instance:
    """One simulated vLLM instance: a weights pool, a KV pool and a model."""

    def __init__(self, seed: int = 0) -> None:
        self.allocator = FakeAllocator()
        pool = self.allocator.allocate("weights", POOL_BYTES)
        self.kv_ptr = self.allocator.allocate("kv_cache", POOL_BYTES)
        self.model = nn.Linear(64, 32)
        generator = torch.Generator().manual_seed(seed)
        self.model.weight = nn.Parameter(tensor_at(pool, 64 * 32).view(32, 64))
        self.model.bias = nn.Parameter(tensor_at(pool + 64 * 32 * 4, 32))
        self.model.scale = tensor_at(pool + 16384, 16)
        with torch.no_grad():
            for t in (self.model.weight, self.model.bias, self.model.scale):
                t.copy_(torch.randn(t.shape, generator=generator))
        self.expected = [
            t.clone() for t in (self.model.weight, self.model.bias, self.model.scale)
        ]
        self.backend = SharedWeightsBackend()
        self.backend.bind(SimpleNamespace(), {"model": self.model})

    def weights_intact(self) -> bool:
        tensors = (self.model.weight, self.model.bias, self.model.scale)
        return all(torch.equal(t, e) for t, e in zip(tensors, self.expected))

    def sleep(self, monkeypatch) -> None:
        monkeypatch.setattr(sw, "_allocator", lambda: self.allocator)
        self.backend.suspend(level=1)

    def wake(self, monkeypatch) -> None:
        monkeypatch.setattr(sw, "_allocator", lambda: self.allocator)
        self.backend.resume()


@pytest.fixture
def d2h_copies(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_SHARED_WEIGHTS_DIR", str(tmp_path))
    monkeypatch.setattr(sw, "_host_register", lambda ptr, nbytes: None)
    monkeypatch.setattr(
        sw, "_unmap", lambda handle: ctypes.memset(handle[2], 0, handle[1])
    )
    copies: list[int] = []

    def memcpy(dst: int, src: int, nbytes: int) -> None:
        copies.append(nbytes)
        ctypes.memmove(dst, src, nbytes)

    monkeypatch.setattr(sw, "_memcpy", memcpy)
    monkeypatch.setattr(SharedWeightsBackend, "_numa_node", staticmethod(lambda: None))
    monkeypatch.setattr(
        SharedWeightsBackend, "_key", lambda self, sizes, numa: "model-rank0"
    )
    monkeypatch.setattr("torch.accelerator.synchronize", lambda: None)
    monkeypatch.setattr("torch.accelerator.empty_cache", lambda: None)
    return copies


def snapshot_files(tmp_path) -> list[str]:
    return sorted(p.name for p in tmp_path.iterdir() if p.suffix != ".lock")


def test_registered_with_factory():
    assert (
        SleepModeBackendFactory.get_backend_class("shared_weights")
        is SharedWeightsBackend
    )


@pytest.mark.parametrize(
    "overrides, feature",
    [
        ({"lora_config": object()}, "LoRA"),
        ({"enable_eplb": True}, "EPLB"),
        ({"weight_transfer_config": object()}, "weight transfer"),
    ],
)
def test_rejects_features_that_mutate_weights(overrides, feature):
    def make_config(**kw):
        return SimpleNamespace(
            lora_config=kw.get("lora_config"),
            weight_transfer_config=kw.get("weight_transfer_config"),
            parallel_config=SimpleNamespace(enable_eplb=kw.get("enable_eplb", False)),
        )

    SharedWeightsBackend.verify_config(make_config())
    with pytest.raises(ValueError, match=feature):
        SharedWeightsBackend.verify_config(make_config(**overrides))


def test_rejects_level_2():
    with pytest.raises(ValueError, match="level 1"):
        SharedWeightsBackend().suspend(level=2)


def test_sleep_frees_everything_and_wake_restores_weights(monkeypatch, d2h_copies):
    a = Instance()
    a.sleep(monkeypatch)
    assert all(d.is_asleep for d in a.allocator.pointer_to_data.values())
    assert not a.weights_intact()
    assert d2h_copies == [POOL_BYTES]

    a.wake(monkeypatch)
    assert a.weights_intact()

    # The snapshot is taken once; later sleeps only unmap.
    a.sleep(monkeypatch)
    a.wake(monkeypatch)
    assert a.weights_intact()
    assert d2h_copies == [POOL_BYTES]


def test_identical_instance_attaches_without_copying(monkeypatch, d2h_copies, tmp_path):
    a, b = Instance(), Instance()
    a.sleep(monkeypatch)
    b.sleep(monkeypatch)
    assert d2h_copies == [POOL_BYTES]
    assert snapshot_files(tmp_path) == ["model-rank0.bin", "model-rank0.json"]

    b.wake(monkeypatch)
    assert b.weights_intact()


def test_different_weights_fall_back_to_private_snapshot(
    monkeypatch, d2h_copies, tmp_path
):
    a, c = Instance(seed=0), Instance(seed=1)
    a.sleep(monkeypatch)
    c.sleep(monkeypatch)
    assert d2h_copies == [POOL_BYTES, POOL_BYTES]
    assert snapshot_files(tmp_path) == ["model-rank0.bin", "model-rank0.json"]

    a.wake(monkeypatch)
    c.wake(monkeypatch)
    assert a.weights_intact()
    assert c.weights_intact()


def test_last_user_removes_snapshot(monkeypatch, d2h_copies, tmp_path):
    a, b = Instance(), Instance()
    a.sleep(monkeypatch)
    b.sleep(monkeypatch)

    a.backend._snapshot.release()
    assert snapshot_files(tmp_path) == ["model-rank0.bin", "model-rank0.json"]
    b.backend._snapshot.release()
    assert snapshot_files(tmp_path) == []


def test_numa_binding_applies_to_shared_mapping(tmp_path):
    path = str(tmp_path / "snapshot")
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    os.ftruncate(fd, POOL_BYTES)
    snapshot = sw.HostSnapshot(fd, POOL_BYTES)
    if not sw.bind_to_numa_node(snapshot.buffer.data_ptr(), POOL_BYTES, 0):
        pytest.skip("mbind is not permitted here")
    with open("/proc/self/numa_maps") as f:
        assert any("prefer:0" in line and path in line for line in f)
