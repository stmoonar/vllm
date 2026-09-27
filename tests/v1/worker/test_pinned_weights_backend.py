# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the ``pinned_weights`` sleep-mode backend.

Device memory is simulated with host memory: "unmapping" zeroes it and waking
copies the backup back, mirroring ``CuMemAllocator.wake_up``.
"""

import ctypes
from types import SimpleNamespace

import pytest
import torch

import vllm.device_allocator.pinned_weights_backend as pw
from vllm.device_allocator import AllocationData, cumem
from vllm.device_allocator.pinned_weights_backend import PinnedWeightsBackend
from vllm.device_allocator.sleep_mode_backend import SleepModeBackendFactory

POOL_BYTES = 1 << 16


class FakeAllocator:
    def __init__(self) -> None:
        self.memory: dict[str, torch.Tensor] = {}
        self.pointer_to_data: dict[int, AllocationData] = {}
        self.d2h_copies: list[int] = []
        for tag in ("weights", "kv_cache"):
            buf = torch.randint(0, 256, (POOL_BYTES,), dtype=torch.uint8)
            self.memory[tag] = buf
            ptr = buf.data_ptr()
            self.pointer_to_data[ptr] = AllocationData((0, POOL_BYTES, ptr, 0), tag)

    def wake_up(self, tags: list[str] | None = None) -> None:
        for ptr, data in self.pointer_to_data.items():
            if data.is_asleep and (tags is None or data.tag in tags):
                data.is_asleep = False
                if data.cpu_backup_tensor is not None:
                    ctypes.memmove(
                        ptr, data.cpu_backup_tensor.data_ptr(), data.handle[1]
                    )
                    data.cpu_backup_tensor = None


@pytest.fixture
def allocator(monkeypatch):
    fake = FakeAllocator()
    monkeypatch.setattr(cumem.CuMemAllocator, "get_instance", lambda: fake)
    monkeypatch.setattr(
        cumem,
        "unmap_and_release",
        lambda handle: ctypes.memset(handle[2], 0, handle[1]),
    )

    def memcpy(dst: int, src: int, nbytes: int) -> None:
        fake.d2h_copies.append(nbytes)
        ctypes.memmove(dst, src, nbytes)

    monkeypatch.setattr(cumem, "libcudart", SimpleNamespace(cudaMemcpy=memcpy))
    monkeypatch.setattr(pw, "PIN_MEMORY", False)
    monkeypatch.setattr("torch.accelerator.synchronize", lambda: None)
    monkeypatch.setattr("torch.accelerator.empty_cache", lambda: None)
    return fake


def test_registered_with_factory():
    assert (
        SleepModeBackendFactory.get_backend_class("pinned_weights")
        is PinnedWeightsBackend
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

    PinnedWeightsBackend.verify_config(make_config())
    with pytest.raises(ValueError, match=feature):
        PinnedWeightsBackend.verify_config(make_config(**overrides))


def test_rejects_level_2():
    with pytest.raises(ValueError, match="level 1"):
        PinnedWeightsBackend().suspend(level=2)


def test_host_copy_taken_once_and_kept_after_wake(allocator):
    weights = allocator.memory["weights"]
    expected = weights.clone()
    backend = PinnedWeightsBackend()

    backend.suspend(level=1)
    assert all(d.is_asleep for d in allocator.pointer_to_data.values())
    assert not torch.equal(weights, expected)
    assert not allocator.memory["kv_cache"].any()
    assert allocator.d2h_copies == [POOL_BYTES]
    (backup,) = backend._backups.values()

    for _ in range(2):
        backend.resume()
        assert torch.equal(weights, expected)
        assert backend._backups[weights.data_ptr()] is backup
        backend.suspend(level=1)
    assert allocator.d2h_copies == [POOL_BYTES]


def test_only_level_1_is_supported():
    from vllm.exceptions import VLLMValidationError

    assert PinnedWeightsBackend.supported_sleep_levels() == (1,)
    model_config = SimpleNamespace(sleep_mode_backend="pinned_weights")
    SleepModeBackendFactory.verify_sleep_level(model_config, 1)
    with pytest.raises(VLLMValidationError, match="level 2"):
        SleepModeBackendFactory.verify_sleep_level(model_config, 2)


def test_failed_host_copy_is_not_kept(allocator, monkeypatch):
    """A first suspend that fails part way through the host copy must leave
    no partial copy behind, or every later suspend fails the pool check."""
    extra = torch.randint(0, 256, (POOL_BYTES,), dtype=torch.uint8)
    allocator.memory["weights_2"] = extra
    allocator.pointer_to_data[extra.data_ptr()] = AllocationData(
        (0, POOL_BYTES, extra.data_ptr(), 0), "weights"
    )
    copy = cumem.libcudart.cudaMemcpy
    calls = 0

    def fail_second_copy(dst: int, src: int, nbytes: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("out of pinned memory")
        copy(dst, src, nbytes)

    monkeypatch.setattr(
        cumem, "libcudart", SimpleNamespace(cudaMemcpy=fail_second_copy)
    )
    backend = PinnedWeightsBackend()
    with pytest.raises(RuntimeError, match="pinned memory"):
        backend.suspend(level=1)
    assert backend._backups == {}
    assert not any(d.is_asleep for d in allocator.pointer_to_data.values())

    monkeypatch.setattr(cumem, "libcudart", SimpleNamespace(cudaMemcpy=copy))
    backend.suspend(level=1)
    assert len(backend._backups) == 2
    assert all(d.is_asleep for d in allocator.pointer_to_data.values())
