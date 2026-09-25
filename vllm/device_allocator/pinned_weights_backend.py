# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sleep-mode backend that keeps a pinned host copy of immutable weights.

The first suspend copies the ``weights`` pool to pinned host memory, as
``CuMemAllocator.sleep`` does, but the copy is kept for the life of the
process. Later suspends only unmap device memory and resume only copies host
to device.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import torch

from vllm.device_allocator import cumem
from vllm.device_allocator.sleep_mode_backend import SleepModeBackend
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import PIN_MEMORY

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.device_allocator import AllocationData

logger = init_logger(__name__)

WEIGHTS_TAG = "weights"


class PinnedWeightsBackend(SleepModeBackend):
    """Level-1 sleep that copies the weights to host memory only once."""

    def __init__(self) -> None:
        super().__init__()
        self._backups: dict[int, torch.Tensor] = {}

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
                "Sleep-mode backend 'pinned_weights' requires weights to stay "
                f"unchanged after loading, which {', '.join(mutating)} violates."
            )

    def suspend(self, level: int = 1) -> None:
        if level != 1:
            raise ValueError(
                "Sleep-mode backend 'pinned_weights' only supports level 1; "
                "level 2 implies the weights will be replaced."
            )
        allocator = cumem.CuMemAllocator.get_instance()
        torch.accelerator.synchronize()
        self._attach_backups(
            {
                ptr: data
                for ptr, data in allocator.pointer_to_data.items()
                if data.tag == WEIGHTS_TAG
            }
        )

        freed = 0
        for data in allocator.pointer_to_data.values():
            if data.is_asleep:
                continue
            cumem.unmap_and_release(data.handle)
            data.is_asleep = True
            freed += data.handle[1]
        self._state = "SUSPENDED"
        logger.info("pinned_weights: released %.2f GiB.", freed / 1024**3)
        gc.collect()
        torch.accelerator.empty_cache()

    def resume(self, tags: list[str] | None = None) -> None:
        self._state = "RESUMING"
        cumem.CuMemAllocator.get_instance().wake_up(tags)
        self._state = "RUNNING"

    def _attach_backups(self, weights: dict[int, AllocationData]) -> None:
        if not self._backups:
            if any(data.is_asleep for data in weights.values()):
                raise RuntimeError("Cannot back up weights that are already released.")
            for ptr, data in weights.items():
                backup = torch.empty(
                    data.handle[1],
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=PIN_MEMORY,
                )
                cumem.libcudart.cudaMemcpy(backup.data_ptr(), ptr, data.handle[1])
                self._backups[ptr] = backup
        elif weights.keys() != self._backups.keys() or any(
            data.handle[1] != self._backups[ptr].numel()
            for ptr, data in weights.items()
        ):
            raise RuntimeError("The weights pool changed after the host copy.")
        for ptr, data in weights.items():
            data.cpu_backup_tensor = self._backups[ptr]
