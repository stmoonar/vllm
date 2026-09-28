# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A cold compile must not leave autotuning scratch memory reserved."""

import pytest
import torch
import torch.fx as fx

import vllm.envs as envs
from vllm.compilation import compiler_interface
from vllm.compilation.compiler_interface import InductorStandaloneAdaptor
from vllm.config.utils import Range


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_standalone_compile_releases_autotune_scratch(monkeypatch, tmp_path):
    scratch_bytes = 256 << 20
    sentinel = object()

    def fake_standalone_compile(graph, example_inputs, **kwargs):
        # Stand-in for compile-time autotuning: a large random input that is
        # freed before compile returns.
        scratch = torch.empty(scratch_bytes, dtype=torch.uint8, device="cuda")
        del scratch
        return sentinel

    monkeypatch.setattr(torch._inductor, "standalone_compile", fake_standalone_compile)
    monkeypatch.setattr(envs, "VLLM_USE_MEGA_AOT_ARTIFACT", False)
    monkeypatch.setattr(
        compiler_interface, "is_compile_cache_enabled", lambda config: False
    )

    torch.cuda.init()
    torch.accelerator.empty_cache()
    reserved_before = torch.cuda.memory_reserved()

    adaptor = InductorStandaloneAdaptor(save_format="binary")
    adaptor.initialize_cache(str(tmp_path))
    compiled, _ = adaptor.compile(
        fx.GraphModule(torch.nn.Module(), fx.Graph()),
        [],
        {},
        Range(start=1, end=8),
        key="k",
    )

    assert compiled is sentinel
    assert torch.cuda.memory_reserved() - reserved_before < scratch_bytes


def test_release_is_noop_without_cuda_context(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: calls.append(1))
    compiler_interface.release_autotune_scratch_memory()
    assert calls == []
