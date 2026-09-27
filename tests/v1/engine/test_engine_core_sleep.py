# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for how EngineCore and Executor handle a failed sleep.

A sleep pauses the scheduler before the executor releases GPU memory. If the
executor then fails, the engine must not stay paused while awake: it would
report sleeping and hold every new request until a manual wake_up.
"""

from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from vllm.device_allocator.sleep_mode_backend import (
    CuMemBackend,
    SleepModeBackendFactory,
)
from vllm.exceptions import VLLMValidationError
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.engine.core import DPEngineCoreProc, EngineCore, EngineCoreProc
from vllm.v1.executor.uniproc_executor import UniProcExecutor


class _Level1OnlyBackend(CuMemBackend):
    @classmethod
    def supported_sleep_levels(cls) -> tuple[int, ...]:
        return (1,)


@pytest.fixture
def level1_only_backend(monkeypatch):
    monkeypatch.setitem(
        SleepModeBackendFactory._registry, "level1_only", lambda: _Level1OnlyBackend
    )
    return "level1_only"


class _FakeScheduler:
    def __init__(self) -> None:
        self.pause_state = PauseState.UNPAUSED
        self.aborted = 0

    def set_pause_state(self, pause_state: PauseState) -> None:
        self.pause_state = pause_state

    def finish_requests(self, request_ids, finished_status):
        self.aborted += 1
        return []

    def has_requests(self) -> bool:
        return False


class _FakeExecutor:
    def __init__(self, error: Exception | None = None, asleep_on_error=False):
        self.is_sleeping = False
        self.error = error
        self.asleep_on_error = asleep_on_error
        self.sleep_calls: list[int] = []

    def sleep(self, level: int = 1) -> None:
        self.sleep_calls.append(level)
        if self.error is not None:
            self.is_sleeping = self.asleep_on_error
            raise self.error
        self.is_sleeping = True


def _engine_core(cls=EngineCore, executor=None, backend="cumem"):
    core = object.__new__(cls)
    core.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(sleep_mode_backend=backend)
    )
    core.scheduler = _FakeScheduler()
    core.model_executor = executor or _FakeExecutor()
    core._finish_pause = lambda clear_cache: None
    if isinstance(core, EngineCoreProc):
        # Busy engine: the pause (and so the executor sleep) is deferred until
        # the engine goes idle.
        core.engines_running = True
        core.batch_queue = None
        core._idle_state_callbacks = []
        core._send_abort_outputs = lambda aborted_reqs: None
    return core


def _sleep(core, level: int = 1) -> None:
    """Run ``core.sleep`` to completion, driving a deferred pause if needed."""
    result = core.sleep(level)
    if isinstance(result, Future):
        assert not result.done()
        core.engines_running = False
        core._notify_idle_state_callbacks()
        result.result(timeout=0)
    else:
        assert result is None


@pytest.mark.parametrize("cls", [EngineCore, EngineCoreProc])
def test_successful_sleep_pauses_scheduler(cls):
    core = _engine_core(cls)
    _sleep(core)
    assert core.model_executor.sleep_calls == [1]
    assert core.scheduler.pause_state == PauseState.PAUSED_NEW
    assert core.is_sleeping()


@pytest.mark.parametrize("cls", [EngineCore, EngineCoreProc])
def test_unsupported_level_rejected_before_pause(cls, level1_only_backend):
    core = _engine_core(cls, backend=level1_only_backend)
    with pytest.raises(VLLMValidationError, match="sleep level 2"):
        core.sleep(2)
    assert core.scheduler.aborted == 0
    assert core.scheduler.pause_state == PauseState.UNPAUSED
    assert core.model_executor.sleep_calls == []
    assert not core.is_sleeping()


@pytest.mark.parametrize("cls", [EngineCore, EngineCoreProc])
def test_failed_executor_sleep_resumes_scheduler(cls):
    core = _engine_core(cls, _FakeExecutor(RuntimeError("suspend failed")))
    with pytest.raises(RuntimeError, match="suspend failed"):
        _sleep(core)
    assert core.model_executor.sleep_calls == [1]
    assert core.scheduler.pause_state == PauseState.UNPAUSED
    assert not core.is_sleeping()


@pytest.mark.parametrize("cls", [EngineCore, EngineCoreProc])
def test_executor_left_asleep_keeps_scheduler_paused(cls):
    """If the executor could not be restored it stays asleep, and the engine
    must keep reporting sleeping until wake_up brings the memory back."""
    executor = _FakeExecutor(RuntimeError("suspend failed"), asleep_on_error=True)
    core = _engine_core(cls, executor)
    with pytest.raises(RuntimeError, match="suspend failed"):
        _sleep(core)
    assert core.scheduler.pause_state == PauseState.PAUSED_NEW
    assert core.is_sleeping()


def test_dp_engine_leaves_resume_to_wake_up():
    """Resuming a DP engine is collective, so a rank whose sleep failed alone
    must not resume by itself."""
    core = _engine_core(DPEngineCoreProc, _FakeExecutor(RuntimeError("boom")))
    core.scheduler.set_pause_state(PauseState.PAUSED_NEW)

    def unexpected_resume():
        raise AssertionError("resume_scheduler must not be called")

    core.resume_scheduler = unexpected_resume
    with pytest.raises(RuntimeError, match="boom"):
        core._sleep_executor(1)
    assert core.scheduler.pause_state == PauseState.PAUSED_NEW


def _executor(fail: set[str]) -> tuple[UniProcExecutor, list[str]]:
    executor = object.__new__(UniProcExecutor)
    executor.is_sleeping = False
    executor.sleeping_tags = set()
    calls: list[str] = []

    def collective_rpc(method, kwargs=None):
        calls.append(method)
        if method in fail:
            raise RuntimeError(f"{method} failed")

    executor.collective_rpc = collective_rpc
    return executor, calls


def test_executor_restores_memory_after_failed_level_1_sleep():
    executor, calls = _executor(fail={"sleep"})
    with pytest.raises(RuntimeError, match="sleep failed"):
        executor.sleep(1)
    assert calls == ["sleep", "wake_up"]
    assert not executor.is_sleeping
    assert executor.sleeping_tags == set()


def test_executor_stays_asleep_after_failed_level_2_sleep():
    # Level 2 may have discarded the weights, so waking must follow the
    # caller's level-2 protocol (wake, reload weights) rather than happen here.
    executor, calls = _executor(fail={"sleep"})
    with pytest.raises(RuntimeError, match="sleep failed"):
        executor.sleep(2)
    assert calls == ["sleep"]
    assert executor.is_sleeping
    assert executor.sleeping_tags == {"weights", "kv_cache"}


def test_executor_stays_asleep_when_restore_fails():
    executor, calls = _executor(fail={"sleep", "wake_up"})
    with pytest.raises(RuntimeError, match="sleep failed"):
        executor.sleep(1)
    assert calls == ["sleep", "wake_up"]
    assert executor.is_sleeping

    # A later wake_up retries the restore and clears the state.
    executor.collective_rpc = lambda method, kwargs=None: None
    executor.wake_up()
    assert not executor.is_sleeping
