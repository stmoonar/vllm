# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for how EngineCore and Executor handle a failed sleep.

A sleep pauses the scheduler before the executor releases GPU memory. If the
executor then fails, the engine must not stay paused while awake: it would
report sleeping and hold every new request until a manual wake_up.

Also covers ``--sleep-reject-new`` in the engine (new requests that reach a
paused scheduler) and publishing ``vllm:num_requests_paused`` while paused.
"""

import queue
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.device_allocator.sleep_mode_backend import (
    CuMemBackend,
    SleepModeBackendFactory,
)
from vllm.exceptions import VLLMValidationError
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import (
    SLEEP_REJECTED_STOP_REASON,
    EngineCoreRequestType,
    FinishReason,
)
from vllm.v1.engine.core import (
    DPEngineCoreProc,
    EngineCore,
    EngineCoreProc,
    EngineShutdownState,
)
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats


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


# ---------------------------------------------------------------------------
# --sleep-reject-new and vllm:num_requests_paused (engine side)
# ---------------------------------------------------------------------------


class _QueueScheduler(_FakeScheduler):
    """Scheduler stand-in that keeps new requests in a waiting queue."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: dict[str, object] = {}
        self.waiting: list[object] = []

    def add_request(self, request) -> None:
        self.requests[request.request_id] = request
        self.waiting.append(request)

    def get_num_paused_requests(self) -> int:
        return 0 if self.pause_state == PauseState.UNPAUSED else len(self.waiting)

    def make_stats(self) -> SchedulerStats:
        return SchedulerStats(
            num_waiting_reqs=len(self.waiting),
            num_paused_reqs=self.get_num_paused_requests(),
        )


def _proc_core(reject_new: bool, log_stats: bool = True) -> EngineCoreProc:
    core = _engine_core(EngineCoreProc)
    core.engines_running = False
    core.scheduler = _QueueScheduler()
    core.sleep_reject_new = reject_new
    core.log_stats = log_stats
    core.shutdown_state = EngineShutdownState.RUNNING
    core.input_queue = queue.Queue()
    core.output_queue = queue.Queue()
    core.aborts_queue = queue.Queue()
    core.process_input_queue_block = False
    core._last_published_num_paused = 0
    # Admission itself is not under test here.
    core.add_request = lambda request, request_wave=0: core.scheduler.add_request(
        request
    )
    return core


def _new_request(request_id: str = "r1", **kwargs):
    fields = dict(
        request_id=request_id,
        client_index=3,
        pooling_params=None,
        abort_immediately=False,
    )
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def _add(core, request) -> None:
    core.input_queue.put((EngineCoreRequestType.ADD, (request, 0)))
    core._process_input_queue()


def _drain(q: queue.Queue) -> list:
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    return items


def test_paused_engine_rejects_new_request_when_enabled():
    core = _proc_core(reject_new=True)
    core.scheduler.set_pause_state(PauseState.PAUSED_NEW)
    _add(core, _new_request())

    assert core.scheduler.requests == {}
    outputs = [eco for _, eco in _drain(core.output_queue) if eco.outputs]
    assert len(outputs) == 1
    (output,) = outputs[0].outputs
    assert output.request_id == "r1"
    assert output.finish_reason == FinishReason.ABORT
    assert output.stop_reason == SLEEP_REJECTED_STOP_REASON
    assert outputs[0].finished_requests == {"r1"}


@pytest.mark.parametrize("pause_state", [PauseState.PAUSED_NEW, PauseState.PAUSED_ALL])
def test_paused_engine_queues_new_request_by_default(pause_state):
    core = _proc_core(reject_new=False)
    core.scheduler.set_pause_state(pause_state)
    _add(core, _new_request())
    assert list(core.scheduler.requests) == ["r1"]
    assert not any(eco.outputs for _, eco in _drain(core.output_queue))


@pytest.mark.parametrize(
    "request_kwargs",
    [
        {"pooling_params": object()},
        {"abort_immediately": True},
    ],
)
def test_reject_new_skips_non_generation_adds(request_kwargs):
    core = _proc_core(reject_new=True)
    core.scheduler.set_pause_state(PauseState.PAUSED_NEW)
    _add(core, _new_request(**request_kwargs))
    assert list(core.scheduler.requests) == ["r1"]


def test_reject_new_keeps_streaming_input_sessions():
    """The next input chunk of a session admitted before the pause is not a
    new request."""
    core = _proc_core(reject_new=True)
    _add(core, _new_request())
    core.scheduler.set_pause_state(PauseState.PAUSED_NEW)
    _add(core, _new_request())
    assert len(core.scheduler.waiting) == 2
    assert not any(eco.outputs for _, eco in _drain(core.output_queue))


def test_unpaused_engine_accepts_with_reject_new():
    core = _proc_core(reject_new=True)
    _add(core, _new_request())
    assert list(core.scheduler.requests) == ["r1"]


def _published_paused_counts(core) -> list[int]:
    return [
        eco.scheduler_stats.num_paused_reqs
        for _, eco in _drain(core.output_queue)
        if eco.scheduler_stats is not None
    ]


def test_paused_queue_is_published_while_engine_idles():
    """A paused engine does not step, so stats must be sent when requests
    queue up behind the pause, and once more when the pause ends."""
    core = _proc_core(reject_new=False)
    _add(core, _new_request("a"))
    assert _published_paused_counts(core) == []

    core.scheduler.waiting.clear()
    core.scheduler.set_pause_state(PauseState.PAUSED_NEW)
    _add(core, _new_request("b"))
    _add(core, _new_request("c"))
    assert _published_paused_counts(core) == [1, 2]

    # Unchanged count: nothing new to publish.
    core._maybe_publish_paused_stats()
    assert _published_paused_counts(core) == []

    core.scheduler.set_pause_state(PauseState.UNPAUSED)
    core._maybe_publish_paused_stats()
    assert _published_paused_counts(core) == [0]


def test_paused_stats_not_published_without_log_stats():
    core = _proc_core(reject_new=False, log_stats=False)
    core.scheduler.set_pause_state(PauseState.PAUSED_NEW)
    _add(core, _new_request())
    assert _published_paused_counts(core) == []


def _real_scheduler(pause_state: PauseState, waiting: int, skipped: int = 0):
    scheduler = object.__new__(Scheduler)
    scheduler._pause_state = pause_state
    scheduler.waiting = [object()] * waiting
    scheduler.skipped_waiting = [object()] * skipped
    scheduler.num_waiting_for_streaming_input = 0
    return scheduler


def test_scheduler_counts_paused_requests():
    assert _real_scheduler(PauseState.UNPAUSED, 3).get_num_paused_requests() == 0
    assert _real_scheduler(PauseState.PAUSED_NEW, 3, 1).get_num_paused_requests() == 4
    assert _real_scheduler(PauseState.PAUSED_ALL, 2).get_num_paused_requests() == 2


def test_scheduler_stats_carry_paused_requests():
    scheduler = _real_scheduler(PauseState.PAUSED_NEW, 2)
    scheduler.log_stats = True
    scheduler.running = []
    scheduler.connector_prefix_cache_stats = None
    scheduler.kv_metrics_collector = None
    scheduler.kv_cache_manager = SimpleNamespace(
        make_prefix_cache_stats=PrefixCacheStats, usage=0.0
    )
    stats = scheduler.make_stats()
    assert stats.num_waiting_reqs == 2
    assert stats.num_paused_reqs == 2


def test_prometheus_logger_exports_paused_gauge():
    stat_logger = MagicMock(gauge_lora_info=None, kv_cache_metrics_enabled=False)
    stats = SchedulerStats(num_waiting_reqs=3, num_paused_reqs=3)
    PrometheusStatLogger.record(stat_logger, stats, None, engine_idx=0)
    stat_logger.gauge_scheduler_paused[0].set.assert_called_once_with(3)
    stat_logger.gauge_scheduler_waiting[0].set.assert_called_once_with(3)
