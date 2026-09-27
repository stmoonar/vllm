# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for ``--sleep-reject-new`` in the API server: while the
engine sleeps or is paused, new requests get HTTP 503 with ``Retry-After``
instead of queueing until the engine wakes up."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from vllm.entrypoints.launchers.cli_args import make_arg_parser
from vllm.entrypoints.serve.dev.sleep.api_router import attach_router
from vllm.entrypoints.serve.exception_handling.register import init_exception_handler
from vllm.exceptions import EngineSleepingError
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.engine import SLEEP_REJECTED_STOP_REASON
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.output_processor import RequestOutputCollector


def _engine(reject_new: bool = True) -> AsyncLLM:
    engine = object.__new__(AsyncLLM)
    engine.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(sleep_mode_backend="cumem")
    )
    engine.scheduler_config = SimpleNamespace(
        max_num_queued_reqs=None, max_num_queued_tokens=None
    )
    engine._reject_new_while_paused = reject_new
    engine.engine_core = MagicMock()
    for method in (
        "sleep_async",
        "wake_up_async",
        "pause_scheduler_async",
        "resume_scheduler_async",
    ):
        setattr(engine.engine_core, method, AsyncMock())
    engine.engine_core.is_sleeping_async = AsyncMock(return_value=False)
    engine.renderer = MagicMock()
    engine.renderer.clear_mm_cache_async = AsyncMock()
    engine.logger_manager = None
    engine.log_requests = False
    return engine


def _app(engine: AsyncLLM) -> TestClient:
    app = FastAPI()
    app.state.engine_client = engine
    app.state.args = SimpleNamespace(log_error_stack=False)
    init_exception_handler(app)
    attach_router(app)

    @app.post("/v1/probe")
    async def probe(raw_request: Request):
        # What the OpenAI serving layer runs before starting a response.
        raw_request.app.state.engine_client.check_admission(1)
        return {"ok": True}

    return TestClient(app, raise_server_exceptions=False)


def _assert_rejected(response) -> None:
    assert response.status_code == 503, response.text
    assert response.headers["retry-after"] == "1"
    error = response.json()["error"]
    assert error["type"] == "EngineSleeping"
    assert error["code"] == 503


def test_rejects_new_requests_between_sleep_and_wake_up():
    engine = _engine()
    with _app(engine) as client:
        assert client.post("/v1/probe").status_code == 200

        assert client.post("/sleep", params={"level": "1"}).status_code == 200
        engine.engine_core.is_sleeping_async.return_value = True
        _assert_rejected(client.post("/v1/probe"))
        _assert_rejected(client.post("/v1/probe"))

        engine.engine_core.is_sleeping_async.return_value = False
        assert client.post("/wake_up").status_code == 200
        assert client.post("/v1/probe").status_code == 200


def test_level_0_sleep_also_rejects():
    engine = _engine()
    with _app(engine) as client:
        assert client.post("/sleep", params={"level": "0"}).status_code == 200
        engine.engine_core.is_sleeping_async.return_value = True
        _assert_rejected(client.post("/v1/probe"))


def test_default_queues_requests_during_sleep():
    engine = _engine(reject_new=False)
    with _app(engine) as client:
        assert client.post("/sleep", params={"level": "1"}).status_code == 200
        assert client.post("/v1/probe").status_code == 200
        assert client.post("/wake_up").status_code == 200
    engine.engine_core.is_sleeping_async.assert_not_called()


def test_partial_wake_up_keeps_rejecting():
    engine = _engine()
    engine.engine_core.is_sleeping_async.return_value = True
    with _app(engine) as client:
        client.post("/sleep", params={"level": "1"})
        response = client.post("/wake_up", params={"tags": "weights"})
        assert response.status_code == 200
        _assert_rejected(client.post("/v1/probe"))


def test_failed_sleep_reopens_admission_when_engine_resumed():
    engine = _engine()
    engine.engine_core.sleep_async.side_effect = RuntimeError("suspend failed")
    with _app(engine) as client:
        assert client.post("/sleep", params={"level": "1"}).status_code == 500
        assert client.post("/v1/probe").status_code == 200


def test_admission_closes_before_sleep_rpc_is_sent():
    """Requests that pass the check are sent before the sleep RPC, so the
    engine handles them with the pause mode; later ones are rejected."""
    engine = _engine()
    seen = []

    def check_closed(*args):
        with pytest.raises(EngineSleepingError):
            engine.check_admission(1)
        seen.append(args)

    engine.renderer.clear_mm_cache_async.side_effect = check_closed
    engine.engine_core.sleep_async.side_effect = check_closed
    asyncio.run(engine.sleep(1))
    assert len(seen) == 2


def test_stale_wake_up_refresh_is_ignored():
    """A sleep that starts while wake_up reads the engine state wins."""
    engine = _engine()
    asyncio.run(engine.sleep(1))

    async def racing_is_sleeping():
        engine._pause_epoch += 1  # a concurrent /sleep call
        engine._engine_paused = True
        return False

    engine.engine_core.is_sleeping_async.side_effect = racing_is_sleeping
    asyncio.run(engine.wake_up())
    with pytest.raises(EngineSleepingError):
        engine.check_admission(1)


def test_stale_paused_state_is_reread_in_background():
    """With several API server processes, another one may wake the engine
    up. A rejected request makes this one re-read the engine state (at most
    every 0.5 s), so it stops rejecting within about one Retry-After."""
    engine = _engine()

    async def run():
        await engine.sleep(1)
        is_sleeping = engine.engine_core.is_sleeping_async
        is_sleeping.return_value = True
        with pytest.raises(EngineSleepingError):
            engine.check_admission(1)
        await engine._paused_refresh_task
        with pytest.raises(EngineSleepingError):
            engine.check_admission(1)  # still asleep; refresh rate-limited
        assert is_sleeping.await_count == 1

        is_sleeping.return_value = False  # woken up via another API server
        engine._last_paused_refresh = 0.0
        with pytest.raises(EngineSleepingError):
            engine.check_admission(1)
        await engine._paused_refresh_task
        engine.check_admission(1)

    asyncio.run(run())


def test_no_background_refresh_during_sleep_call():
    """While a sleep call is in flight the engine may not be paused yet, so
    a background re-read must not reopen admission."""
    engine = _engine()

    async def sleep_rpc(*args):
        with pytest.raises(EngineSleepingError):
            engine.check_admission(1)
        assert engine._paused_refresh_task is None

    engine.engine_core.sleep_async.side_effect = sleep_rpc
    asyncio.run(engine.sleep(1))
    with pytest.raises(EngineSleepingError):
        engine.check_admission(1)


def test_pause_and_resume_generation():
    engine = _engine()
    asyncio.run(engine.pause_generation(mode="keep", clear_cache=False))
    with pytest.raises(EngineSleepingError):
        engine.check_admission(1)
    asyncio.run(engine.resume_generation())
    engine.check_admission(1)


def _final_output(stop_reason) -> RequestOutput:
    return RequestOutput(
        request_id="r",
        prompt="p",
        prompt_token_ids=[1, 2],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text="",
                token_ids=[],
                cumulative_logprob=None,
                logprobs=None,
                finish_reason="abort",
                stop_reason=stop_reason,
            )
        ],
        finished=True,
    )


async def _generate(engine: AsyncLLM, output: RequestOutput) -> list[RequestOutput]:
    collector = RequestOutputCollector(RequestOutputKind.FINAL_ONLY, "r")
    collector.put(output)
    engine.add_request = AsyncMock(return_value=collector)
    return [out async for out in engine.generate("p", SamplingParams(), "r")]


def test_engine_side_rejection_becomes_engine_sleeping_error():
    """A request that reached the engine after it paused (e.g. sent by another
    API server) is rejected there and surfaces as the same 503 error."""
    engine = _engine()
    with pytest.raises(EngineSleepingError):
        asyncio.run(_generate(engine, _final_output(SLEEP_REJECTED_STOP_REASON)))


def test_ordinary_abort_is_returned():
    engine = _engine()
    (out,) = asyncio.run(_generate(engine, _final_output(None)))
    assert out.outputs[0].finish_reason == "abort"


def test_cli_flag():
    parser = make_arg_parser(FlexibleArgumentParser())
    assert parser.parse_args([]).sleep_reject_new is False
    assert parser.parse_args(["--sleep-reject-new"]).sleep_reject_new is True
