# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests: an invalid /sleep request is a client error (HTTP 400)
and is rejected before the engine pauses scheduling."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.device_allocator.sleep_mode_backend import (
    CuMemBackend,
    SleepModeBackendFactory,
)
from vllm.entrypoints.serve.dev.sleep.api_router import attach_router
from vllm.entrypoints.serve.exception_handling.register import init_exception_handler
from vllm.v1.engine.async_llm import AsyncLLM


class _Level1OnlyBackend(CuMemBackend):
    @classmethod
    def supported_sleep_levels(cls) -> tuple[int, ...]:
        return (1,)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setitem(
        SleepModeBackendFactory._registry, "level1_only", lambda: _Level1OnlyBackend
    )
    engine = object.__new__(AsyncLLM)
    engine.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(sleep_mode_backend="level1_only")
    )
    engine.engine_core = MagicMock()
    engine.engine_core.sleep_async = AsyncMock()
    engine.renderer = MagicMock()
    engine.renderer.clear_mm_cache_async = AsyncMock()
    engine.logger_manager = None

    app = FastAPI()
    app.state.engine_client = engine
    app.state.args = SimpleNamespace(log_error_stack=False)
    init_exception_handler(app)
    attach_router(app)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client, engine.engine_core.sleep_async


@pytest.mark.parametrize(
    "params, param",
    [({"level": "2"}, "level"), ({"level": "1", "mode": "bogus"}, "mode")],
)
def test_invalid_sleep_request_returns_400(client, params, param):
    test_client, sleep_async = client
    response = test_client.post("/sleep", params=params)
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"] == param
    sleep_async.assert_not_called()


def test_supported_sleep_level_reaches_engine(client):
    test_client, sleep_async = client
    response = test_client.post("/sleep", params={"level": "1"})
    assert response.status_code == 200, response.text
    sleep_async.assert_awaited_once_with(1, "abort")
