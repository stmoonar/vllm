# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for ``--abort-return-token-ids``: a response that ends with
``finish_reason="abort"`` carries the prompt token IDs and the token IDs
generated so far, so that a router can continue the request elsewhere by
sending ``prompt_token_ids + generated_token_ids`` as a token-ID prompt."""

import asyncio
import json

import pytest

from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.launchers.cli_args import make_arg_parser
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.utils.argparse_utils import FlexibleArgumentParser

PROMPT_IDS = [1, 2, 3]


def _output(token_ids, text, finish_reason=None, prompt_ids=PROMPT_IDS):
    return RequestOutput(
        request_id="r",
        prompt="p",
        prompt_token_ids=prompt_ids,
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text=text,
                token_ids=token_ids,
                cumulative_logprob=None,
                logprobs=None,
                finish_reason=finish_reason,
            )
        ],
        finished=finish_reason is not None,
    )


# Streaming deltas: two tokens, then the abort (prompt IDs only come first).
ABORTED_STREAM = [
    _output([7], "a"),
    _output([8], "b", prompt_ids=None),
    _output([], "", "abort", prompt_ids=None),
]
ABORTED_BEFORE_FIRST_TOKEN = [_output([], "", "abort")]


def _completion_serving(abort_ids: bool) -> OpenAIServingCompletion:
    serving = OpenAIServingCompletion.__new__(OpenAIServingCompletion)
    serving.abort_return_token_ids = abort_ids
    serving.enable_prompt_tokens_details = False
    serving.enable_force_include_usage = False
    serving.enable_per_request_metrics = False
    serving.system_fingerprint = None
    return serving


def _chat_serving(abort_ids: bool) -> OpenAIServingChat:
    serving = OpenAIServingChat.__new__(OpenAIServingChat)
    serving.abort_return_token_ids = abort_ids
    serving.parser_cls = None
    serving.response_role = "assistant"
    serving.enable_auto_tools = False
    serving.enable_prompt_tokens_details = False
    serving.enable_force_include_usage = False
    serving.enable_per_request_metrics = False
    serving.enable_log_outputs = False
    serving.request_logger = None
    serving.system_fingerprint = None
    serving._include_reasoning_tokens_details = False
    return serving


async def _results(outputs, indexed: bool):
    for out in outputs:
        yield (0, out) if indexed else out


def _events(chunks: list[str]) -> list[dict]:
    events = []
    for chunk in chunks:
        assert chunk.startswith("data: "), chunk
        payload = chunk[len("data: ") :].strip()
        if payload != "[DONE]":
            events.append(json.loads(payload))
    return events


async def _collect(agen) -> list[str]:
    return [chunk async for chunk in agen]


def _completion_stream(abort_ids: bool, outputs) -> list[dict]:
    serving = _completion_serving(abort_ids)
    request = CompletionRequest(model="m", prompt="p", max_tokens=16, stream=True)
    agen = serving.completion_stream_generator(
        request,
        [],
        _results(outputs, indexed=True),
        "cmpl-r",
        0,
        "m",
        num_prompts=1,
        tokenizer=None,
        request_metadata=RequestResponseMetadata(request_id="cmpl-r"),
    )
    return _events(asyncio.run(_collect(agen)))


def _chat_stream(abort_ids: bool, outputs) -> list[dict]:
    serving = _chat_serving(abort_ids)
    request = ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    agen = serving.chat_completion_stream_generator(
        request,
        _results(outputs, indexed=False),
        "chatcmpl-r",
        "m",
        conversation=[],
        tokenizer=None,
        request_metadata=RequestResponseMetadata(request_id="chatcmpl-r"),
    )
    return _events(asyncio.run(_collect(agen)))


def _choices(events: list[dict]) -> list[dict]:
    return [choice for event in events for choice in event["choices"]]


def test_completion_stream_abort_chunk_carries_token_ids():
    choices = _choices(_completion_stream(True, ABORTED_STREAM))
    *tokens, final = choices
    assert [c["text"] for c in tokens] == ["a", "b"]
    assert all("generated_token_ids" not in c for c in tokens)
    assert final["finish_reason"] == "abort"
    assert final["generated_token_ids"] == [7, 8]
    assert final["prompt_token_ids"] == PROMPT_IDS


def test_completion_stream_unchanged_by_default():
    choices = _choices(_completion_stream(False, ABORTED_STREAM))
    assert choices[-1]["finish_reason"] == "abort"
    assert all("generated_token_ids" not in c for c in choices)
    assert all(c.get("prompt_token_ids") is None for c in choices)


def test_completion_stream_abort_before_first_token():
    # By default an abort before the first token sends no chunk at all.
    assert _choices(_completion_stream(False, ABORTED_BEFORE_FIRST_TOKEN)) == []
    (final,) = _choices(_completion_stream(True, ABORTED_BEFORE_FIRST_TOKEN))
    assert final["finish_reason"] == "abort"
    assert final["generated_token_ids"] == []
    assert final["prompt_token_ids"] == PROMPT_IDS


def test_completion_stream_other_finish_reasons_unchanged():
    outputs = [_output([7], "a"), _output([8], "b", "length", prompt_ids=None)]
    choices = _choices(_completion_stream(True, outputs))
    assert choices[-1]["finish_reason"] == "length"
    assert all("generated_token_ids" not in c for c in choices)


@pytest.mark.parametrize(
    "abort_ids, finish_reason, expected",
    [
        (True, "abort", ([7, 8], PROMPT_IDS)),
        (False, "abort", (None, None)),
        (True, "length", (None, None)),
    ],
)
def test_completion_non_streaming(abort_ids, finish_reason, expected):
    serving = _completion_serving(abort_ids)
    request = CompletionRequest(model="m", prompt="p", max_tokens=16)
    response = serving.request_output_to_completion_response(
        [_output([7, 8], "ab", finish_reason)],
        request,
        "cmpl-r",
        0,
        "m",
        None,
        RequestResponseMetadata(request_id="cmpl-r"),
    )
    (choice,) = response.choices
    assert choice.finish_reason == finish_reason
    assert (choice.token_ids, choice.prompt_token_ids) == expected


def test_chat_stream_abort_chunk_carries_token_ids():
    events = _chat_stream(True, ABORTED_STREAM)
    final_event = events[-1]
    (final,) = final_event["choices"]
    assert final["finish_reason"] == "abort"
    assert final["generated_token_ids"] == [7, 8]
    assert final_event["prompt_token_ids"] == PROMPT_IDS
    assert all("generated_token_ids" not in c for c in _choices(events[:-1]))


def test_chat_stream_unchanged_by_default():
    events = _chat_stream(False, ABORTED_STREAM)
    assert _choices(events)[-1]["finish_reason"] == "abort"
    assert all("generated_token_ids" not in c for c in _choices(events))
    assert all(e.get("prompt_token_ids") is None for e in events)


def test_chat_stream_abort_before_first_token():
    events = _chat_stream(True, ABORTED_BEFORE_FIRST_TOKEN)
    final = _choices(events)[-1]
    assert final["finish_reason"] == "abort"
    assert final["generated_token_ids"] == []
    assert events[-1]["prompt_token_ids"] == PROMPT_IDS


@pytest.mark.parametrize(
    "abort_ids, finish_reason, expected",
    [
        (True, "abort", ([7, 8], PROMPT_IDS)),
        (False, "abort", (None, None)),
        (True, "length", (None, None)),
    ],
)
def test_chat_non_streaming(abort_ids, finish_reason, expected):
    serving = _chat_serving(abort_ids)
    request = ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}]
    )
    response = asyncio.run(
        serving.chat_completion_full_generator(
            request,
            _results([_output([7, 8], "ab", finish_reason)], indexed=False),
            "chatcmpl-r",
            "m",
            conversation=[],
            tokenizer=None,
            request_metadata=RequestResponseMetadata(request_id="chatcmpl-r"),
        )
    )
    (choice,) = response.choices
    assert choice.finish_reason == finish_reason
    assert choice.message.content == "ab"
    assert (choice.token_ids, response.prompt_token_ids) == expected


def test_cli_flag():
    parser = make_arg_parser(FlexibleArgumentParser())
    assert parser.parse_args([]).abort_return_token_ids is False
    args = parser.parse_args(["--abort-return-token-ids"])
    assert args.abort_return_token_ids is True
