# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU self-check: does vLLM tokenize a model exactly like its tokenizer.json?

For each model directory, compares the tokenizer returned by vLLM's
``get_tokenizer`` with ``tokenizers.Tokenizer.from_file(<dir>/tokenizer.json)``:

* encode (no special tokens) on samples with repeated spaces, newlines, tabs,
  CJK, emoji, accented letters and code;
* decode of the reference ids, the decode round trip, and incremental
  (DecodeStream) decoding through vLLM's backend tokenizer, which is what the
  serving detokenizer uses;
* the ids of the rendered chat template (if the model has one).

The BOS policy (``add_bos_token`` in tokenizer_config.json vs. what
``encode(add_special_tokens=True)`` does) is reported as a warning only.

Exit status is 0 when every model matches, 1 otherwise.

Usage: python tools/check_tokenizer_consistency.py MODEL_DIR [MODEL_DIR ...]
"""

import argparse
import json
import os
import sys
from pathlib import Path

SAMPLES = [
    "The capital of France is Paris.",
    "two  spaces,   three spaces,    four",
    " leading space and trailing space ",
    "line one\nline two\n\n\nafter blank lines\n",
    "\tTabbed\tcolumns\t\tend",
    "你好，世界！这是一个中文句子。",
    "Emoji 😀🚀 and flags 🇫🇷 mixed in.",
    "café naïve résumé Ångström",
    "def add(x, y):\n    return x + y  # sum\n\nprint(add(1, 2))\n",
    '{"key": [1, 2, 3], "nested": {"a": null}}',
    "Mixed 中文 and English\twith\ttabs 😀 é\n  indented",
]

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Name three  capital cities.\nAnswer briefly."},
    {"role": "assistant", "content": "Paris, Berlin,  Tokyo."},
    {"role": "user", "content": "再说一个，用中文。\tThanks 😀"},
]


def _get_tokenizer():
    try:
        from vllm.tokenizers import get_tokenizer
    except ImportError:  # older vLLM layouts
        from vllm.transformers_utils.tokenizer import get_tokenizer
    return get_tokenizer


def _as_ids(result):
    if hasattr(result, "keys") and "input_ids" in result:
        result = result["input_ids"]
    return list(result)


def _stream_decode(backend, prefix_ids, ids):
    from tokenizers.decoders import DecodeStream

    stream = DecodeStream(ids=list(prefix_ids), skip_special_tokens=False)
    return "".join((stream.step(backend, t) or "") for t in ids)


def check_model(model_dir, args):
    from tokenizers import Tokenizer

    failures, warnings = [], []
    raw = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    tok = _get_tokenizer()(
        model_dir,
        tokenizer_mode=args.tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
    )
    backend = getattr(tok, "backend_tokenizer", None) or getattr(
        tok, "_tokenizer", None
    )
    info = {"class": type(tok).__name__}

    for text in SAMPLES:
        ref = raw.encode(text, add_special_tokens=False).ids
        got = list(tok.encode(text, add_special_tokens=False))
        if got != ref:
            failures.append(f"encode {text!r}: {len(got)} ids vs {len(ref)} ref")
            continue
        ref_text = raw.decode(ref)
        if tok.decode(ref) != ref_text:
            failures.append(f"decode {text!r}: {tok.decode(ref)!r} vs {ref_text!r}")
        if ref_text == text and tok.decode(got) != text:
            failures.append(f"round trip {text!r}: {tok.decode(got)!r}")
        if backend is not None:
            streamed = _stream_decode(backend, [], ref)
            if streamed != _stream_decode(raw, [], ref):
                failures.append(f"stream decode {text!r}: {streamed!r}")

    chat_template = getattr(tok, "chat_template", None)
    if chat_template:
        rendered = tok.apply_chat_template(
            MESSAGES, add_generation_prompt=True, tokenize=False
        )
        chat_ids = _as_ids(
            tok.apply_chat_template(MESSAGES, add_generation_prompt=True, tokenize=True)
        )
        ref = raw.encode(rendered, add_special_tokens=False).ids
        info["chat_ids"] = len(chat_ids)
        if chat_ids != ref:
            first = next(
                (i for i, (a, b) in enumerate(zip(chat_ids, ref)) if a != b),
                min(len(chat_ids), len(ref)),
            )
            failures.append(
                f"chat template ids: {len(chat_ids)} vs {len(ref)} ref, "
                f"first diff at {first}"
            )
    else:
        warnings.append("no chat template; chat check skipped")

    cfg_path = Path(model_dir) / "tokenizer_config.json"
    add_bos = None
    if cfg_path.exists():
        add_bos = json.loads(cfg_path.read_text(encoding="utf-8")).get("add_bos_token")
    bos_id = getattr(tok, "bos_token_id", None)
    with_special = list(tok.encode(SAMPLES[0], add_special_tokens=True))
    adds_bos = bos_id is not None and with_special[:1] == [bos_id]
    info["add_bos_token(config)"] = add_bos
    info["encode_adds_bos"] = adds_bos
    if add_bos is not None and bool(add_bos) != adds_bos:
        warnings.append(
            f"tokenizer_config add_bos_token={add_bos} but encode adds BOS={adds_bos}"
        )
    return failures, warnings, info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model_dirs", nargs="+", help="local model directories")
    parser.add_argument("--tokenizer-mode", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(argv)

    import vllm

    print(f"vllm {getattr(vllm, '__version__', '?')}")
    all_ok = True
    for model_dir in args.model_dirs:
        name = Path(model_dir).name
        try:
            failures, warnings, info = check_model(model_dir, args)
        except Exception as e:  # report and keep going
            failures, warnings, info = [f"error: {type(e).__name__}: {e}"], [], {}
        status = "PASS" if not failures else "FAIL"
        all_ok &= not failures
        print(f"[{status}] {name} {json.dumps(info, ensure_ascii=False)}")
        for line in failures:
            print(f"    FAIL {line}")
        for line in warnings:
            print(f"    WARN {line}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
