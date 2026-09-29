# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from tokenizers import Tokenizer
from transformers import AutoConfig, AutoTokenizer, LlamaTokenizer
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.hf import CachedHfTokenizer
from vllm.tokenizers.registry import (
    TokenizerRegistry,
    _has_byte_level,
    _is_sentencepiece_class_over_byte_level_bpe,
    cached_get_tokenizer,
    cached_resolve_tokenizer_args,
    cached_tokenizer_from_config,
    get_tokenizer,
    resolve_tokenizer_args,
)
from vllm.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeConfig


class TestTokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> "TestTokenizer":
        return TestTokenizer(path_or_repo_id)  # type: ignore

    def __init__(self, path_or_repo_id: str | Path) -> None:
        super().__init__()

        self.path_or_repo_id = path_or_repo_id

    @property
    def bos_token_id(self) -> int:
        return 0

    @property
    def eos_token_id(self) -> int:
        return 1

    @property
    def pad_token_id(self) -> int:
        return 2

    @property
    def is_fast(self) -> bool:
        return True


@pytest.mark.parametrize("runner_type", ["generate", "pooling"])
def test_resolve_tokenizer_args_idempotent(runner_type):
    tokenizer_mode, tokenizer_name, args, kwargs = resolve_tokenizer_args(
        "facebook/opt-125m",
        runner_type=runner_type,
    )

    assert (tokenizer_mode, tokenizer_name, args, kwargs) == resolve_tokenizer_args(
        tokenizer_name, *args, **kwargs
    )


@pytest.mark.parametrize(
    ("tokenizer_mode", "input_kwargs"),
    [
        ("hf", {}),
        ("hf", {"mistral_format": False}),
        ("slow", {}),
        ("slow", {"mistral_format": False}),
    ],
)
def test_resolve_tokenizer_args_forces_hf_mistral_format_false(
    tokenizer_mode, input_kwargs
):
    resolved_mode, _, _, kwargs = resolve_tokenizer_args(
        "mistralai/Mistral-Nemo-Instruct-2407",
        tokenizer_mode=tokenizer_mode,
        **input_kwargs,
    )

    assert resolved_mode == "hf"
    assert kwargs["mistral_format"] is False


@pytest.mark.parametrize("tokenizer_mode", ["hf", "slow"])
def test_resolve_tokenizer_args_rejects_hf_mistral_format_true(tokenizer_mode):
    with pytest.raises(
        ValueError,
        match="mistral_format=True is not supported with tokenizer_mode='hf'",
    ):
        resolve_tokenizer_args(
            "mistralai/Mistral-Nemo-Instruct-2407",
            tokenizer_mode=tokenizer_mode,
            mistral_format=True,
        )


@pytest.mark.parametrize(
    ("transformers_version", "model_type", "native_filename", "expect_error"),
    [
        pytest.param("5.14.0", "mistral", "tekken.json", True),
        pytest.param("5.15.0", "mistral", "tekken.json", False),
        pytest.param("5.14.0", "mistral3", "tekken.json", True),
        pytest.param("5.15.0", "mistral3", "tekken.json", False),
        pytest.param("5.14.0", "mixtral", "tokenizer.model.v1", True),
        pytest.param("5.15.0", "mixtral", "tokenizer.model.v1", False),
        pytest.param("5.14.0", "mistral", None, False),
        pytest.param("5.14.0", "qwen3_5_moe", "tekken.json", False),
    ],
)
def test_get_tokenizer_handles_hf_mistral_transformers_compatibility(
    tmp_path: Path,
    transformers_version: str,
    model_type: str,
    native_filename: str | None,
    expect_error: bool,
):
    if native_filename is not None:
        (tmp_path / native_filename).touch()

    with (
        patch(
            "vllm.tokenizers.registry.get_config",
            return_value=SimpleNamespace(model_type=model_type),
        ),
        patch(
            "vllm.tokenizers.registry.transformers.__version__",
            transformers_version,
        ),
        patch.object(
            CachedHfTokenizer,
            "from_pretrained",
            return_value=SimpleNamespace(is_fast=True),
        ) as from_pretrained,
    ):
        if expect_error:
            with pytest.raises(ValueError, match="requires transformers>=5.15.0"):
                get_tokenizer(str(tmp_path), tokenizer_mode="hf")
        else:
            tokenizer = get_tokenizer(str(tmp_path), tokenizer_mode="hf")
            assert tokenizer.is_fast is True

    assert from_pretrained.called == (not expect_error)


def test_customized_tokenizer():
    TokenizerRegistry.register("test_tokenizer", __name__, TestTokenizer.__name__)

    tokenizer = TokenizerRegistry.load_tokenizer("test_tokenizer", "abc")
    assert isinstance(tokenizer, TestTokenizer)
    assert tokenizer.path_or_repo_id == "abc"
    assert tokenizer.bos_token_id == 0
    assert tokenizer.eos_token_id == 1
    assert tokenizer.pad_token_id == 2

    tokenizer = get_tokenizer("abc", tokenizer_mode="test_tokenizer")
    assert isinstance(tokenizer, TestTokenizer)
    assert tokenizer.path_or_repo_id == "abc"
    assert tokenizer.bos_token_id == 0
    assert tokenizer.eos_token_id == 1
    assert tokenizer.pad_token_id == 2


def test_cached_tokenizer_from_config_registers_local_config(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5_moe"}),
        encoding="utf-8",
    )

    model_config = SimpleNamespace(
        skip_tokenizer_init=False,
        tokenizer=str(tmp_path),
        runner_type="generate",
        tokenizer_mode="hf",
        tokenizer_revision=None,
        trust_remote_code=True,
        hf_config=Qwen3_5MoeConfig(),
    )

    registered_config = CONFIG_MAPPING._extra_content.pop("qwen3_5_moe", None)
    cached_get_tokenizer.cache_clear()
    cached_resolve_tokenizer_args.cache_clear()

    try:

        def fake_from_pretrained(path_or_repo_id: str, *args, **kwargs):
            passed_config = kwargs.pop("config")
            assert isinstance(passed_config, Qwen3_5MoeConfig)
            loaded_config = AutoConfig.from_pretrained(
                path_or_repo_id,
                trust_remote_code=False,
            )
            assert isinstance(loaded_config, Qwen3_5MoeConfig)
            return SimpleNamespace(is_fast=True)

        with (
            patch(
                "vllm.tokenizers.registry.logger.debug_once",
                lambda *args, **kwargs: None,
            ),
            patch(
                "vllm.tokenizers.hf.AutoTokenizer.from_pretrained",
                side_effect=fake_from_pretrained,
            ),
            patch(
                "vllm.tokenizers.hf.get_cached_tokenizer",
                side_effect=lambda tokenizer: tokenizer,
            ),
        ):
            tokenizer = cached_tokenizer_from_config(model_config)

        assert tokenizer.is_fast is True
    finally:
        cached_get_tokenizer.cache_clear()
        cached_resolve_tokenizer_args.cache_clear()
        CONFIG_MAPPING._extra_content.pop("qwen3_5_moe", None)
        if registered_config is not None:
            CONFIG_MAPPING._extra_content["qwen3_5_moe"] = registered_config


# --- Byte-level BPE checkpoints that declare a SentencePiece Llama class ---

_CORPUS = [
    "The capital of France is Paris.\nThe capital of Germany is Berlin.",
    "def f(x):\n\treturn x  +  1  # two  spaces",
    "caf\u00e9 na\u00efve \u4f60\u597d\u4e16\u754c \U0001f600 done",
] * 20
_SAMPLE = (
    "Hello  world,\n\n  indented line.\n"
    "\tcafé 你好 \U0001f600 x  +  1"
)


def _write_checkpoint(
    path: Path,
    *,
    model_type: str,
    tokenizer_class: str,
    byte_level: bool,
) -> Tokenizer:
    from tokenizers import decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, Qwen2Config

    special = ["<s>", "</s>", "<unk>"]
    if byte_level:
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=400,
            special_tokens=special,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
    else:
        # Llama-2 style SentencePiece BPE: Metaspace + byte fallback.
        tok = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True))
        tok.pre_tokenizer = pre_tokenizers.Metaspace()
        tok.decoder = decoders.Metaspace()
        trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=special)
    tok.train_from_iterator(_CORPUS, trainer)
    tok.save(str(path / "tokenizer.json"))

    config_cls = {"llama": LlamaConfig, "qwen2": Qwen2Config}[model_type]
    config_cls(
        vocab_size=tok.get_vocab_size(),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    ).save_pretrained(path)
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": tokenizer_class,
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
            }
        ),
        encoding="utf-8",
    )
    return tok


def _load(path: Path):
    cached_get_tokenizer.cache_clear()
    cached_resolve_tokenizer_args.cache_clear()
    with patch("vllm.tokenizers.registry.logger.info") as log_info:
        tokenizer = get_tokenizer(str(path))
    overridden = any(
        "byte-level BPE" in str(c.args[0]) for c in log_info.call_args_list
    )
    return tokenizer, overridden


def test_byte_level_llama_class_is_overridden(tmp_path: Path):
    raw = _write_checkpoint(
        tmp_path,
        model_type="llama",
        tokenizer_class="LlamaTokenizerFast",
        byte_level=True,
    )
    assert _is_sentencepiece_class_over_byte_level_bpe(tmp_path, "llama")

    tokenizer, overridden = _load(tmp_path)

    assert overridden
    assert not isinstance(tokenizer, LlamaTokenizer)
    ids = raw.encode(_SAMPLE, add_special_tokens=False).ids
    assert tokenizer.encode(_SAMPLE, add_special_tokens=False) == ids
    assert tokenizer.decode(ids) == raw.decode(ids) == _SAMPLE


def test_sentencepiece_llama_is_not_overridden(tmp_path: Path):
    _write_checkpoint(
        tmp_path,
        model_type="llama",
        tokenizer_class="LlamaTokenizerFast",
        byte_level=False,
    )
    assert not _is_sentencepiece_class_over_byte_level_bpe(tmp_path, "llama")

    tokenizer, overridden = _load(tmp_path)

    assert not overridden
    assert isinstance(tokenizer, LlamaTokenizer)


def test_qwen2_with_llama_class_is_left_to_transformers(tmp_path: Path):
    # DeepSeek-R1-Distill-Qwen-* declare LlamaTokenizerFast; transformers
    # already remaps qwen2, so vLLM must not change that path.
    _write_checkpoint(
        tmp_path,
        model_type="qwen2",
        tokenizer_class="LlamaTokenizerFast",
        byte_level=True,
    )
    assert not _is_sentencepiece_class_over_byte_level_bpe(tmp_path, "qwen2")

    tokenizer, overridden = _load(tmp_path)
    reference = AutoTokenizer.from_pretrained(tmp_path)

    assert not overridden
    assert type(tokenizer).__name__ == f"Cached{type(reference).__name__}"
    assert tokenizer.encode(_SAMPLE, add_special_tokens=False) == reference.encode(
        _SAMPLE, add_special_tokens=False
    )


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ({"type": "ByteLevel"}, True),
        (
            {
                "type": "Sequence",
                "pretokenizers": [{"type": "Split"}, {"type": "ByteLevel"}],
            },
            True,
        ),
        ({"type": "Metaspace", "replacement": "\u2581"}, False),
        (None, False),
    ],
)
def test_has_byte_level(spec, expected):
    assert _has_byte_level(spec) is expected
