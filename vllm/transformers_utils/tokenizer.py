# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import warnings
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import huggingface_hub
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing_extensions import assert_never

from vllm import envs
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_sentence_transformer_tokenizer_config
from vllm.transformers_utils.tokenizers import MistralTokenizer
from vllm.transformers_utils.utils import check_gguf_file

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.transformers_utils.tokenizer_base import TokenizerBase
else:
    ModelConfig = Any
    TokenizerBase = Any
import vllm.global_var as gv 

logger = init_logger(__name__)

AnyTokenizer: TypeAlias = PreTrainedTokenizer | PreTrainedTokenizerFast | TokenizerBase


def __getattr__(name: str):
    # Keep until lm-eval is updated
    if name == "get_tokenizer":
        from vllm.tokenizers import get_tokenizer

        warnings.warn(
            "`vllm.transformers_utils.tokenizer.get_tokenizer` "
            "has been moved to `vllm.tokenizers.get_tokenizer`. "
            "The old name will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )

        return get_tokenizer

        tokenizer = TokenizerRegistry.get_tokenizer(
            str(tokenizer_name),
            *args,
            revision=revision,
            download_dir=download_dir,
            **kwargs,
        )
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                *args,
                trust_remote_code=trust_remote_code,
                revision=revision,
                **kwargs,
            )
        except ValueError as e:
            # If the error pertains to the tokenizer class not existing or not
            # currently being imported,
            # suggest using the --trust-remote-code flag.
            if not trust_remote_code and (
                "does not exist or is not currently imported." in str(e)
                or "requires you to execute the tokenizer file" in str(e)
            ):
                err_msg = (
                    "Failed to load the tokenizer. If the tokenizer "
                    "is a custom tokenizer not yet available in the "
                    "HuggingFace transformers library, consider "
                    "setting `trust_remote_code=True` in LLM or using "
                    "the `--trust-remote-code` flag in the CLI."
                )
                raise RuntimeError(err_msg) from e
            else:
                raise e

        # The special_tokens in tokenizer should also be
        # controlled by do_lower_case in encoder_config
        encoder_config = get_sentence_transformer_tokenizer_config(
            tokenizer_name, revision
        )
        if isinstance(encoder_config, dict) and encoder_config.get(
            "do_lower_case", False
        ):
            special_tokens_map = {
                k: v.lower() for k, v in tokenizer.special_tokens_map.items()
            }
            tokenizer.add_special_tokens(special_tokens_map)

        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            logger.warning(
                "Using a slow tokenizer. This might cause a significant "
                "slowdown. Consider using a fast tokenizer instead."
            )
        tokenizer = get_cached_tokenizer(tokenizer)

    return tokenizer


cached_get_tokenizer = lru_cache(get_tokenizer)

def cached_tokenizer_from_config(
    model_config: ModelConfig,
    **kwargs: Any,
):
    return cached_get_tokenizer(
        model_config.tokenizer,
        tokenizer_mode=model_config.tokenizer_mode,
        revision=model_config.tokenizer_revision,
        trust_remote_code=model_config.trust_remote_code,
        **kwargs,
    )


def init_tokenizer_from_configs(model_config: ModelConfig):
    if model_config.tokenizer in gv.TOKENIZER_MAP:
        return gv.TOKENIZER_MAP[model_config.tokenizer]
    runner_type = model_config.runner_type
    if runner_type == "generate" or runner_type == "draft":
        truncation_side = "left"
    elif runner_type == "pooling":
        truncation_side = "right"
    else:
        assert_never(runner_type)

    return get_tokenizer(
        model_config.tokenizer,
        tokenizer_mode=model_config.tokenizer_mode,
        trust_remote_code=model_config.trust_remote_code,
        revision=model_config.tokenizer_revision,
        truncation_side=truncation_side,
    )
