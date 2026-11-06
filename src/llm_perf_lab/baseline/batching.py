from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from llm_perf_lab.config import GenerationConfig
from llm_perf_lab.monitoring import synchronize_cuda
from llm_perf_lab.types import CausalLMProtocol


@dataclass(frozen=True)
class BatchMetrics:
    batch_size: int
    prompt_tokens_total: int
    padded_prompt_tokens_total: int
    padding_tokens_total: int
    padding_ratio: float
    generated_tokens_total: int
    batch_latency_seconds: float
    requests_per_second: float
    output_tokens_per_second: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StaticBatchResult:
    generated_texts: tuple[str, ...]
    metrics: BatchMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_texts": list(self.generated_texts),
            "metrics": self.metrics.to_dict(),
        }


def create_batch_inputs_from_prompts(
    *,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    device: str,
) -> dict[str, torch.Tensor]:
    if not prompts:
        raise ValueError("At least one prompt is required.")

    if isinstance(prompts, str):
        raise TypeError("prompts must be a sequence of strings, not a single string.")

    if any(not prompt.strip() for prompt in prompts):
        raise ValueError("Prompts cannot be empty.")

    encoded = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
    )

    return {
        key: value.to(device) for key, value in encoded.items() if isinstance(value, torch.Tensor)
    }


def create_batch_inputs(
    *,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    batch_size: int,
    device: str,
) -> dict[str, torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    return create_batch_inputs_from_prompts(
        tokenizer=tokenizer,
        prompts=(prompt,) * batch_size,
        device=device,
    )


def generate_static_batch(
    *,
    model: CausalLMProtocol,
    tokenizer: PreTrainedTokenizerBase,
    inputs: dict[str, torch.Tensor],
    generation_config: GenerationConfig,
) -> StaticBatchResult:
    input_ids = inputs["input_ids"]

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch_size, sequence_length].")

    batch_size = int(input_ids.shape[0])
    padded_input_length = int(input_ids.shape[1])

    attention_mask = inputs.get("attention_mask")

    padded_prompt_tokens_total = batch_size * padded_input_length

    if attention_mask is None:
        prompt_tokens_total = padded_prompt_tokens_total
    else:
        prompt_tokens_total = int(attention_mask.sum().item())

    padding_tokens_total = padded_prompt_tokens_total - prompt_tokens_total

    padding_ratio = (
        padding_tokens_total / padded_prompt_tokens_total if padded_prompt_tokens_total > 0 else 0.0
    )

    generation_kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": generation_config.max_new_tokens,
        "do_sample": generation_config.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }

    if generation_config.do_sample:
        generation_kwargs["temperature"] = generation_config.temperature

    synchronize_cuda()
    start_time = perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(**generation_kwargs)

    synchronize_cuda()
    batch_latency_seconds = perf_counter() - start_time

    generated_ids = output_ids[:, padded_input_length:]

    generated_tokens_total = int(generated_ids.numel())

    decoded_texts = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    generated_texts = tuple(decoded_texts)

    requests_per_second = batch_size / batch_latency_seconds
    output_tokens_per_second = generated_tokens_total / batch_latency_seconds

    return StaticBatchResult(
        generated_texts=generated_texts,
        metrics=BatchMetrics(
            batch_size=batch_size,
            prompt_tokens_total=prompt_tokens_total,
            padded_prompt_tokens_total=(padded_prompt_tokens_total),
            padding_tokens_total=padding_tokens_total,
            padding_ratio=padding_ratio,
            generated_tokens_total=generated_tokens_total,
            batch_latency_seconds=batch_latency_seconds,
            requests_per_second=requests_per_second,
            output_tokens_per_second=(output_tokens_per_second),
        ),
    )
