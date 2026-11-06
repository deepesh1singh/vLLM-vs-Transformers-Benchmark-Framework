from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from llm_perf_lab.baseline.token_streamer import generate_streaming
from llm_perf_lab.config import GenerationConfig, ModelConfig
from llm_perf_lab.metrics import (
    InferenceMetrics,
    LatencySummary,
    calculate_token_streaming_metrics,
    create_inference_metrics,
    summarize_latencies,
)
from llm_perf_lab.monitoring import (
    GPUInfo,
    GPUMemoryStats,
    get_gpu_info,
    get_gpu_memory_stats,
    reset_gpu_memory_stats,
    synchronize_cuda,
)
from llm_perf_lab.types import CausalLMProtocol


@dataclass(slots=True)
class InferenceRun:
    """Result from one measured inference execution."""

    run_index: int
    generated_text: str
    metrics: InferenceMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_index": self.run_index,
            "generated_text": self.generated_text,
            "metrics": self.metrics.to_dict(),
        }


@dataclass(slots=True)
class BaselineResult:
    """Complete result from a baseline benchmark."""

    model: dict[str, Any]
    generation: dict[str, Any]
    hardware: GPUInfo
    memory: GPUMemoryStats
    streaming: bool
    latency_summary: LatencySummary
    runs: list[InferenceRun]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "generation": self.generation,
            "hardware": self.hardware.to_dict(),
            "memory": self.memory.to_dict(),
            "streaming": self.streaming,
            "latency_summary": self.latency_summary.to_dict(),
            "runs": [run.to_dict() for run in self.runs],
        }


def validate_environment(model_config: ModelConfig) -> None:
    """Validate that the requested execution device is available."""

    if model_config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot access an NVIDIA GPU. "
            "Check the NVIDIA driver, CUDA-compatible PyTorch installation "
            "and the output of torch.cuda.is_available()."
        )


def load_tokenizer(
    model_config: ModelConfig,
) -> PreTrainedTokenizerBase:
    """Load the tokenizer associated with the selected model."""

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name,
        trust_remote_code=model_config.trust_remote_code,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model(
    model_config: ModelConfig,
) -> CausalLMProtocol:
    """Load a causal language model and move it to the target device."""

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        dtype=model_config.dtype,
        trust_remote_code=model_config.trust_remote_code,
        low_cpu_mem_usage=True,
    )

    model = cast(Any, model)
    model.to(model_config.device)
    model.eval()

    return cast(CausalLMProtocol, model)


def tokenize_prompt(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    device: str,
) -> dict[str, torch.Tensor]:
    """
    Tokenize a prompt using the model's chat template when available.

    Falls back to plain tokenization for models without a chat template.
    """

    if getattr(tokenizer, "chat_template", None):
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    else:
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
        )

    if not isinstance(encoded, BatchEncoding):
        raise TypeError("Tokenization did not return a BatchEncoding")

    if not isinstance(encoded, BatchEncoding):
        raise TypeError("apply_chat_template() did not return a BatchEncoding")

    return {key: value.to(device) for key, value in encoded.items()}


def generate_once(
    model: CausalLMProtocol,
    tokenizer: PreTrainedTokenizerBase,
    model_inputs: dict[str, torch.Tensor],
    generation_config: GenerationConfig,
    run_index: int,
) -> InferenceRun:
    """Execute and measure one model inference."""

    prompt_tokens = int(model_inputs["input_ids"].shape[-1])

    generation_kwargs: dict[str, Any] = {
        **model_inputs,
        "max_new_tokens": generation_config.max_new_tokens,
        "do_sample": generation_config.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }

    if generation_config.do_sample:
        generation_kwargs["temperature"] = generation_config.temperature

    synchronize_cuda()
    start_time = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(**generation_kwargs)

    synchronize_cuda()
    latency_seconds = time.perf_counter() - start_time

    generated_ids = output_ids[0, prompt_tokens:]
    generated_tokens = int(generated_ids.shape[-1])

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    if not isinstance(generated_text, str):
        raise TypeError("Tokenizer decode() did not return a string")

    metrics = create_inference_metrics(
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        latency_seconds=latency_seconds,
    )

    return InferenceRun(
        run_index=run_index,
        generated_text=generated_text,
        metrics=metrics,
    )


def generate_streaming_once(
    *,
    model: CausalLMProtocol,
    tokenizer: PreTrainedTokenizerBase,
    inputs: dict[str, torch.Tensor],
    generation_config: GenerationConfig,
    device: str,
    run_index: int,
) -> InferenceRun:
    """
    Execute one token-level streamed generation and collect latency metrics.
    """

    streaming_result = generate_streaming(
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        device=device,
        max_new_tokens=generation_config.max_new_tokens,
        do_sample=generation_config.do_sample,
        temperature=generation_config.temperature,
    )

    prompt_tokens = int(inputs["input_ids"].shape[-1])

    streaming_metrics = calculate_token_streaming_metrics(
        token_events=streaming_result.token_events,
        inter_token_latencies_seconds=(streaming_result.inter_token_latencies_seconds),
        decode_after_first_token_seconds=(streaming_result.decode_after_first_token_seconds),
    )

    metrics = create_inference_metrics(
        prompt_tokens=prompt_tokens,
        generated_tokens=streaming_metrics.token_event_count,
        latency_seconds=streaming_result.total_latency_seconds,
        time_to_first_token_seconds=(streaming_result.time_to_first_token_seconds),
        decode_after_first_token_seconds=(streaming_result.decode_after_first_token_seconds),
        mean_inter_token_latency_seconds=(streaming_metrics.mean_inter_token_latency_seconds),
        p50_inter_token_latency_seconds=(streaming_metrics.p50_inter_token_latency_seconds),
        p95_inter_token_latency_seconds=(streaming_metrics.p95_inter_token_latency_seconds),
        token_event_count=streaming_metrics.token_event_count,
        decode_tokens_per_second=(streaming_metrics.decode_tokens_per_second),
    )

    return InferenceRun(
        run_index=run_index,
        generated_text=streaming_result.generated_text,
        metrics=metrics,
    )


def run_warmup(
    model: CausalLMProtocol,
    tokenizer: PreTrainedTokenizerBase,
    model_inputs: dict[str, torch.Tensor],
    generation_config: GenerationConfig,
    warmup_runs: int,
) -> None:
    """Run unmeasured inference iterations to warm up the model and GPU."""

    if warmup_runs < 0:
        raise ValueError("warmup_runs cannot be negative.")

    for _ in range(warmup_runs):
        with torch.inference_mode():
            warmup_kwargs: dict[str, Any] = {
                **model_inputs,
                "max_new_tokens": generation_config.max_new_tokens,
                "do_sample": generation_config.do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "use_cache": True,
            }

            if generation_config.do_sample:
                warmup_kwargs["temperature"] = generation_config.temperature

            model.generate(**warmup_kwargs)
        synchronize_cuda()


def run_baseline(
    prompt: str,
    model_config: ModelConfig,
    generation_config: GenerationConfig,
    warmup_runs: int = 1,
    repetitions: int = 3,
    streaming: bool = False,
) -> BaselineResult:
    """Load the model and execute the complete baseline benchmark."""

    if repetitions <= 0:
        raise ValueError("repetitions must be greater than zero.")

    validate_environment(model_config)

    tokenizer = load_tokenizer(model_config)
    model = load_model(model_config)

    model_inputs = tokenize_prompt(
        tokenizer=tokenizer,
        prompt=prompt,
        device=model_config.device,
    )

    run_warmup(
        model=model,
        tokenizer=tokenizer,
        model_inputs=model_inputs,
        generation_config=generation_config,
        warmup_runs=warmup_runs,
    )

    reset_gpu_memory_stats()

    runs: list[InferenceRun] = []

    for i in range(repetitions):
        reset_gpu_memory_stats(model_config.device)

        if streaming:
            run = generate_streaming_once(
                model=model,
                tokenizer=tokenizer,
                inputs=model_inputs,
                generation_config=generation_config,
                device=model_config.device,
                run_index=i + 1,
            )
        else:
            run = generate_once(
                model=model,
                tokenizer=tokenizer,
                model_inputs=model_inputs,
                generation_config=generation_config,
                run_index=i + 1,
            )

        runs.append(run)

    memory_stats = get_gpu_memory_stats()
    gpu_info = get_gpu_info()

    latency_summary = summarize_latencies([run.metrics.latency_seconds for run in runs])

    return BaselineResult(
        model={
            "name": model_config.model_name,
            "device": model_config.device,
            "dtype": str(model_config.dtype).removeprefix("torch."),
            "trust_remote_code": model_config.trust_remote_code,
        },
        streaming=streaming,
        generation=asdict(generation_config),
        hardware=gpu_info,
        memory=memory_stats,
        latency_summary=latency_summary,
        runs=runs,
    )


def save_result(
    result: BaselineResult,
    output_path: str | Path,
) -> Path:
    """Save a baseline result as formatted JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            result.to_dict(),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return path
