from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Literal

from transformers import PreTrainedTokenizerBase

from llm_perf_lab.baseline.batching import (
    StaticBatchResult,
    create_batch_inputs_from_prompts,
    generate_static_batch,
)
from llm_perf_lab.baseline.runner import (
    load_model,
    load_tokenizer,
)
from llm_perf_lab.config import (
    GenerationConfig,
    ModelConfig,
    resolve_dtype,
)
from llm_perf_lab.monitoring import (
    get_gpu_info,
    get_gpu_memory_stats,
    reset_gpu_memory_stats,
    synchronize_cuda,
)

BatchingStrategy = Literal[
    "mixed",
    "length_bucketed",
]


@dataclass(frozen=True)
class StrategyRun:
    strategy: BatchingStrategy
    request_count: int
    micro_batch_size: int
    micro_batch_count: int
    prompt_tokens_total: int
    padded_prompt_tokens_total: int
    padding_tokens_total: int
    padding_ratio: float
    generated_tokens_total: int
    latency_seconds: float
    requests_per_second: float
    output_tokens_per_second: float
    peak_allocated_mb: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Compare mixed-order and length-bucketed static batching."),
    )

    parser.add_argument(
        "--model",
        required=True,
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--dtype",
        choices=(
            "float16",
            "bfloat16",
            "float32",
        ),
        default="float16",
    )
    parser.add_argument(
        "--request-count",
        type=int,
        default=256,
        help="Total requests processed per strategy.",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=16,
        help="Requests in each model.generate() call.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def validate_arguments(
    arguments: argparse.Namespace,
) -> None:
    if arguments.request_count <= 0:
        raise ValueError("--request-count must be greater than zero.")

    if arguments.micro_batch_size <= 0:
        raise ValueError("--micro-batch-size must be greater than zero.")

    if arguments.request_count % arguments.micro_batch_size != 0:
        raise ValueError("--request-count must be divisible by --micro-batch-size.")

    if arguments.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be greater than zero.")

    if arguments.warmup_runs < 0:
        raise ValueError("--warmup-runs cannot be negative.")

    if arguments.repetitions <= 0:
        raise ValueError("--repetitions must be greater than zero.")

    if arguments.do_sample and arguments.temperature <= 0:
        raise ValueError("--temperature must be greater than zero when sampling.")

    if not arguments.do_sample and arguments.temperature != 0:
        raise ValueError("--temperature must be 0 unless sampling is enabled.")


def load_prompts(
    path: Path,
) -> tuple[str, ...]:
    content = json.loads(
        path.read_text(encoding="utf-8"),
    )

    if not isinstance(content, list):
        raise ValueError("The prompts file must contain a JSON list.")

    if not all(isinstance(prompt, str) and prompt.strip() for prompt in content):
        raise ValueError("Every prompt must be a non-empty string.")

    if not content:
        raise ValueError("The prompts file cannot be empty.")

    return tuple(content)


def repeat_prompts(
    prompts: tuple[str, ...],
    request_count: int,
) -> tuple[str, ...]:
    return tuple(prompts[index % len(prompts)] for index in range(request_count))


def get_prompt_length(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
) -> int:
    encoded = tokenizer(
        prompt,
        add_special_tokens=True,
    )

    input_ids = encoded["input_ids"]

    return len(input_ids)


def order_prompts(
    *,
    tokenizer: PreTrainedTokenizerBase,
    prompts: tuple[str, ...],
    strategy: BatchingStrategy,
) -> tuple[str, ...]:
    if strategy == "mixed":
        return prompts

    return tuple(
        sorted(
            prompts,
            key=lambda prompt: get_prompt_length(
                tokenizer,
                prompt,
            ),
        )
    )


def create_micro_batches(
    prompts: Sequence[str],
    micro_batch_size: int,
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(prompts[start : start + micro_batch_size])
        for start in range(
            0,
            len(prompts),
            micro_batch_size,
        )
    )


def run_micro_batch(
    *,
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    prompts: tuple[str, ...],
    device: str,
    generation_config: GenerationConfig,
) -> StaticBatchResult:
    inputs = create_batch_inputs_from_prompts(
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
    )

    return generate_static_batch(
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        generation_config=generation_config,
    )


def run_strategy_once(
    *,
    strategy: BatchingStrategy,
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    requests: tuple[str, ...],
    micro_batch_size: int,
    device: str,
    generation_config: GenerationConfig,
) -> StrategyRun:
    ordered_prompts = order_prompts(
        tokenizer=tokenizer,
        prompts=requests,
        strategy=strategy,
    )

    micro_batches = create_micro_batches(
        ordered_prompts,
        micro_batch_size,
    )

    reset_gpu_memory_stats(device)

    synchronize_cuda()
    start_time = time.perf_counter()

    results = [
        run_micro_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=micro_batch,
            device=device,
            generation_config=generation_config,
        )
        for micro_batch in micro_batches
    ]

    synchronize_cuda()
    latency_seconds = time.perf_counter() - start_time

    memory = get_gpu_memory_stats(device)

    prompt_tokens_total = sum(result.metrics.prompt_tokens_total for result in results)
    padded_prompt_tokens_total = sum(
        result.metrics.padded_prompt_tokens_total for result in results
    )
    padding_tokens_total = sum(result.metrics.padding_tokens_total for result in results)
    generated_tokens_total = sum(result.metrics.generated_tokens_total for result in results)

    padding_ratio = (
        padding_tokens_total / padded_prompt_tokens_total if padded_prompt_tokens_total > 0 else 0.0
    )

    request_count = len(requests)

    return StrategyRun(
        strategy=strategy,
        request_count=request_count,
        micro_batch_size=micro_batch_size,
        micro_batch_count=len(micro_batches),
        prompt_tokens_total=prompt_tokens_total,
        padded_prompt_tokens_total=(padded_prompt_tokens_total),
        padding_tokens_total=padding_tokens_total,
        padding_ratio=padding_ratio,
        generated_tokens_total=generated_tokens_total,
        latency_seconds=latency_seconds,
        requests_per_second=(request_count / latency_seconds),
        output_tokens_per_second=(generated_tokens_total / latency_seconds),
        peak_allocated_mb=memory.peak_allocated_mb,
    )


def run_warmup(
    *,
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    requests: tuple[str, ...],
    micro_batch_size: int,
    device: str,
    generation_config: GenerationConfig,
    warmup_runs: int,
) -> None:
    representative_batch = create_micro_batches(
        requests,
        micro_batch_size,
    )[0]

    for _ in range(warmup_runs):
        run_micro_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=representative_batch,
            device=device,
            generation_config=generation_config,
        )


def summarize_runs(
    runs: list[StrategyRun],
) -> dict[str, float | int]:
    return {
        "count": len(runs),
        "mean_latency_seconds": mean(run.latency_seconds for run in runs),
        "mean_requests_per_second": mean(run.requests_per_second for run in runs),
        "mean_output_tokens_per_second": mean(run.output_tokens_per_second for run in runs),
        "mean_padding_ratio": mean(run.padding_ratio for run in runs),
        "mean_peak_allocated_mb": mean(run.peak_allocated_mb for run in runs),
    }


def main() -> None:
    arguments = parse_arguments()
    validate_arguments(arguments)

    model_config = ModelConfig(
        model_name=arguments.model,
        device=arguments.device,
        dtype=resolve_dtype(arguments.dtype),
        trust_remote_code=(arguments.trust_remote_code),
    )

    generation_config = GenerationConfig(
        max_new_tokens=arguments.max_new_tokens,
        temperature=arguments.temperature,
        do_sample=arguments.do_sample,
    )

    prompts = load_prompts(
        arguments.prompts_file,
    )
    requests = repeat_prompts(
        prompts,
        arguments.request_count,
    )

    print(f"Loading model: {model_config.model_name}")
    print(f"Requests: {arguments.request_count}")
    print(f"Micro-batch size: {arguments.micro_batch_size}")

    tokenizer = load_tokenizer(model_config)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token.")

        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model = load_model(model_config)

    run_warmup(
        model=model,
        tokenizer=tokenizer,
        requests=requests,
        micro_batch_size=(arguments.micro_batch_size),
        device=model_config.device,
        generation_config=generation_config,
        warmup_runs=arguments.warmup_runs,
    )

    strategy_runs: dict[
        BatchingStrategy,
        list[StrategyRun],
    ] = {
        "mixed": [],
        "length_bucketed": [],
    }

    for repetition in range(arguments.repetitions):
        strategy_order: tuple[
            BatchingStrategy,
            BatchingStrategy,
        ]

        if repetition % 2 == 0:
            strategy_order = (
                "mixed",
                "length_bucketed",
            )
        else:
            strategy_order = (
                "length_bucketed",
                "mixed",
            )

        for strategy in strategy_order:
            print(f"Repetition {repetition + 1}: {strategy}")

            result = run_strategy_once(
                strategy=strategy,
                model=model,
                tokenizer=tokenizer,
                requests=requests,
                micro_batch_size=(arguments.micro_batch_size),
                device=model_config.device,
                generation_config=(generation_config),
            )

            strategy_runs[strategy].append(result)

    output: dict[str, Any] = {
        "model": {
            "name": model_config.model_name,
            "device": model_config.device,
            "dtype": str(model_config.dtype).removeprefix("torch."),
        },
        "generation": asdict(generation_config),
        "hardware": asdict(get_gpu_info()),
        "request_count": (arguments.request_count),
        "micro_batch_size": (arguments.micro_batch_size),
        "repetitions": arguments.repetitions,
        "strategies": {
            strategy: {
                "summary": summarize_runs(runs),
                "runs": [run.to_dict() for run in runs],
            }
            for strategy, runs in strategy_runs.items()
        },
    }

    arguments.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    arguments.output.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("Length-bucketing comparison")
    print("---------------------------")

    for strategy, runs in strategy_runs.items():
        summary = summarize_runs(runs)

        print(
            f"{strategy}: "
            f"{summary['mean_latency_seconds']:.4f} s, "
            f"{summary['mean_requests_per_second']:.2f} req/s, "
            f"{summary['mean_output_tokens_per_second']:.2f} tok/s, "
            f"{summary['mean_padding_ratio']:.1%} padding, "
            f"{summary['mean_peak_allocated_mb']:.2f} MB peak"
        )

    print(f"Results written to: {arguments.output}")


if __name__ == "__main__":
    main()
