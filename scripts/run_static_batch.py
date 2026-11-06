from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Any

import torch

from llm_perf_lab.baseline.batching import (
    StaticBatchResult,
    create_batch_inputs_from_prompts,
    generate_static_batch,
)
from llm_perf_lab.baseline.runner import load_model, load_tokenizer
from llm_perf_lab.config import GenerationConfig, ModelConfig, resolve_dtype
from llm_perf_lab.metrics import percentile
from llm_perf_lab.monitoring import (
    get_gpu_info,
    get_gpu_memory_stats,
    reset_gpu_memory_stats,
)
from llm_perf_lab.profiling.telemetry import (
    TelemetryCollector,
    TelemetryResult,
    summarize_telemetry_results,
)


@dataclass(frozen=True)
class SustainedRepetitionResult:
    repetition_index: int
    batch_count: int
    request_count: int
    prompt_tokens_total: int
    padded_prompt_tokens_total: int
    padding_tokens_total: int
    generated_tokens_total: int
    wall_time_seconds: float
    requests_per_second: float
    output_tokens_per_second: float
    mean_batch_latency_seconds: float
    median_batch_latency_seconds: float
    p95_batch_latency_seconds: float
    batch_metrics: tuple[dict[str, Any], ...]

    def to_dict(
        self,
        *,
        include_batch_details: bool,
    ) -> dict[str, Any]:
        data = asdict(self)
        batch_metrics = data.pop("batch_metrics")

        if include_batch_details:
            data["batch_metrics"] = batch_metrics

        return data


def parse_batch_sizes(value: str) -> tuple[int, ...]:
    try:
        batch_sizes = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Batch sizes must be comma-separated integers.") from error

    if not batch_sizes:
        raise argparse.ArgumentTypeError("At least one batch size is required.")

    if any(batch_size <= 0 for batch_size in batch_sizes):
        raise argparse.ArgumentTypeError("Batch sizes must be greater than zero.")

    return batch_sizes


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Benchmark sustained static batching for causal language models.")
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model name or local path.",
    )

    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument(
        "--prompt",
        help="Prompt repeated across the workload.",
    )
    prompts.add_argument(
        "--prompts-file",
        type=Path,
        help="JSON file containing a list of prompts.",
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--batch-sizes",
        type=parse_batch_sizes,
        default=(32, 64, 128),
        help="Comma-separated batch sizes.",
    )
    parser.add_argument(
        "--requests-per-repetition",
        type=int,
        default=384,
        help="Total requests processed in every repetition.",
    )
    parser.add_argument(
        "--warmup-repetitions",
        type=int,
        default=1,
        help=("Unmeasured full-workload repetitions per batch size."),
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
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
        "--profile",
        action="store_true",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--telemetry-interval-seconds",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--include-telemetry-samples",
        action="store_true",
    )
    parser.add_argument(
        "--include-batch-details",
        action="store_true",
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
    if arguments.requests_per_repetition <= 0:
        raise ValueError("--requests-per-repetition must be positive.")

    for batch_size in arguments.batch_sizes:
        if arguments.requests_per_repetition % batch_size != 0:
            raise ValueError("--requests-per-repetition must be divisible by every batch size.")

    if arguments.warmup_repetitions < 0:
        raise ValueError("--warmup-repetitions cannot be negative.")

    if arguments.repetitions <= 0:
        raise ValueError("--repetitions must be positive.")

    if arguments.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")

    if arguments.do_sample:
        if arguments.temperature <= 0:
            raise ValueError("--temperature must be positive with --do-sample.")
    elif arguments.temperature != 0:
        raise ValueError("--temperature must be 0 unless --do-sample is enabled.")

    if arguments.gpu_index < 0:
        raise ValueError("--gpu-index cannot be negative.")

    if arguments.telemetry_interval_seconds <= 0:
        raise ValueError("--telemetry-interval-seconds must be positive.")

    if arguments.profile and not arguments.device.startswith("cuda"):
        raise ValueError("--profile requires a CUDA device.")


def load_prompts(
    arguments: argparse.Namespace,
) -> tuple[str, ...]:
    if arguments.prompt is not None:
        prompt = arguments.prompt.strip()

        if not prompt:
            raise ValueError("--prompt cannot be empty.")

        return (prompt,)

    if arguments.prompts_file is None:
        raise ValueError("A prompt source is required.")

    content = json.loads(arguments.prompts_file.read_text(encoding="utf-8"))

    if not isinstance(content, list) or not content:
        raise ValueError("The prompts file must contain a non-empty JSON list.")

    if not all(isinstance(prompt, str) and prompt.strip() for prompt in content):
        raise ValueError("Every prompt must be a non-empty string.")

    return tuple(content)


def build_batch_prompts(
    prompts: tuple[str, ...],
    batch_size: int,
) -> tuple[str, ...]:
    return tuple(prompts[index % len(prompts)] for index in range(batch_size))


def coefficient_of_variation(
    values: list[float],
) -> float:
    average = mean(values)

    if average == 0:
        return 0.0

    return pstdev(values) / average


def classify_stability(
    coefficient: float,
) -> str:
    if coefficient <= 0.05:
        return "stable"

    if coefficient <= 0.10:
        return "noisy"

    return "unstable"


def format_optional(
    value: float | None,
    *,
    decimals: int,
    suffix: str = "",
) -> str:
    if value is None:
        return "n/a"

    return f"{value:.{decimals}f}{suffix}"


def run_single_batch(
    *,
    model: Any,
    tokenizer: Any,
    inputs: dict[str, torch.Tensor],
    generation_config: GenerationConfig,
) -> StaticBatchResult:
    return generate_static_batch(
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        generation_config=generation_config,
    )


def run_sustained_repetition(
    *,
    model: Any,
    tokenizer: Any,
    inputs: dict[str, torch.Tensor],
    generation_config: GenerationConfig,
    repetition_index: int,
    batches_per_repetition: int,
    profile: bool,
    gpu_index: int,
    telemetry_interval_seconds: float,
) -> tuple[
    SustainedRepetitionResult,
    TelemetryResult | None,
]:
    collector: TelemetryCollector | None = None

    if profile:
        collector = TelemetryCollector(
            gpu_index=gpu_index,
            sample_interval_seconds=(telemetry_interval_seconds),
        )
        collector.start()

    batches: list[StaticBatchResult] = []
    started_at = perf_counter()

    try:
        for _ in range(batches_per_repetition):
            batches.append(
                run_single_batch(
                    model=model,
                    tokenizer=tokenizer,
                    inputs=inputs,
                    generation_config=(generation_config),
                )
            )
    finally:
        wall_time_seconds = perf_counter() - started_at
        telemetry = collector.stop() if collector is not None else None

    batch_latencies = [batch.metrics.batch_latency_seconds for batch in batches]

    request_count = sum(batch.metrics.batch_size for batch in batches)
    prompt_tokens_total = sum(batch.metrics.prompt_tokens_total for batch in batches)
    padded_prompt_tokens_total = sum(batch.metrics.padded_prompt_tokens_total for batch in batches)
    padding_tokens_total = sum(batch.metrics.padding_tokens_total for batch in batches)
    generated_tokens_total = sum(batch.metrics.generated_tokens_total for batch in batches)

    result = SustainedRepetitionResult(
        repetition_index=repetition_index,
        batch_count=len(batches),
        request_count=request_count,
        prompt_tokens_total=prompt_tokens_total,
        padded_prompt_tokens_total=(padded_prompt_tokens_total),
        padding_tokens_total=padding_tokens_total,
        generated_tokens_total=(generated_tokens_total),
        wall_time_seconds=wall_time_seconds,
        requests_per_second=(request_count / wall_time_seconds),
        output_tokens_per_second=(generated_tokens_total / wall_time_seconds),
        mean_batch_latency_seconds=mean(batch_latencies),
        median_batch_latency_seconds=percentile(
            batch_latencies,
            50,
        ),
        p95_batch_latency_seconds=percentile(
            batch_latencies,
            95,
        ),
        batch_metrics=tuple(batch.to_dict()["metrics"] for batch in batches),
    )

    return result, telemetry


def summarize_repetitions(
    results: list[SustainedRepetitionResult],
) -> dict[str, Any]:
    request_throughputs = [result.requests_per_second for result in results]
    output_throughputs = [result.output_tokens_per_second for result in results]
    total_latencies = [result.wall_time_seconds for result in results]
    median_batch_latencies = [result.median_batch_latency_seconds for result in results]

    throughput_cv = coefficient_of_variation(output_throughputs)

    return {
        "repetition_count": len(results),
        "median_requests_per_second": percentile(
            request_throughputs,
            50,
        ),
        "minimum_requests_per_second": min(request_throughputs),
        "maximum_requests_per_second": max(request_throughputs),
        "median_output_tokens_per_second": percentile(
            output_throughputs,
            50,
        ),
        "minimum_output_tokens_per_second": min(output_throughputs),
        "maximum_output_tokens_per_second": max(output_throughputs),
        "output_tokens_per_second_cv": throughput_cv,
        "median_repetition_wall_time_seconds": (percentile(total_latencies, 50)),
        "median_batch_latency_seconds": percentile(
            median_batch_latencies,
            50,
        ),
        "status": classify_stability(throughput_cv),
        "valid": throughput_cv <= 0.10,
    }


def energy_per_1k_output_tokens(
    *,
    telemetry: TelemetryResult,
    generated_tokens: int,
) -> float | None:
    energy = telemetry.summary.estimated_gpu_energy_joules

    if energy is None or generated_tokens <= 0:
        return None

    return energy / generated_tokens * 1000.0


def run_batch_size(
    *,
    model: Any,
    tokenizer: Any,
    prompts: tuple[str, ...],
    batch_size: int,
    device: str,
    generation_config: GenerationConfig,
    requests_per_repetition: int,
    warmup_repetitions: int,
    repetitions: int,
    profile: bool,
    gpu_index: int,
    telemetry_interval_seconds: float,
    include_telemetry_samples: bool,
    include_batch_details: bool,
) -> dict[str, Any]:
    batches_per_repetition = requests_per_repetition // batch_size
    batch_prompts = build_batch_prompts(
        prompts,
        batch_size,
    )
    inputs = create_batch_inputs_from_prompts(
        tokenizer=tokenizer,
        prompts=batch_prompts,
        device=device,
    )

    print()
    print(f"Batch size {batch_size}")
    print(f"  Batches per repetition: {batches_per_repetition}")

    for warmup_index in range(
        1,
        warmup_repetitions + 1,
    ):
        print(f"  Warm-up {warmup_index}/{warmup_repetitions}")

        for _ in range(batches_per_repetition):
            run_single_batch(
                model=model,
                tokenizer=tokenizer,
                inputs=inputs,
                generation_config=(generation_config),
            )

    reset_gpu_memory_stats(device)

    repetition_results: list[SustainedRepetitionResult] = []
    telemetry_results: list[TelemetryResult] = []

    for repetition_index in range(
        1,
        repetitions + 1,
    ):
        print(f"  Repetition {repetition_index}/{repetitions}")

        result, telemetry = run_sustained_repetition(
            model=model,
            tokenizer=tokenizer,
            inputs=inputs,
            generation_config=(generation_config),
            repetition_index=(repetition_index),
            batches_per_repetition=(batches_per_repetition),
            profile=profile,
            gpu_index=gpu_index,
            telemetry_interval_seconds=(telemetry_interval_seconds),
        )

        repetition_results.append(result)

        if telemetry is not None:
            telemetry_results.append(telemetry)

    memory = get_gpu_memory_stats(device)
    summary = summarize_repetitions(repetition_results)

    print(f"  Median throughput: {summary['median_requests_per_second']:.2f} req/s")
    print(f"  Median output throughput: {summary['median_output_tokens_per_second']:.2f} tok/s")
    print(f"  Throughput CV: {summary['output_tokens_per_second_cv']:.2%}")
    print(f"  Stability: {summary['status']}")
    print(f"  Median batch latency: {summary['median_batch_latency_seconds']:.4f} s")
    print(f"  Peak allocated memory: {memory.peak_allocated_mb:.2f} MB")

    serialized_repetitions: list[dict[str, Any]] = []

    for index, result in enumerate(repetition_results):
        item = result.to_dict(include_batch_details=(include_batch_details))

        if index < len(telemetry_results):
            telemetry = telemetry_results[index]
            item["telemetry"] = telemetry.to_dict(include_samples=(include_telemetry_samples))
            item["energy_joules_per_1k_output_tokens"] = energy_per_1k_output_tokens(
                telemetry=telemetry,
                generated_tokens=(result.generated_tokens_total),
            )

        serialized_repetitions.append(item)

    level_data: dict[str, Any] = {
        "batch_size": batch_size,
        "batches_per_repetition": (batches_per_repetition),
        "requests_per_repetition": (requests_per_repetition),
        "summary": summary,
        "memory": asdict(memory),
        "repetitions": serialized_repetitions,
    }

    if telemetry_results:
        level_telemetry = summarize_telemetry_results(telemetry_results)
        energy_values = [
            energy
            for result, telemetry in zip(
                repetition_results,
                telemetry_results,
                strict=True,
            )
            if (
                energy := energy_per_1k_output_tokens(
                    telemetry=telemetry,
                    generated_tokens=(result.generated_tokens_total),
                )
            )
            is not None
        ]

        level_data["telemetry"] = level_telemetry.to_dict()
        level_data["median_energy_joules_per_1k_output_tokens"] = (
            percentile(energy_values, 50) if energy_values else None
        )

        print(
            "  GPU utilization: "
            + format_optional(
                level_telemetry.median_gpu_utilization_percent,
                decimals=1,
                suffix="%",
            )
        )
        print(
            "  GPU power: "
            + format_optional(
                level_telemetry.median_power_watts,
                decimals=1,
                suffix=" W",
            )
        )

        if energy_values:
            print(f"  Energy: {percentile(energy_values, 50):.2f} J/1k output tokens")

    return level_data


def main() -> None:
    arguments = parse_arguments()
    validate_arguments(arguments)
    prompts = load_prompts(arguments)

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

    print(f"Loading model: {model_config.model_name}")
    print(f"Device: {model_config.device}")
    print(f"Data type: {model_config.dtype}")
    print(f"Batch sizes: {arguments.batch_sizes}")
    print(f"Requests per repetition: {arguments.requests_per_repetition}")

    tokenizer = load_tokenizer(model_config)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token.")

        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    model = load_model(model_config)

    results: list[dict[str, Any]] = []

    for batch_size in arguments.batch_sizes:
        try:
            result = run_batch_size(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                batch_size=batch_size,
                device=model_config.device,
                generation_config=(generation_config),
                requests_per_repetition=(arguments.requests_per_repetition),
                warmup_repetitions=(arguments.warmup_repetitions),
                repetitions=arguments.repetitions,
                profile=arguments.profile,
                gpu_index=arguments.gpu_index,
                telemetry_interval_seconds=(arguments.telemetry_interval_seconds),
                include_telemetry_samples=(arguments.include_telemetry_samples),
                include_batch_details=(arguments.include_batch_details),
            )
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()

            print(f"Batch size {batch_size}: CUDA out of memory")

            results.append(
                {
                    "batch_size": batch_size,
                    "status": "out_of_memory",
                    "error": str(error),
                }
            )
            break

        result["status"] = "completed"
        results.append(result)

    output: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "backend": "transformers",
        "model": {
            "name": model_config.model_name,
            "device": model_config.device,
            "dtype": str(model_config.dtype).removeprefix("torch."),
            "trust_remote_code": (model_config.trust_remote_code),
        },
        "generation": asdict(generation_config),
        "hardware": asdict(get_gpu_info()),
        "workload": {
            "prompt_count": len(prompts),
            "batch_sizes": list(arguments.batch_sizes),
            "requests_per_repetition": (arguments.requests_per_repetition),
            "warmup_repetitions": (arguments.warmup_repetitions),
            "repetitions": (arguments.repetitions),
            "profiling": {
                "enabled": arguments.profile,
                "gpu_index": (arguments.gpu_index),
                "sample_interval_seconds": (arguments.telemetry_interval_seconds),
            },
        },
        "results": results,
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
    print(f"Results written to: {arguments.output}")


if __name__ == "__main__":
    main()
