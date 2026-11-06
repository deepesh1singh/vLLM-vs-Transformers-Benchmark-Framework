from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from llm_perf_lab.client.benchmark import (
    ConcurrentBenchmarkResult,
    combine_benchmark_results,
    run_concurrent_benchmark,
    summarize_repetition_stability,
    wait_for_server,
)
from llm_perf_lab.profiling.telemetry import (
    TelemetryCollector,
    TelemetryResult,
    summarize_telemetry_results,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Benchmark concurrent streaming requests against a vLLM server.")
    )

    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="local-vllm-key")
    parser.add_argument("--model", default="qwen25-3b")

    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompts-file", type=Path)

    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32],
    )
    parser.add_argument("--requests-per-level", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--warmup-concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)

    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--gpu-index", type=int, default=0)
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
        "--include-request-details",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    if not arguments.concurrency:
        raise ValueError("At least one concurrency level is required.")

    if any(level <= 0 for level in arguments.concurrency):
        raise ValueError("Concurrency values must be positive.")

    positive_values = {
        "--requests-per-level": arguments.requests_per_level,
        "--repetitions": arguments.repetitions,
        "--warmup-concurrency": arguments.warmup_concurrency,
        "--max-tokens": arguments.max_tokens,
        "--timeout-seconds": arguments.timeout_seconds,
        "--telemetry-interval-seconds": (arguments.telemetry_interval_seconds),
    }

    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")

    if arguments.requests_per_level < max(arguments.concurrency):
        raise ValueError("--requests-per-level must be at least the maximum concurrency.")

    if arguments.warmup_requests < 0:
        raise ValueError("--warmup-requests cannot be negative.")

    if arguments.temperature < 0:
        raise ValueError("--temperature cannot be negative.")

    if arguments.gpu_index < 0:
        raise ValueError("--gpu-index cannot be negative.")


def load_prompts(arguments: argparse.Namespace) -> tuple[str, ...]:
    if arguments.prompt is not None:
        prompt = arguments.prompt.strip()

        if not prompt:
            raise ValueError("--prompt cannot be empty.")

        return (prompt,)

    if arguments.prompts_file is None:
        raise ValueError("A prompt source is required.")

    data = json.loads(arguments.prompts_file.read_text(encoding="utf-8"))

    if not isinstance(data, list) or not data:
        raise ValueError("The prompts file must contain a non-empty JSON list.")

    if not all(isinstance(prompt, str) and prompt.strip() for prompt in data):
        raise ValueError("Every prompt must be a non-empty string.")

    return tuple(data)


def format_optional(
    value: float | None,
    *,
    decimals: int,
    suffix: str = "",
) -> str:
    if value is None:
        return "n/a"

    return f"{value:.{decimals}f}{suffix}"


def format_milliseconds(value: float | None) -> str:
    if value is None:
        return "n/a"

    return f"{value * 1000:.2f} ms"


async def run_batch(
    *,
    client: httpx.AsyncClient,
    arguments: argparse.Namespace,
    prompts: tuple[str, ...],
    request_count: int,
    concurrency: int,
) -> ConcurrentBenchmarkResult:
    return await run_concurrent_benchmark(
        client=client,
        base_url=arguments.base_url,
        api_key=arguments.api_key,
        model=arguments.model,
        prompts=prompts,
        max_tokens=arguments.max_tokens,
        temperature=arguments.temperature,
        request_count=request_count,
        concurrency=concurrency,
    )


async def run_repetition(
    *,
    client: httpx.AsyncClient,
    arguments: argparse.Namespace,
    prompts: tuple[str, ...],
    concurrency: int,
) -> tuple[ConcurrentBenchmarkResult, TelemetryResult | None]:
    collector: TelemetryCollector | None = None

    if arguments.profile:
        collector = TelemetryCollector(
            gpu_index=arguments.gpu_index,
            sample_interval_seconds=(arguments.telemetry_interval_seconds),
        )
        collector.start()

    try:
        result = await run_batch(
            client=client,
            arguments=arguments,
            prompts=prompts,
            request_count=arguments.requests_per_level,
            concurrency=concurrency,
        )
    finally:
        telemetry = collector.stop() if collector is not None else None

    return result, telemetry


def serialize_repetitions(
    *,
    results: list[ConcurrentBenchmarkResult],
    telemetry: list[TelemetryResult],
    arguments: argparse.Namespace,
) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []

    for index, result in enumerate(results):
        item = result.to_dict(include_request_details=(arguments.include_request_details))

        if index < len(telemetry):
            item["telemetry"] = telemetry[index].to_dict(
                include_samples=(arguments.include_telemetry_samples)
            )

        serialized.append(item)

    return serialized


async def run_level(
    *,
    client: httpx.AsyncClient,
    arguments: argparse.Namespace,
    prompts: tuple[str, ...],
    concurrency: int,
) -> dict[str, Any]:
    print()
    print(f"Concurrency {concurrency}")

    warmup_requests = max(16, concurrency * 2)
    print(f"  Level warm-up: {warmup_requests} requests")

    await run_batch(
        client=client,
        arguments=arguments,
        prompts=prompts,
        request_count=warmup_requests,
        concurrency=concurrency,
    )

    results: list[ConcurrentBenchmarkResult] = []
    telemetry_results: list[TelemetryResult] = []

    for repetition in range(1, arguments.repetitions + 1):
        print(f"  Repetition {repetition}/{arguments.repetitions}")

        result, telemetry = await run_repetition(
            client=client,
            arguments=arguments,
            prompts=prompts,
            concurrency=concurrency,
        )

        results.append(result)

        if telemetry is not None:
            telemetry_results.append(telemetry)

    aggregate = combine_benchmark_results(results)
    stability = summarize_repetition_stability(results)

    print(f"  Success: {aggregate.success_rate:.1%}")
    print(f"  Median throughput: {stability.median_requests_per_second:.2f} req/s")
    print(f"  Median output throughput: {stability.median_output_tokens_per_second:.2f} tok/s")
    print(
        "  Throughput range: "
        f"{stability.minimum_output_tokens_per_second:.2f}-"
        f"{stability.maximum_output_tokens_per_second:.2f} tok/s"
    )
    print(f"  Throughput CV: {stability.output_tokens_per_second_cv:.2%}")
    print(f"  Stability: {stability.status}")
    print(f"  Median latency p50: {format_milliseconds(stability.median_latency_p50_seconds)}")
    print(f"  Median TTFT p50: {format_milliseconds(stability.median_ttft_p50_seconds)}")

    level_data: dict[str, Any] = {
        "concurrency": concurrency,
        "level_warmup_requests": warmup_requests,
        "stability": stability.to_dict(),
        "pooled_requests": aggregate.to_dict(include_request_details=False),
        "repetitions": serialize_repetitions(
            results=results,
            telemetry=telemetry_results,
            arguments=arguments,
        ),
    }

    if telemetry_results:
        level_telemetry = summarize_telemetry_results(telemetry_results)

        print(
            "  GPU utilization: "
            + format_optional(
                level_telemetry.median_gpu_utilization_percent,
                decimals=1,
                suffix="%",
            )
        )
        print(
            "  GPU memory peak: "
            + format_optional(
                level_telemetry.maximum_gpu_memory_used_mb,
                decimals=0,
                suffix=" MB",
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
        print(
            "  GPU temperature max: "
            + format_optional(
                level_telemetry.maximum_temperature_celsius,
                decimals=0,
                suffix=" °C",
            )
        )

        level_data["telemetry"] = level_telemetry.to_dict()

    return level_data


async def run() -> None:
    arguments = parse_arguments()
    validate_arguments(arguments)

    prompts = load_prompts(arguments)
    maximum_concurrency = max(arguments.concurrency)

    timeout = httpx.Timeout(
        connect=10.0,
        read=arguments.timeout_seconds,
        write=30.0,
        pool=arguments.timeout_seconds,
    )
    limits = httpx.Limits(
        max_connections=maximum_concurrency,
        max_keepalive_connections=maximum_concurrency,
    )

    levels: list[dict[str, Any]] = []

    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        trust_env=False,
    ) as client:
        print(f"Checking server: {arguments.base_url}")

        await wait_for_server(
            client=client,
            base_url=arguments.base_url,
            api_key=arguments.api_key,
        )

        print("Server is ready.")

        if arguments.warmup_requests > 0:
            warmup_concurrency = min(
                arguments.warmup_concurrency,
                arguments.warmup_requests,
                maximum_concurrency,
            )

            print(
                f"Warm-up: {arguments.warmup_requests} requests, concurrency {warmup_concurrency}"
            )

            await run_batch(
                client=client,
                arguments=arguments,
                prompts=prompts,
                request_count=arguments.warmup_requests,
                concurrency=warmup_concurrency,
            )

        for concurrency in arguments.concurrency:
            levels.append(
                await run_level(
                    client=client,
                    arguments=arguments,
                    prompts=prompts,
                    concurrency=concurrency,
                )
            )

    output: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "server": {
            "base_url": arguments.base_url,
            "model": arguments.model,
        },
        "workload": {
            "prompt_count": len(prompts),
            "max_tokens": arguments.max_tokens,
            "temperature": arguments.temperature,
            "concurrency_levels": arguments.concurrency,
            "requests_per_level": arguments.requests_per_level,
            "repetitions": arguments.repetitions,
            "global_warmup_requests": arguments.warmup_requests,
            "level_warmup_policy": "max(16, concurrency * 2)",
            "profiling": {
                "enabled": arguments.profile,
                "gpu_index": arguments.gpu_index,
                "sample_interval_seconds": (arguments.telemetry_interval_seconds),
            },
        },
        "levels": levels,
    }

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print(f"Results written to: {arguments.output}")


if __name__ == "__main__":
    asyncio.run(run())
