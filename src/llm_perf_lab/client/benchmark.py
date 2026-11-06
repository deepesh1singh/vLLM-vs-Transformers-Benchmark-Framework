from __future__ import annotations

import asyncio
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import ceil, floor
from statistics import mean, median, pstdev
from typing import Any, Literal

import httpx

from llm_perf_lab.client.async_client import generate_streaming_request
from llm_perf_lab.client.models import AsyncRequestResult

type StabilityStatus = Literal[
    "stable",
    "noisy",
    "unstable",
    "insufficient_data",
]


@dataclass(frozen=True)
class DistributionSummary:
    count: int
    mean: float | None
    minimum: float | None
    p50: float | None
    p95: float | None
    p99: float | None
    maximum: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RepetitionStabilitySummary:
    repetition_count: int

    median_requests_per_second: float
    minimum_requests_per_second: float
    maximum_requests_per_second: float
    requests_per_second_cv: float

    median_output_tokens_per_second: float
    minimum_output_tokens_per_second: float
    maximum_output_tokens_per_second: float
    output_tokens_per_second_cv: float

    median_latency_p50_seconds: float | None
    median_ttft_p50_seconds: float | None

    status: StabilityStatus
    valid: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConcurrentBenchmarkResult:
    concurrency: int
    request_count: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    wall_time_seconds: float
    requests_per_second: float
    output_tokens_per_second: float
    prompt_tokens_total: int
    completion_tokens_total: int
    latency_seconds: DistributionSummary
    time_to_first_token_seconds: DistributionSummary
    decode_seconds: DistributionSummary
    post_stream_overhead_seconds: DistributionSummary
    inter_event_latency_seconds: DistributionSummary
    decode_tokens_per_second: DistributionSummary
    error_counts: dict[str, int]
    request_results: tuple[AsyncRequestResult, ...]

    def to_dict(
        self,
        *,
        include_request_details: bool = False,
    ) -> dict[str, Any]:
        output: dict[str, Any] = {
            "concurrency": self.concurrency,
            "request_count": self.request_count,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.success_rate,
            "wall_time_seconds": self.wall_time_seconds,
            "requests_per_second": self.requests_per_second,
            "output_tokens_per_second": (self.output_tokens_per_second),
            "prompt_tokens_total": self.prompt_tokens_total,
            "completion_tokens_total": (self.completion_tokens_total),
            "latency_seconds": (self.latency_seconds.to_dict()),
            "time_to_first_token_seconds": (self.time_to_first_token_seconds.to_dict()),
            "decode_seconds": self.decode_seconds.to_dict(),
            "post_stream_overhead_seconds": (self.post_stream_overhead_seconds.to_dict()),
            "inter_event_latency_seconds": (self.inter_event_latency_seconds.to_dict()),
            "decode_tokens_per_second": (self.decode_tokens_per_second.to_dict()),
            "error_counts": self.error_counts,
        }

        if include_request_details:
            output["requests"] = [result.to_dict() for result in self.request_results]

        return output


def calculate_percentile(
    values: Sequence[float],
    percentile: float,
) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile for empty values.")

    if not 0 <= percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100.")

    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100

    lower_index = floor(position)
    upper_index = ceil(position)

    if lower_index == upper_index:
        return ordered[lower_index]

    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    fraction = position - lower_index

    return lower_value + (upper_value - lower_value) * fraction


def summarize_distribution(
    values: Sequence[float],
) -> DistributionSummary:
    if not values:
        return DistributionSummary(
            count=0,
            mean=None,
            minimum=None,
            p50=None,
            p95=None,
            p99=None,
            maximum=None,
        )

    return DistributionSummary(
        count=len(values),
        mean=mean(values),
        minimum=min(values),
        p50=calculate_percentile(values, 50),
        p95=calculate_percentile(values, 95),
        p99=calculate_percentile(values, 99),
        maximum=max(values),
    )


async def wait_for_server(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    attempts: int = 30,
    delay_seconds: float = 1.0,
) -> None:
    url = f"{base_url.rstrip('/')}/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    last_error: Exception | None = None

    for _ in range(attempts):
        try:
            response = await client.get(
                url,
                headers=headers,
            )

            if response.status_code == 200:
                return

            last_error = RuntimeError(f"Server returned HTTP {response.status_code}.")

        except httpx.HTTPError as error:
            last_error = error

        await asyncio.sleep(delay_seconds)

    message = f"vLLM server is not ready at {base_url}."

    if last_error is not None:
        message += f" Last error: {type(last_error).__name__}: {last_error}"

    raise RuntimeError(message)


async def run_request_batch(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    prompts: Sequence[str],
    max_tokens: int,
    temperature: float,
    request_count: int,
    concurrency: int,
) -> tuple[
    tuple[AsyncRequestResult, ...],
    float,
]:
    if not prompts:
        raise ValueError("At least one prompt is required.")

    if request_count <= 0:
        raise ValueError("request_count must be greater than zero.")

    if concurrency <= 0:
        raise ValueError("concurrency must be greater than zero.")

    semaphore = asyncio.Semaphore(concurrency)

    async def run_one_request(
        request_index: int,
    ) -> AsyncRequestResult:
        prompt = prompts[request_index % len(prompts)]

        async with semaphore:
            return await generate_streaming_request(
                client=client,
                base_url=base_url,
                api_key=api_key,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                request_index=request_index + 1,
                temperature=temperature,
            )

    start_time = time.perf_counter()

    tasks = [
        asyncio.create_task(run_one_request(request_index))
        for request_index in range(request_count)
    ]

    results = await asyncio.gather(*tasks)

    wall_time_seconds = time.perf_counter() - start_time

    return tuple(results), wall_time_seconds


def summarize_request_results(
    *,
    concurrency: int,
    request_results: Sequence[AsyncRequestResult],
    wall_time_seconds: float,
) -> ConcurrentBenchmarkResult:
    successful = [result for result in request_results if result.success]
    failed = [result for result in request_results if not result.success]

    request_count = len(request_results)
    successful_count = len(successful)
    failed_count = len(failed)

    latency_values = [result.latency_seconds for result in successful]
    ttft_values = [
        result.time_to_first_token_seconds
        for result in successful
        if result.time_to_first_token_seconds is not None
    ]
    decode_values = [
        result.decode_seconds for result in successful if result.decode_seconds is not None
    ]
    post_stream_values = [
        result.post_stream_overhead_seconds
        for result in successful
        if result.post_stream_overhead_seconds is not None
    ]

    inter_event_values = [
        latency for result in successful for latency in (result.inter_event_latencies_seconds)
    ]

    decode_throughput_values: list[float] = []

    for result in successful:
        completion_tokens = result.completion_tokens
        decode_seconds = result.decode_seconds

        if (
            completion_tokens is not None
            and completion_tokens > 1
            and decode_seconds is not None
            and decode_seconds > 0
        ):
            decode_throughput_values.append((completion_tokens - 1) / decode_seconds)

    prompt_tokens_total = sum(result.prompt_tokens or 0 for result in successful)
    completion_tokens_total = sum(result.completion_tokens or 0 for result in successful)

    success_rate = successful_count / request_count if request_count > 0 else 0.0

    requests_per_second = successful_count / wall_time_seconds if wall_time_seconds > 0 else 0.0

    output_tokens_per_second = (
        completion_tokens_total / wall_time_seconds if wall_time_seconds > 0 else 0.0
    )

    error_counts = dict(Counter(result.error or "Unknown error" for result in failed))

    return ConcurrentBenchmarkResult(
        concurrency=concurrency,
        request_count=request_count,
        successful_requests=successful_count,
        failed_requests=failed_count,
        success_rate=success_rate,
        wall_time_seconds=wall_time_seconds,
        requests_per_second=requests_per_second,
        output_tokens_per_second=(output_tokens_per_second),
        prompt_tokens_total=prompt_tokens_total,
        completion_tokens_total=(completion_tokens_total),
        latency_seconds=summarize_distribution(latency_values),
        time_to_first_token_seconds=(summarize_distribution(ttft_values)),
        decode_seconds=summarize_distribution(decode_values),
        post_stream_overhead_seconds=(summarize_distribution(post_stream_values)),
        inter_event_latency_seconds=(summarize_distribution(inter_event_values)),
        decode_tokens_per_second=(summarize_distribution(decode_throughput_values)),
        error_counts=error_counts,
        request_results=tuple(request_results),
    )


async def run_concurrent_benchmark(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    prompts: Sequence[str],
    max_tokens: int,
    temperature: float,
    request_count: int,
    concurrency: int,
) -> ConcurrentBenchmarkResult:
    results, wall_time_seconds = await run_request_batch(
        client=client,
        base_url=base_url,
        api_key=api_key,
        model=model,
        prompts=prompts,
        max_tokens=max_tokens,
        temperature=temperature,
        request_count=request_count,
        concurrency=concurrency,
    )

    return summarize_request_results(
        concurrency=concurrency,
        request_results=results,
        wall_time_seconds=wall_time_seconds,
    )


def combine_benchmark_results(
    results: Sequence[ConcurrentBenchmarkResult],
) -> ConcurrentBenchmarkResult:
    if not results:
        raise ValueError("At least one benchmark result is required.")

    concurrency = results[0].concurrency

    if any(result.concurrency != concurrency for result in results):
        raise ValueError("All benchmark results must use the same concurrency.")

    request_results = tuple(request for result in results for request in result.request_results)
    wall_time_seconds = sum(result.wall_time_seconds for result in results)

    return summarize_request_results(
        concurrency=concurrency,
        request_results=request_results,
        wall_time_seconds=wall_time_seconds,
    )


def calculate_coefficient_of_variation(
    values: Sequence[float],
) -> float:
    if not values:
        raise ValueError("Cannot calculate variation for empty values.")

    average = mean(values)

    if average == 0:
        return 0.0

    return pstdev(values) / average


def classify_stability(
    *,
    coefficient_of_variation: float,
    repetition_count: int,
) -> StabilityStatus:
    if repetition_count < 3:
        return "insufficient_data"

    if coefficient_of_variation <= 0.05:
        return "stable"

    if coefficient_of_variation <= 0.10:
        return "noisy"

    return "unstable"


def median_optional(
    values: Sequence[float | None],
) -> float | None:
    available_values = [value for value in values if value is not None]

    if not available_values:
        return None

    return median(available_values)


def summarize_repetition_stability(
    results: Sequence[ConcurrentBenchmarkResult],
) -> RepetitionStabilitySummary:
    if not results:
        raise ValueError("At least one benchmark result is required.")

    requests_per_second = [result.requests_per_second for result in results]
    output_tokens_per_second = [result.output_tokens_per_second for result in results]

    request_throughput_cv = calculate_coefficient_of_variation(requests_per_second)
    token_throughput_cv = calculate_coefficient_of_variation(output_tokens_per_second)

    maximum_cv = max(
        request_throughput_cv,
        token_throughput_cv,
    )

    status = classify_stability(
        coefficient_of_variation=maximum_cv,
        repetition_count=len(results),
    )

    return RepetitionStabilitySummary(
        repetition_count=len(results),
        median_requests_per_second=median(requests_per_second),
        minimum_requests_per_second=min(requests_per_second),
        maximum_requests_per_second=max(requests_per_second),
        requests_per_second_cv=request_throughput_cv,
        median_output_tokens_per_second=median(output_tokens_per_second),
        minimum_output_tokens_per_second=min(output_tokens_per_second),
        maximum_output_tokens_per_second=max(output_tokens_per_second),
        output_tokens_per_second_cv=token_throughput_cv,
        median_latency_p50_seconds=median_optional(
            [result.latency_seconds.p50 for result in results]
        ),
        median_ttft_p50_seconds=median_optional(
            [result.time_to_first_token_seconds.p50 for result in results]
        ),
        status=status,
        valid=status
        in {
            "stable",
            "noisy",
        },
    )
