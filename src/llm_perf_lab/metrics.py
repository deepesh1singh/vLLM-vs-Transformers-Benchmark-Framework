from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import floor
from statistics import mean, median
from typing import Any

from llm_perf_lab.baseline.token_streamer import TokenEvent


@dataclass(frozen=True)
class InferenceMetrics:
    """
    Metrics collected from one inference execution.

    Streaming fields are optional because a normal non-streaming generation
    does not produce them.
    """

    prompt_tokens: int
    generated_tokens: int
    latency_seconds: float
    output_tokens_per_second: float
    total_tokens_per_second: float

    time_to_first_token_seconds: float | None = None
    decode_after_first_token_seconds: float | None = None
    post_generation_overhead_seconds: float | None = None
    mean_inter_token_latency_seconds: float | None = None
    p50_inter_token_latency_seconds: float | None = None
    p95_inter_token_latency_seconds: float | None = None
    token_event_count: int | None = None
    decode_tokens_per_second: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LatencySummary:
    """
    Aggregate statistics calculated from multiple latency measurements.
    """

    count: int
    mean_seconds: float
    median_seconds: float
    minimum_seconds: float
    maximum_seconds: float
    p90_seconds: float
    p95_seconds: float
    p99_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TokenStreamingMetrics:
    mean_inter_token_latency_seconds: float | None
    p50_inter_token_latency_seconds: float | None
    p95_inter_token_latency_seconds: float | None
    token_event_count: int
    decode_tokens_per_second: float | None


def calculate_throughput(
    *,
    prompt_tokens: int,
    generated_tokens: int,
    latency_seconds: float,
) -> tuple[float, float]:
    """
    Calculate output and total token throughput.

    Output throughput includes only generated tokens.

    Total throughput includes prompt and generated tokens. This is useful as
    an auxiliary metric but should not be interpreted as decode throughput.
    """

    if prompt_tokens < 0:
        raise ValueError("prompt_tokens cannot be negative")

    if generated_tokens < 0:
        raise ValueError("generated_tokens cannot be negative")

    if latency_seconds <= 0:
        raise ValueError("latency_seconds must be greater than zero")

    output_tokens_per_second = generated_tokens / latency_seconds
    total_tokens_per_second = (prompt_tokens + generated_tokens) / latency_seconds

    return output_tokens_per_second, total_tokens_per_second


def create_inference_metrics(
    *,
    prompt_tokens: int,
    generated_tokens: int,
    latency_seconds: float,
    time_to_first_token_seconds: float | None = None,
    decode_after_first_token_seconds: float | None = None,
    mean_inter_token_latency_seconds: float | None = None,
    p50_inter_token_latency_seconds: float | None = None,
    p95_inter_token_latency_seconds: float | None = None,
    token_event_count: int | None = None,
    decode_tokens_per_second: float | None = None,
) -> InferenceMetrics:
    """
    Create and validate metrics for one inference execution.

    Streaming metrics may be omitted for regular non-streaming generation.
    """

    _validate_streaming_metrics(
        generated_tokens=generated_tokens,
        latency_seconds=latency_seconds,
        time_to_first_token_seconds=time_to_first_token_seconds,
        decode_after_first_token_seconds=decode_after_first_token_seconds,
        mean_inter_token_latency_seconds=mean_inter_token_latency_seconds,
        p50_inter_token_latency_seconds=p50_inter_token_latency_seconds,
        p95_inter_token_latency_seconds=p95_inter_token_latency_seconds,
        token_event_count=token_event_count,
        decode_tokens_per_second=decode_tokens_per_second,
    )

    output_throughput, total_throughput = calculate_throughput(
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        latency_seconds=latency_seconds,
    )

    post_generation_overhead_seconds: float | None = None

    if time_to_first_token_seconds is not None and decode_after_first_token_seconds is not None:
        post_generation_overhead_seconds = max(
            latency_seconds - time_to_first_token_seconds - decode_after_first_token_seconds,
            0.0,
        )

    return InferenceMetrics(
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        latency_seconds=latency_seconds,
        output_tokens_per_second=output_throughput,
        total_tokens_per_second=total_throughput,
        time_to_first_token_seconds=time_to_first_token_seconds,
        decode_after_first_token_seconds=decode_after_first_token_seconds,
        post_generation_overhead_seconds=post_generation_overhead_seconds,
        mean_inter_token_latency_seconds=mean_inter_token_latency_seconds,
        p50_inter_token_latency_seconds=p50_inter_token_latency_seconds,
        p95_inter_token_latency_seconds=p95_inter_token_latency_seconds,
        token_event_count=token_event_count,
        decode_tokens_per_second=decode_tokens_per_second,
    )


def _validate_streaming_metrics(
    *,
    generated_tokens: int,
    latency_seconds: float,
    time_to_first_token_seconds: float | None,
    decode_after_first_token_seconds: float | None,
    mean_inter_token_latency_seconds: float | None,
    p50_inter_token_latency_seconds: float | None,
    p95_inter_token_latency_seconds: float | None,
    token_event_count: int | None,
    decode_tokens_per_second: float | None,
) -> None:
    """
    Validate optional metrics produced during streaming generation.
    """

    streaming_values = (
        time_to_first_token_seconds,
        decode_after_first_token_seconds,
        mean_inter_token_latency_seconds,
        p50_inter_token_latency_seconds,
        p95_inter_token_latency_seconds,
        token_event_count,
        decode_tokens_per_second,
    )

    has_streaming_metrics = any(value is not None for value in streaming_values)

    if not has_streaming_metrics:
        return

    if time_to_first_token_seconds is None:
        raise ValueError("time_to_first_token_seconds is required for streaming metrics")

    if decode_after_first_token_seconds is None:
        raise ValueError("decode_after_first_token_seconds is required for streaming metrics")

    if token_event_count is None:
        raise ValueError("token_event_count is required for streaming metrics")

    if time_to_first_token_seconds < 0:
        raise ValueError("time_to_first_token_seconds cannot be negative")

    if time_to_first_token_seconds > latency_seconds:
        raise ValueError("time_to_first_token_seconds cannot exceed latency_seconds")

    if decode_after_first_token_seconds < 0:
        raise ValueError("decode_after_first_token_seconds cannot be negative")

    last_token_time_seconds = time_to_first_token_seconds + decode_after_first_token_seconds

    tolerance_seconds = 1e-6

    if last_token_time_seconds > latency_seconds + tolerance_seconds:
        raise ValueError("The last-token timestamp cannot exceed latency_seconds")

    if token_event_count < 0:
        raise ValueError("token_event_count cannot be negative")

    if token_event_count != generated_tokens:
        raise ValueError("token_event_count must equal generated_tokens")

    optional_non_negative_metrics = {
        "mean_inter_token_latency_seconds": (mean_inter_token_latency_seconds),
        "p50_inter_token_latency_seconds": (p50_inter_token_latency_seconds),
        "p95_inter_token_latency_seconds": (p95_inter_token_latency_seconds),
        "decode_tokens_per_second": decode_tokens_per_second,
    }

    for metric_name, metric_value in optional_non_negative_metrics.items():
        if metric_value is not None and metric_value < 0:
            raise ValueError(f"{metric_name} cannot be negative")

    if token_event_count <= 1:
        interval_metrics = (
            mean_inter_token_latency_seconds,
            p50_inter_token_latency_seconds,
            p95_inter_token_latency_seconds,
        )

        if any(value is not None for value in interval_metrics):
            raise ValueError(
                "Inter-token latency metrics must be None when fewer "
                "than two token events were produced"
            )


def percentile(
    values: Sequence[float],
    percentile_value: float,
) -> float:
    """
    Calculate a percentile using linear interpolation.
    """

    if not values:
        raise ValueError("At least one value is required")

    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile_value must be between 0 and 100")

    sorted_values = sorted(values)

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (percentile_value / 100) * (len(sorted_values) - 1)

    lower_index = floor(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    interpolation_weight = position - lower_index

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]

    return lower_value + (upper_value - lower_value) * interpolation_weight


def summarize_latencies(
    latencies_seconds: Sequence[float],
) -> LatencySummary:
    """
    Summarize latency measurements from multiple executions.
    """

    if not latencies_seconds:
        raise ValueError("At least one latency is required")

    if any(latency < 0 for latency in latencies_seconds):
        raise ValueError("Latency values cannot be negative")

    return LatencySummary(
        count=len(latencies_seconds),
        mean_seconds=mean(latencies_seconds),
        median_seconds=median(latencies_seconds),
        minimum_seconds=min(latencies_seconds),
        maximum_seconds=max(latencies_seconds),
        p90_seconds=percentile(latencies_seconds, 90),
        p95_seconds=percentile(latencies_seconds, 95),
        p99_seconds=percentile(latencies_seconds, 99),
    )


def calculate_token_streaming_metrics(
    *,
    token_events: Sequence[TokenEvent],
    inter_token_latencies_seconds: Sequence[float],
    decode_after_first_token_seconds: float,
) -> TokenStreamingMetrics:
    """
    Calculate aggregate metrics from token-level streaming events.
    """

    token_event_count = len(token_events)

    if token_event_count == 0:
        raise ValueError("At least one token event is required")

    if inter_token_latencies_seconds:
        mean_inter_token_latency_seconds = mean(inter_token_latencies_seconds)
        p50_inter_token_latency_seconds = percentile(
            inter_token_latencies_seconds,
            50,
        )
        p95_inter_token_latency_seconds = percentile(
            inter_token_latencies_seconds,
            95,
        )
    else:
        mean_inter_token_latency_seconds = None
        p50_inter_token_latency_seconds = None
        p95_inter_token_latency_seconds = None

    decode_token_count = max(token_event_count - 1, 0)

    decode_tokens_per_second = (
        decode_token_count / decode_after_first_token_seconds
        if decode_after_first_token_seconds > 0
        else None
    )

    return TokenStreamingMetrics(
        mean_inter_token_latency_seconds=(mean_inter_token_latency_seconds),
        p50_inter_token_latency_seconds=(p50_inter_token_latency_seconds),
        p95_inter_token_latency_seconds=(p95_inter_token_latency_seconds),
        token_event_count=token_event_count,
        decode_tokens_per_second=decode_tokens_per_second,
    )
