from typing import Any

import pytest

from llm_perf_lab.metrics import (
    calculate_throughput,
    create_inference_metrics,
    percentile,
    summarize_latencies,
)


def test_calculate_throughput() -> None:
    output_throughput, total_throughput = calculate_throughput(
        prompt_tokens=20,
        generated_tokens=30,
        latency_seconds=2.0,
    )

    assert output_throughput == 15.0
    assert total_throughput == 25.0


@pytest.mark.parametrize(
    ("prompt_tokens", "generated_tokens", "latency_seconds"),
    [
        (-1, 10, 1.0),
        (10, -1, 1.0),
        (10, 10, 0.0),
        (10, 10, -1.0),
    ],
)
def test_calculate_throughput_rejects_invalid_values(
    prompt_tokens: int,
    generated_tokens: int,
    latency_seconds: float,
) -> None:
    with pytest.raises(ValueError):
        calculate_throughput(
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            latency_seconds=latency_seconds,
        )


def test_create_inference_metrics() -> None:
    metrics = create_inference_metrics(
        prompt_tokens=10,
        generated_tokens=20,
        latency_seconds=2.0,
    )

    assert metrics.prompt_tokens == 10
    assert metrics.generated_tokens == 20
    assert metrics.latency_seconds == 2.0
    assert metrics.output_tokens_per_second == 10.0
    assert metrics.total_tokens_per_second == 15.0


def test_percentile_with_single_value() -> None:
    assert percentile([0.5], 95) == 0.5


def test_percentile_with_linear_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0]

    assert percentile(values, 50) == pytest.approx(2.5)
    assert percentile(values, 90) == pytest.approx(3.7)


def test_percentile_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="At least one value"):
        percentile([], 95)


@pytest.mark.parametrize("percentile_value", [-1, 101])
def test_percentile_rejects_invalid_percentile(
    percentile_value: float,
) -> None:
    with pytest.raises(ValueError):
        percentile([1.0, 2.0], percentile_value)


def test_summarize_latencies() -> None:
    summary = summarize_latencies([0.1, 0.2, 0.3, 0.4, 0.5])

    assert summary.count == 5
    assert summary.mean_seconds == pytest.approx(0.3)
    assert summary.median_seconds == pytest.approx(0.3)
    assert summary.minimum_seconds == pytest.approx(0.1)
    assert summary.maximum_seconds == pytest.approx(0.5)
    assert summary.p90_seconds == pytest.approx(0.46)
    assert summary.p95_seconds == pytest.approx(0.48)
    assert summary.p99_seconds == pytest.approx(0.496)


def test_summarize_latencies_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError, match="At least one latency"):
        summarize_latencies([])


def test_summarize_latencies_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        summarize_latencies([0.1, -0.2, 0.3])


def test_create_inference_metrics_with_streaming_values() -> None:
    metrics = create_inference_metrics(
        prompt_tokens=10,
        generated_tokens=4,
        latency_seconds=2.0,
        time_to_first_token_seconds=0.2,
        decode_after_first_token_seconds=1.5,
        mean_inter_token_latency_seconds=0.5,
        p50_inter_token_latency_seconds=0.5,
        p95_inter_token_latency_seconds=0.7,
        token_event_count=4,
        decode_tokens_per_second=2.0,
    )

    assert metrics.time_to_first_token_seconds == 0.2
    assert metrics.decode_after_first_token_seconds == 1.5
    assert metrics.post_generation_overhead_seconds == pytest.approx(0.3)
    assert metrics.mean_inter_token_latency_seconds == 0.5
    assert metrics.p50_inter_token_latency_seconds == 0.5
    assert metrics.p95_inter_token_latency_seconds == 0.7
    assert metrics.token_event_count == 4
    assert metrics.decode_tokens_per_second == 2.0


def test_create_inference_metrics_without_streaming_values() -> None:
    metrics = create_inference_metrics(
        prompt_tokens=10,
        generated_tokens=4,
        latency_seconds=2.0,
    )

    assert metrics.time_to_first_token_seconds is None
    assert metrics.decode_after_first_token_seconds is None
    assert metrics.post_generation_overhead_seconds is None
    assert metrics.mean_inter_token_latency_seconds is None
    assert metrics.p50_inter_token_latency_seconds is None
    assert metrics.p95_inter_token_latency_seconds is None
    assert metrics.token_event_count is None
    assert metrics.decode_tokens_per_second is None


@pytest.mark.parametrize(
    ("value", "expected_message"),
    [
        (-0.1, "time_to_first_token_seconds cannot be negative"),
        (
            2.1,
            "time_to_first_token_seconds cannot exceed latency_seconds",
        ),
    ],
)
def test_rejects_invalid_time_to_first_token(
    value: float,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        create_inference_metrics(
            prompt_tokens=10,
            generated_tokens=4,
            latency_seconds=2.0,
            time_to_first_token_seconds=value,
            decode_after_first_token_seconds=1.0,
            token_event_count=4,
        )


def test_rejects_negative_decode_after_first_token() -> None:
    with pytest.raises(
        ValueError,
        match="decode_after_first_token_seconds cannot be negative",
    ):
        create_inference_metrics(
            prompt_tokens=10,
            generated_tokens=4,
            latency_seconds=2.0,
            time_to_first_token_seconds=0.2,
            decode_after_first_token_seconds=-0.1,
            token_event_count=4,
        )


def test_rejects_last_token_time_exceeding_latency() -> None:
    with pytest.raises(
        ValueError,
        match="The last-token timestamp cannot exceed latency_seconds",
    ):
        create_inference_metrics(
            prompt_tokens=10,
            generated_tokens=4,
            latency_seconds=2.0,
            time_to_first_token_seconds=0.5,
            decode_after_first_token_seconds=1.6,
            token_event_count=4,
        )


def test_accepts_post_generation_overhead() -> None:
    metrics = create_inference_metrics(
        prompt_tokens=10,
        generated_tokens=4,
        latency_seconds=2.0,
        time_to_first_token_seconds=0.2,
        decode_after_first_token_seconds=1.5,
        token_event_count=4,
    )

    assert metrics.post_generation_overhead_seconds == pytest.approx(0.3)


def test_accepts_small_last_token_rounding_difference() -> None:
    metrics = create_inference_metrics(
        prompt_tokens=10,
        generated_tokens=4,
        latency_seconds=2.0,
        time_to_first_token_seconds=0.5,
        decode_after_first_token_seconds=1.5000005,
        token_event_count=4,
    )

    assert metrics.post_generation_overhead_seconds == 0.0


@pytest.mark.parametrize(
    ("field_name", "value", "expected_message"),
    [
        (
            "mean_inter_token_latency_seconds",
            -0.1,
            "mean_inter_token_latency_seconds cannot be negative",
        ),
        (
            "p50_inter_token_latency_seconds",
            -0.1,
            "p50_inter_token_latency_seconds cannot be negative",
        ),
        (
            "p95_inter_token_latency_seconds",
            -0.1,
            "p95_inter_token_latency_seconds cannot be negative",
        ),
        (
            "decode_tokens_per_second",
            -0.1,
            "decode_tokens_per_second cannot be negative",
        ),
    ],
)
def test_rejects_negative_streaming_metric(
    field_name: str,
    value: float,
    expected_message: str,
) -> None:
    arguments: dict[str, Any] = {
        "prompt_tokens": 10,
        "generated_tokens": 4,
        "latency_seconds": 2.0,
        "time_to_first_token_seconds": 0.2,
        "decode_after_first_token_seconds": 1.5,
        "token_event_count": 4,
        field_name: value,
    }

    with pytest.raises(ValueError, match=expected_message):
        create_inference_metrics(**arguments)


def test_rejects_negative_token_event_count() -> None:
    with pytest.raises(
        ValueError,
        match="token_event_count cannot be negative",
    ):
        create_inference_metrics(
            prompt_tokens=10,
            generated_tokens=4,
            latency_seconds=2.0,
            time_to_first_token_seconds=0.2,
            decode_after_first_token_seconds=1.5,
            token_event_count=-1,
        )


def test_rejects_token_event_count_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="token_event_count must equal generated_tokens",
    ):
        create_inference_metrics(
            prompt_tokens=10,
            generated_tokens=4,
            latency_seconds=2.0,
            time_to_first_token_seconds=0.2,
            decode_after_first_token_seconds=1.5,
            token_event_count=3,
        )


def test_rejects_inter_token_metrics_with_one_token() -> None:
    with pytest.raises(
        ValueError,
        match="Inter-token latency metrics must be None",
    ):
        create_inference_metrics(
            prompt_tokens=10,
            generated_tokens=1,
            latency_seconds=0.5,
            time_to_first_token_seconds=0.2,
            decode_after_first_token_seconds=0.0,
            token_event_count=1,
            mean_inter_token_latency_seconds=0.1,
        )


def test_accepts_one_token_without_interval_metrics() -> None:
    metrics = create_inference_metrics(
        prompt_tokens=10,
        generated_tokens=1,
        latency_seconds=0.5,
        time_to_first_token_seconds=0.2,
        decode_after_first_token_seconds=0.0,
        token_event_count=1,
    )

    assert metrics.token_event_count == 1
    assert metrics.mean_inter_token_latency_seconds is None
    assert metrics.p50_inter_token_latency_seconds is None
    assert metrics.p95_inter_token_latency_seconds is None
