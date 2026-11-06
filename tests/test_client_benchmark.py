import pytest

from llm_perf_lab.client.benchmark import (
    calculate_percentile,
    summarize_distribution,
)


def test_calculate_percentile() -> None:
    values = [1.0, 2.0, 3.0, 4.0]

    assert calculate_percentile(
        values,
        50,
    ) == pytest.approx(2.5)

    assert calculate_percentile(
        values,
        95,
    ) == pytest.approx(3.85)


def test_calculate_percentile_single_value() -> None:
    assert (
        calculate_percentile(
            [3.0],
            99,
        )
        == 3.0
    )


def test_calculate_percentile_rejects_empty_values() -> None:
    with pytest.raises(
        ValueError,
        match="empty values",
    ):
        calculate_percentile([], 50)


def test_summarize_distribution() -> None:
    summary = summarize_distribution([1.0, 2.0, 3.0])

    assert summary.count == 3
    assert summary.mean == 2.0
    assert summary.minimum == 1.0
    assert summary.p50 == 2.0
    assert summary.maximum == 3.0


def test_summarize_empty_distribution() -> None:
    summary = summarize_distribution([])

    assert summary.count == 0
    assert summary.mean is None
    assert summary.p50 is None
    assert summary.p95 is None
    assert summary.p99 is None
