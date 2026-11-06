from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, cast

JsonObject = dict[str, Any]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two LLM benchmark result files.",
    )

    parser.add_argument(
        "result_a",
        type=Path,
        help="First benchmark JSON file, used as the baseline.",
    )
    parser.add_argument(
        "result_b",
        type=Path,
        help="Second benchmark JSON file.",
    )
    parser.add_argument(
        "--label-a",
        help="Display label for the first result.",
    )
    parser.add_argument(
        "--label-b",
        help="Display label for the second result.",
    )

    return parser.parse_args()


def load_result(path: Path) -> JsonObject:
    if not path.is_file():
        raise FileNotFoundError(f"Result file does not exist: {path}")

    try:
        loaded_value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in result file: {path}") from error

    if not isinstance(loaded_value, dict):
        raise ValueError(f"Expected a JSON object in result file: {path}")

    return cast(JsonObject, loaded_value)


def nested_float(
    result: JsonObject,
    *keys: str,
) -> float | None:
    current_value: Any = result

    for key in keys:
        if not isinstance(current_value, dict):
            return None

        current_value = current_value.get(key)

    if isinstance(current_value, bool):
        return None

    if not isinstance(current_value, (int, float)):
        return None

    return float(current_value)


def mean_run_metric(
    result: JsonObject,
    metric_name: str,
) -> float | None:
    runs = result.get("runs")

    if not isinstance(runs, list):
        return None

    values: list[float] = []

    for run in runs:
        if not isinstance(run, dict):
            continue

        metrics = run.get("metrics")

        if not isinstance(metrics, dict):
            continue

        value = metrics.get(metric_name)

        if isinstance(value, bool):
            continue

        if isinstance(value, (int, float)):
            values.append(float(value))

    if not values:
        return None

    return mean(values)


def percentage_change(
    baseline: float | None,
    candidate: float | None,
) -> float | None:
    if baseline is None or candidate is None:
        return None

    if baseline == 0:
        return None

    return ((candidate - baseline) / baseline) * 100


def format_value(
    value: float | None,
    *,
    multiplier: float = 1.0,
    decimals: int = 3,
) -> str:
    if value is None:
        return "—"

    return f"{value * multiplier:.{decimals}f}"


def format_change(change: float | None) -> str:
    if change is None:
        return "—"

    return f"{change:+.2f}%"


def check_compatible_configuration(
    result_a: JsonObject,
    result_b: JsonObject,
) -> None:
    differences: list[str] = []

    if result_a.get("model") != result_b.get("model"):
        differences.append("model configuration")

    if result_a.get("generation") != result_b.get("generation"):
        differences.append("generation configuration")

    if differences:
        joined_differences = ", ".join(differences)

        print(f"Warning: benchmark files use different {joined_differences}.")
        print()


def print_comparison(
    result_a: JsonObject,
    result_b: JsonObject,
    *,
    label_a: str,
    label_b: str,
) -> None:
    rows = [
        (
            "Mean latency",
            nested_float(result_a, "latency_summary", "mean_seconds"),
            nested_float(result_b, "latency_summary", "mean_seconds"),
            1000.0,
            "ms",
        ),
        (
            "Median latency",
            nested_float(result_a, "latency_summary", "median_seconds"),
            nested_float(result_b, "latency_summary", "median_seconds"),
            1000.0,
            "ms",
        ),
        (
            "p95 latency",
            nested_float(result_a, "latency_summary", "p95_seconds"),
            nested_float(result_b, "latency_summary", "p95_seconds"),
            1000.0,
            "ms",
        ),
        (
            "Output throughput",
            mean_run_metric(result_a, "output_tokens_per_second"),
            mean_run_metric(result_b, "output_tokens_per_second"),
            1.0,
            "tok/s",
        ),
        (
            "Total throughput",
            mean_run_metric(result_a, "total_tokens_per_second"),
            mean_run_metric(result_b, "total_tokens_per_second"),
            1.0,
            "tok/s",
        ),
        (
            "Peak GPU memory",
            nested_float(result_a, "memory", "peak_allocated_mb"),
            nested_float(result_b, "memory", "peak_allocated_mb"),
            1.0,
            "MB",
        ),
        (
            "Time to first token",
            mean_run_metric(result_a, "time_to_first_token_seconds"),
            mean_run_metric(result_b, "time_to_first_token_seconds"),
            1000.0,
            "ms",
        ),
        (
            "Mean inter-token latency",
            mean_run_metric(
                result_a,
                "mean_inter_token_latency_seconds",
            ),
            mean_run_metric(
                result_b,
                "mean_inter_token_latency_seconds",
            ),
            1000.0,
            "ms",
        ),
        (
            "p95 inter-token latency",
            mean_run_metric(
                result_a,
                "p95_inter_token_latency_seconds",
            ),
            mean_run_metric(
                result_b,
                "p95_inter_token_latency_seconds",
            ),
            1000.0,
            "ms",
        ),
        (
            "Decode throughput",
            mean_run_metric(result_a, "decode_tokens_per_second"),
            mean_run_metric(result_b, "decode_tokens_per_second"),
            1.0,
            "tok/s",
        ),
        (
            "Post-generation overhead",
            mean_run_metric(
                result_a,
                "post_generation_overhead_seconds",
            ),
            mean_run_metric(
                result_b,
                "post_generation_overhead_seconds",
            ),
            1000.0,
            "ms",
        ),
    ]

    metric_width = max(len(row[0]) for row in rows)
    label_width = max(len(label_a), len(label_b), 12)

    print("Benchmark comparison")
    print("=" * 78)
    print(
        f"{'Metric':<{metric_width}}  "
        f"{label_a:>{label_width}}  "
        f"{label_b:>{label_width}}  "
        f"{'Change':>9}  "
        "Unit"
    )
    print("-" * 78)

    for metric, value_a, value_b, multiplier, unit in rows:
        change = percentage_change(value_a, value_b)

        print(
            f"{metric:<{metric_width}}  "
            f"{format_value(value_a, multiplier=multiplier):>{label_width}}  "
            f"{format_value(value_b, multiplier=multiplier):>{label_width}}  "
            f"{format_change(change):>9}  "
            f"{unit}"
        )


def main() -> None:
    arguments = parse_arguments()

    result_a = load_result(arguments.result_a)
    result_b = load_result(arguments.result_b)

    label_a = arguments.label_a or arguments.result_a.stem
    label_b = arguments.label_b or arguments.result_b.stem

    check_compatible_configuration(result_a, result_b)

    print_comparison(
        result_a,
        result_b,
        label_a=label_a,
        label_b=label_b,
    )


if __name__ == "__main__":
    main()
