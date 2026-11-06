from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

from llm_perf_lab.baseline.runner import run_baseline, save_result
from llm_perf_lab.config import GenerationConfig, ModelConfig

DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run a local LLM inference baseline using PyTorch and Hugging Face Transformers."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model identifier or local model path.",
    )

    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt sent to the model.",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device. Default: cuda.",
    )

    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPES),
        default="float16",
        help="Numerical precision used to load the model.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of tokens to generate. Default: 64.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Default: 0.0.",
    )

    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable probabilistic token sampling.",
    )

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of unmeasured warm-up executions. Default: 1.",
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of measured executions. Default: 3.",
    )

    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("Enable streaming generation metrics. Disabled by default."),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/raw/baseline.json"),
        help=("Destination JSON file. Default: results/raw/baseline.json."),
    )

    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("Allow model-specific code from the Hugging Face repository. Disabled by default."),
    )

    return parser.parse_args(arguments)


def validate_arguments(arguments: argparse.Namespace) -> None:
    """Validate command-line values before loading the model."""

    if not arguments.prompt.strip():
        raise ValueError("--prompt cannot be empty.")

    if arguments.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be greater than zero.")

    if arguments.warmup_runs < 0:
        raise ValueError("--warmup-runs cannot be negative.")

    if arguments.repetitions <= 0:
        raise ValueError("--repetitions must be greater than zero.")

    if arguments.do_sample and arguments.temperature <= 0:
        raise ValueError("--temperature must be greater than zero when using --do-sample.")

    if not arguments.do_sample and arguments.temperature != 0:
        raise ValueError("--temperature must be 0 unless --do-sample is enabled.")


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the baseline benchmark and write its result to JSON."""

    try:
        parsed_arguments = parse_arguments(arguments)
        validate_arguments(parsed_arguments)

        model_config = ModelConfig(
            model_name=parsed_arguments.model,
            device=parsed_arguments.device,
            dtype=DTYPES[parsed_arguments.dtype],
            trust_remote_code=parsed_arguments.trust_remote_code,
        )

        generation_config = GenerationConfig(
            max_new_tokens=parsed_arguments.max_new_tokens,
            temperature=parsed_arguments.temperature,
            do_sample=parsed_arguments.do_sample,
        )

        print(f"Loading model: {model_config.model_name}")
        print(f"Device: {model_config.device}")
        print(f"Data type: {parsed_arguments.dtype}")
        print(f"Warm-up runs: {parsed_arguments.warmup_runs}")
        print(f"Measured repetitions: {parsed_arguments.repetitions}")

        result = run_baseline(
            prompt=parsed_arguments.prompt,
            model_config=model_config,
            generation_config=generation_config,
            warmup_runs=parsed_arguments.warmup_runs,
            streaming=parsed_arguments.streaming,
            repetitions=parsed_arguments.repetitions,
        )

        output_path = save_result(
            result=result,
            output_path=parsed_arguments.output,
        )

        summary = result.latency_summary
        memory = result.memory

        print()
        print("Benchmark completed")
        print("-------------------")
        print(f"Mean latency: {summary.mean_seconds:.4f} s")
        print(f"Median latency: {summary.median_seconds:.4f} s")
        print(f"p95 latency: {summary.p95_seconds:.4f} s")
        print(f"Peak allocated GPU memory: {memory.peak_allocated_mb:.2f} MB")
        print(f"Streaming: {parsed_arguments.streaming}")
        print(f"Results written to: {output_path}")

        return 0

    except KeyboardInterrupt:
        print("\nExecution interrupted.", file=sys.stderr)
        return 130

    except Exception as error:
        print(
            f"Baseline execution failed: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
