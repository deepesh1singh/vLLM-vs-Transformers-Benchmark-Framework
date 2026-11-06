from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass(slots=True)
class ModelConfig:
    """Configuration for loading an LLM."""

    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    trust_remote_code: bool = True


@dataclass(slots=True)
class GenerationConfig:
    """Text generation parameters."""

    max_new_tokens: int = 64
    temperature: float = 0.0
    do_sample: bool = False


@dataclass(slots=True)
class OutputConfig:
    """Output directories."""

    results_dir: Path = Path("results")
    raw_dir: Path = Path("results/raw")
    processed_dir: Path = Path("results/processed")
    profiles_dir: Path = Path("results/profiles")
    figures_dir: Path = Path("results/figures")


@dataclass(slots=True)
class AppConfig:
    """Top-level application configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


config = AppConfig()


def resolve_dtype(value: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    return mapping[value]
