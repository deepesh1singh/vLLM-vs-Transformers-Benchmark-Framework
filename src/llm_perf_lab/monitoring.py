from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(slots=True)
class GPUInfo:
    """Information about the active CUDA device."""

    available: bool
    device_count: int
    device_index: int | None = None
    device_name: str | None = None
    cuda_version: str | None = None
    torch_version: str | None = None
    compute_capability: str | None = None
    total_memory_mb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the GPU information as a JSON-serializable dictionary."""

        return asdict(self)


@dataclass(slots=True)
class GPUMemoryStats:
    """GPU memory statistics for the active CUDA device."""

    allocated_mb: float
    reserved_mb: float
    peak_allocated_mb: float
    peak_reserved_mb: float

    def to_dict(self) -> dict[str, float]:
        """Return memory statistics as a dictionary."""

        return asdict(self)


def bytes_to_mb(value: int) -> float:
    """Convert bytes to megabytes."""

    return round(value / (1024**2), 2)


def get_gpu_info(device_index: str | torch.device | int = 0) -> GPUInfo:
    """Collect information about a CUDA device."""

    device_index = resolve_cuda_device_index(device_index)
    if not torch.cuda.is_available():
        return GPUInfo(
            available=False,
            device_count=0,
            torch_version=torch.__version__,
        )

    device_count = torch.cuda.device_count()

    if device_index < 0 or device_index >= device_count:
        raise ValueError(
            f"Invalid CUDA device index {device_index}. Available devices: 0 to {device_count - 1}."
        )

    properties = torch.cuda.get_device_properties(device_index)

    return GPUInfo(
        available=True,
        device_count=device_count,
        device_index=device_index,
        device_name=properties.name,
        cuda_version=torch.version.cuda,
        torch_version=torch.__version__,
        compute_capability=f"{properties.major}.{properties.minor}",
        total_memory_mb=bytes_to_mb(properties.total_memory),
    )


def resolve_cuda_device_index(
    device: str | torch.device | int,
) -> int:
    """HELPER: Resolve a CUDA device index from a string, torch.device, or int."""

    if isinstance(device, int):
        return device

    resolved_device = torch.device(device)

    if resolved_device.type != "cuda":
        raise ValueError(f"Expected a CUDA device, received {resolved_device}")

    if resolved_device.index is None or resolved_device.index < 0:
        return torch.cuda.current_device()

    return resolved_device.index


def reset_gpu_memory_stats(device_index: str | torch.device | int = 0) -> None:
    """Reset peak GPU memory counters."""

    if not torch.cuda.is_available():
        return

    torch.cuda.reset_peak_memory_stats(device_index)


def get_gpu_memory_stats(device_index: str | torch.device | int = 0) -> GPUMemoryStats:
    """Return current and peak CUDA memory usage."""

    if not torch.cuda.is_available():
        return GPUMemoryStats(
            allocated_mb=0.0,
            reserved_mb=0.0,
            peak_allocated_mb=0.0,
            peak_reserved_mb=0.0,
        )

    return GPUMemoryStats(
        allocated_mb=bytes_to_mb(torch.cuda.memory_allocated(device_index)),
        reserved_mb=bytes_to_mb(torch.cuda.memory_reserved(device_index)),
        peak_allocated_mb=bytes_to_mb(torch.cuda.max_memory_allocated(device_index)),
        peak_reserved_mb=bytes_to_mb(torch.cuda.max_memory_reserved(device_index)),
    )


def synchronize_cuda() -> None:
    """Wait until all CUDA operations have completed."""

    if torch.cuda.is_available():
        torch.cuda.synchronize()
