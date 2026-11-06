import pytest
import torch

from llm_perf_lab.monitoring import (
    bytes_to_mb,
    get_gpu_info,
    get_gpu_memory_stats,
    reset_gpu_memory_stats,
    synchronize_cuda,
)


def test_bytes_to_mb() -> None:
    assert bytes_to_mb(0) == 0.0
    assert bytes_to_mb(1024 * 1024) == 1.0
    assert bytes_to_mb(512 * 1024 * 1024) == 512.0


def test_get_gpu_info_returns_expected_structure() -> None:
    gpu_info = get_gpu_info()

    assert gpu_info.available == torch.cuda.is_available()
    assert gpu_info.device_count == torch.cuda.device_count()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
def test_get_gpu_info_with_cuda() -> None:
    gpu_info = get_gpu_info()

    assert gpu_info.available is True
    assert gpu_info.device_count >= 1
    assert gpu_info.device_index == 0
    assert gpu_info.device_name
    assert gpu_info.compute_capability


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
def test_gpu_memory_stats_are_non_negative() -> None:
    reset_gpu_memory_stats()

    tensor = torch.zeros(
        1024,
        1024,
        device="cuda",
        dtype=torch.float32,
    )

    synchronize_cuda()
    stats = get_gpu_memory_stats()

    assert stats.allocated_mb >= 0
    assert stats.reserved_mb >= 0
    assert stats.peak_allocated_mb >= stats.allocated_mb
    assert stats.peak_reserved_mb >= stats.reserved_mb

    del tensor
    torch.cuda.empty_cache()


def test_synchronize_cuda_does_not_fail() -> None:
    synchronize_cuda()
