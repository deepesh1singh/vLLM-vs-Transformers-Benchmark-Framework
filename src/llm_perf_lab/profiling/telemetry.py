from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from math import ceil, floor
from statistics import mean, median
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Any

import psutil
import pynvml


@dataclass(frozen=True)
class NumericSummary:
    count: int
    mean: float | None
    minimum: float | None
    p50: float | None
    p95: float | None
    maximum: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TelemetrySample:
    elapsed_seconds: float

    gpu_utilization_percent: float | None
    memory_io_utilization_percent: float | None
    gpu_memory_used_mb: float | None

    power_watts: float | None
    temperature_celsius: float | None

    graphics_clock_mhz: float | None
    memory_clock_mhz: float | None

    cpu_utilization_percent: float
    system_memory_utilization_percent: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TelemetrySummary:
    gpu_index: int
    gpu_name: str

    sample_interval_seconds: float
    duration_seconds: float
    sample_count: int

    estimated_gpu_energy_joules: float | None

    gpu_utilization_percent: NumericSummary
    memory_io_utilization_percent: NumericSummary
    gpu_memory_used_mb: NumericSummary

    power_watts: NumericSummary
    temperature_celsius: NumericSummary

    graphics_clock_mhz: NumericSummary
    memory_clock_mhz: NumericSummary

    cpu_utilization_percent: NumericSummary
    system_memory_utilization_percent: NumericSummary

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TelemetryResult:
    summary: TelemetrySummary
    samples: tuple[TelemetrySample, ...]
    sampling_error: str | None

    def to_dict(
        self,
        *,
        include_samples: bool,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "summary": self.summary.to_dict(),
            "sampling_error": self.sampling_error,
        }

        if include_samples:
            data["samples"] = [sample.to_dict() for sample in self.samples]

        return data


@dataclass(frozen=True)
class LevelTelemetrySummary:
    repetition_count: int

    median_gpu_utilization_percent: float | None
    median_memory_io_utilization_percent: float | None
    median_power_watts: float | None

    maximum_gpu_memory_used_mb: float | None
    maximum_temperature_celsius: float | None

    median_graphics_clock_mhz: float | None
    median_cpu_utilization_percent: float | None

    median_estimated_gpu_energy_joules: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _calculate_percentile(
    values: Sequence[float],
    percentile: float,
) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile for empty values.")

    if not 0.0 <= percentile <= 1.0:
        raise ValueError("Percentile must be between 0 and 1.")

    sorted_values = sorted(values)

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * percentile
    lower_index = floor(position)
    upper_index = ceil(position)

    if lower_index == upper_index:
        return sorted_values[lower_index]

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    fraction = position - lower_index

    return lower_value + (upper_value - lower_value) * fraction


def _summarize_optional_values(
    values: Sequence[float | None],
) -> NumericSummary:
    available_values = [value for value in values if value is not None]

    if not available_values:
        return NumericSummary(
            count=0,
            mean=None,
            minimum=None,
            p50=None,
            p95=None,
            maximum=None,
        )

    return NumericSummary(
        count=len(available_values),
        mean=mean(available_values),
        minimum=min(available_values),
        p50=_calculate_percentile(
            available_values,
            0.50,
        ),
        p95=_calculate_percentile(
            available_values,
            0.95,
        ),
        maximum=max(available_values),
    )


def _median_optional(
    values: Sequence[float | None],
) -> float | None:
    available_values = [value for value in values if value is not None]

    if not available_values:
        return None

    return median(available_values)


def _maximum_optional(
    values: Sequence[float | None],
) -> float | None:
    available_values = [value for value in values if value is not None]

    if not available_values:
        return None

    return max(available_values)


def _estimate_energy_joules(
    samples: Sequence[TelemetrySample],
) -> float | None:
    power_samples = [
        (sample.elapsed_seconds, sample.power_watts)
        for sample in samples
        if sample.power_watts is not None
    ]

    if len(power_samples) < 2:
        return None

    energy_joules = 0.0

    for previous, current in itertools.pairwise(
        power_samples,
    ):
        previous_time, previous_power = previous
        current_time, current_power = current

        elapsed_seconds = current_time - previous_time
        average_power = (previous_power + current_power) / 2.0

        energy_joules += average_power * elapsed_seconds

    return energy_joules


class TelemetryCollector:
    def __init__(
        self,
        *,
        gpu_index: int = 0,
        sample_interval_seconds: float = 0.2,
    ) -> None:
        if gpu_index < 0:
            raise ValueError("GPU index cannot be negative.")

        if sample_interval_seconds <= 0:
            raise ValueError("Sample interval must be positive.")

        self._gpu_index = gpu_index
        self._sample_interval_seconds = sample_interval_seconds

        self._stop_event = Event()
        self._lock = Lock()

        self._samples: list[TelemetrySample] = []
        self._sampling_error: str | None = None

        self._thread: Thread | None = None
        self._handle: Any | None = None
        self._gpu_name = "unknown"

        self._started_at: float | None = None
        self._stopped_at: float | None = None
        self._nvml_initialized = False

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Telemetry collection has already started.")

        pynvml.nvmlInit()
        self._nvml_initialized = True

        try:
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)

            raw_name = pynvml.nvmlDeviceGetName(self._handle)

            if isinstance(raw_name, bytes):
                self._gpu_name = raw_name.decode(errors="replace")
            else:
                self._gpu_name = str(raw_name)

            psutil.cpu_percent(interval=None)

            self._started_at = perf_counter()

            self._thread = Thread(
                target=self._sampling_loop,
                name="gpu-telemetry",
                daemon=True,
            )
            self._thread.start()
        except Exception:
            self._shutdown_nvml()
            raise

    def stop(self) -> TelemetryResult:
        if self._thread is None:
            raise RuntimeError("Telemetry collection has not started.")

        self._stop_event.set()
        self._thread.join()

        self._stopped_at = perf_counter()

        self._shutdown_nvml()

        return self._build_result()

    def _shutdown_nvml(self) -> None:
        if not self._nvml_initialized:
            return

        pynvml.nvmlShutdown()
        self._nvml_initialized = False

    def _sampling_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                sample = self._collect_sample()

                with self._lock:
                    self._samples.append(sample)

                self._stop_event.wait(self._sample_interval_seconds)
        except Exception as error:
            with self._lock:
                self._sampling_error = f"{type(error).__name__}: {error}"

    def _read_nvml_number(
        self,
        query: Callable[[], int | float],
    ) -> float | None:
        try:
            return float(query())
        except pynvml.NVMLError:
            return None

    def _collect_sample(self) -> TelemetrySample:
        if self._started_at is None:
            raise RuntimeError("Telemetry collection has not started.")

        if self._handle is None:
            raise RuntimeError("NVML device handle is unavailable.")

        handle = self._handle

        gpu_utilization: float | None = None
        memory_io_utilization: float | None = None

        try:
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization = float(utilization.gpu)
            memory_io_utilization = float(utilization.memory)
        except pynvml.NVMLError:
            pass

        gpu_memory_used_mb: float | None = None

        try:
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_used_mb = float(memory_info.used) / 1024.0 / 1024.0
        except pynvml.NVMLError:
            pass

        power_watts = self._read_nvml_number(
            lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        )

        temperature_celsius = self._read_nvml_number(
            lambda: pynvml.nvmlDeviceGetTemperature(
                handle,
                pynvml.NVML_TEMPERATURE_GPU,
            )
        )

        graphics_clock_mhz = self._read_nvml_number(
            lambda: pynvml.nvmlDeviceGetClockInfo(
                handle,
                pynvml.NVML_CLOCK_GRAPHICS,
            )
        )

        memory_clock_mhz = self._read_nvml_number(
            lambda: pynvml.nvmlDeviceGetClockInfo(
                handle,
                pynvml.NVML_CLOCK_MEM,
            )
        )

        return TelemetrySample(
            elapsed_seconds=(perf_counter() - self._started_at),
            gpu_utilization_percent=(gpu_utilization),
            memory_io_utilization_percent=(memory_io_utilization),
            gpu_memory_used_mb=(gpu_memory_used_mb),
            power_watts=power_watts,
            temperature_celsius=(temperature_celsius),
            graphics_clock_mhz=(graphics_clock_mhz),
            memory_clock_mhz=(memory_clock_mhz),
            cpu_utilization_percent=float(psutil.cpu_percent(interval=None)),
            system_memory_utilization_percent=float(psutil.virtual_memory().percent),
        )

    def _build_result(self) -> TelemetryResult:
        with self._lock:
            samples = tuple(self._samples)
            sampling_error = self._sampling_error

        if self._started_at is None or self._stopped_at is None:
            raise RuntimeError("Telemetry collection did not stop correctly.")

        summary = TelemetrySummary(
            gpu_index=self._gpu_index,
            gpu_name=self._gpu_name,
            sample_interval_seconds=(self._sample_interval_seconds),
            duration_seconds=(self._stopped_at - self._started_at),
            sample_count=len(samples),
            estimated_gpu_energy_joules=(_estimate_energy_joules(samples)),
            gpu_utilization_percent=(
                _summarize_optional_values([sample.gpu_utilization_percent for sample in samples])
            ),
            memory_io_utilization_percent=(
                _summarize_optional_values(
                    [sample.memory_io_utilization_percent for sample in samples]
                )
            ),
            gpu_memory_used_mb=(
                _summarize_optional_values([sample.gpu_memory_used_mb for sample in samples])
            ),
            power_watts=(_summarize_optional_values([sample.power_watts for sample in samples])),
            temperature_celsius=(
                _summarize_optional_values([sample.temperature_celsius for sample in samples])
            ),
            graphics_clock_mhz=(
                _summarize_optional_values([sample.graphics_clock_mhz for sample in samples])
            ),
            memory_clock_mhz=(
                _summarize_optional_values([sample.memory_clock_mhz for sample in samples])
            ),
            cpu_utilization_percent=(
                _summarize_optional_values([sample.cpu_utilization_percent for sample in samples])
            ),
            system_memory_utilization_percent=(
                _summarize_optional_values(
                    [sample.system_memory_utilization_percent for sample in samples]
                )
            ),
        )

        return TelemetryResult(
            summary=summary,
            samples=samples,
            sampling_error=sampling_error,
        )


def summarize_telemetry_results(
    results: Sequence[TelemetryResult],
) -> LevelTelemetrySummary:
    if not results:
        raise ValueError("At least one telemetry result is required.")

    summaries = [result.summary for result in results]

    return LevelTelemetrySummary(
        repetition_count=len(results),
        median_gpu_utilization_percent=(
            _median_optional([summary.gpu_utilization_percent.mean for summary in summaries])
        ),
        median_memory_io_utilization_percent=(
            _median_optional([summary.memory_io_utilization_percent.mean for summary in summaries])
        ),
        median_power_watts=(_median_optional([summary.power_watts.mean for summary in summaries])),
        maximum_gpu_memory_used_mb=(
            _maximum_optional([summary.gpu_memory_used_mb.maximum for summary in summaries])
        ),
        maximum_temperature_celsius=(
            _maximum_optional([summary.temperature_celsius.maximum for summary in summaries])
        ),
        median_graphics_clock_mhz=(
            _median_optional([summary.graphics_clock_mhz.mean for summary in summaries])
        ),
        median_cpu_utilization_percent=(
            _median_optional([summary.cpu_utilization_percent.mean for summary in summaries])
        ),
        median_estimated_gpu_energy_joules=(
            _median_optional([summary.estimated_gpu_energy_joules for summary in summaries])
        ),
    )
