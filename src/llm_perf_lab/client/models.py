from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StreamEvent:
    timestamp_seconds: float
    text: str
    token_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AsyncRequestResult:
    request_index: int
    success: bool
    status_code: int | None
    generated_text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_seconds: float
    time_to_first_token_seconds: float | None
    decode_seconds: float | None
    post_stream_overhead_seconds: float | None
    stream_events: tuple[StreamEvent, ...]
    inter_event_latencies_seconds: tuple[float, ...]
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "stream_events": [event.to_dict() for event in self.stream_events],
            "inter_event_latencies_seconds": list(self.inter_event_latencies_seconds),
        }
