from __future__ import annotations

from time import perf_counter

import pytest
import torch

from llm_perf_lab.baseline.token_streamer import (
    TokenEvent,
    TokenTimingStreamer,
)


def test_rejects_negative_start_time() -> None:
    with pytest.raises(
        ValueError,
        match="start_time cannot be negative",
    ):
        TokenTimingStreamer(start_time=-1.0)


@pytest.mark.parametrize("timeout_seconds", [0.0, -1.0])
def test_rejects_invalid_timeout(
    timeout_seconds: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="timeout_seconds must be greater than zero or None",
    ):
        TokenTimingStreamer(
            start_time=perf_counter(),
            timeout_seconds=timeout_seconds,
        )


def test_accepts_none_timeout() -> None:
    streamer = TokenTimingStreamer(
        start_time=perf_counter(),
        timeout_seconds=None,
    )

    assert streamer.timeout_seconds is None


def test_skips_prompt_by_default() -> None:
    streamer = TokenTimingStreamer(
        start_time=perf_counter(),
        timeout_seconds=0.1,
    )

    streamer.put(torch.tensor([[10, 20, 30]]))
    streamer.end()

    assert list(streamer) == []


def test_records_generated_token_after_prompt() -> None:
    streamer = TokenTimingStreamer(
        start_time=perf_counter(),
        timeout_seconds=0.1,
    )

    streamer.put(torch.tensor([[10, 20, 30]]))
    streamer.put(torch.tensor([40]))
    streamer.end()

    events = list(streamer)

    assert len(events) == 1
    assert events[0].token_id == 40
    assert events[0].elapsed_seconds >= 0


def test_records_multiple_generated_tokens() -> None:
    streamer = TokenTimingStreamer(
        start_time=perf_counter(),
        timeout_seconds=0.1,
    )

    streamer.put(torch.tensor([[10, 20]]))
    streamer.put(torch.tensor([30]))
    streamer.put(torch.tensor([40]))
    streamer.put(torch.tensor([50]))
    streamer.end()

    events = list(streamer)

    assert [event.token_id for event in events] == [30, 40, 50]
    assert all(event.elapsed_seconds >= 0 for event in events)


def test_records_all_tokens_from_one_tensor() -> None:
    streamer = TokenTimingStreamer(
        start_time=perf_counter(),
        skip_prompt=False,
        timeout_seconds=0.1,
    )

    streamer.put(torch.tensor([10, 20, 30]))
    streamer.end()

    events = list(streamer)

    assert [event.token_id for event in events] == [10, 20, 30]


def test_tokens_from_same_put_share_timestamp() -> None:
    streamer = TokenTimingStreamer(
        start_time=perf_counter(),
        skip_prompt=False,
        timeout_seconds=0.1,
    )

    streamer.put(torch.tensor([10, 20, 30]))
    streamer.end()

    events = list(streamer)

    timestamps = {event.elapsed_seconds for event in events}

    assert len(timestamps) == 1


def test_accepts_scalar_tensor() -> None:
    streamer = TokenTimingStreamer(
        start_time=perf_counter(),
        skip_prompt=False,
        timeout_seconds=0.1,
    )

    streamer.put(torch.tensor(42))
    streamer.end()

    events = list(streamer)

    assert len(events) == 1
    assert events[0].token_id == 42


def test_accepts_two_dimensional_batch_of_one() -> None:
    streamer = TokenTimingStreamer(
        start_time=perf_counter(),
        skip_prompt=False,
        timeout_seconds=0.1,
    )

    streamer.put(torch.tensor([[10, 20, 30]]))
    streamer.end()

    events = list(streamer)

    assert [event.token_id for event in events] == [10, 20, 30]


def test_rejects_batch_larger_than_one() -> None:
    streamer = TokenTimingStreamer(
        start_time=perf_counter(),
        skip_prompt=False,
    )

    with pytest.raises(
        ValueError,
        match="supports only batch size 1",
    ):
        streamer.put(
            torch.tensor(
                [
                    [10, 20],
                    [30, 40],
                ]
            )
        )


def test_rejects_tensor_with_more_than_two_dimensions() -> None:
    streamer = TokenTimingStreamer(
        start_time=perf_counter(),
        skip_prompt=False,
    )

    with pytest.raises(
        ValueError,
        match="must have at most two dimensions",
    ):
        streamer.put(torch.zeros((1, 1, 1), dtype=torch.long))


def test_rejects_non_tensor_value() -> None:
    streamer = TokenTimingStreamer(
        start_time=perf_counter(),
        skip_prompt=False,
    )

    with pytest.raises(
        TypeError,
        match=r"streamed value must be a torch.Tensor",
    ):
        streamer.put([10, 20])  # type: ignore[arg-type]


def test_end_stops_iteration() -> None:
    streamer = TokenTimingStreamer(
        start_time=perf_counter(),
        skip_prompt=False,
        timeout_seconds=0.1,
    )

    streamer.put(torch.tensor([10]))
    streamer.end()

    iterator = iter(streamer)

    event = next(iterator)

    assert isinstance(event, TokenEvent)
    assert event.token_id == 10

    with pytest.raises(StopIteration):
        next(iterator)


def test_iteration_times_out_without_token_or_end() -> None:
    streamer = TokenTimingStreamer(
        start_time=perf_counter(),
        timeout_seconds=0.01,
    )

    with pytest.raises(
        TimeoutError,
        match="No generated token was received before the timeout",
    ):
        next(streamer)


def test_events_preserve_arrival_order() -> None:
    streamer = TokenTimingStreamer(
        start_time=perf_counter(),
        skip_prompt=False,
        timeout_seconds=0.1,
    )

    expected_token_ids = [5, 3, 8, 1]

    for token_id in expected_token_ids:
        streamer.put(torch.tensor([token_id]))

    streamer.end()

    actual_token_ids = [event.token_id for event in streamer]

    assert actual_token_ids == expected_token_ids
