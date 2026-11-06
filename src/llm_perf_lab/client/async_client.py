from __future__ import annotations

import itertools
import time
from typing import Any

import httpx

from llm_perf_lab.client.models import (
    AsyncRequestResult,
    StreamEvent,
)
from llm_perf_lab.client.sse import parse_sse_line


def extract_usage(
    payload: dict[str, Any],
) -> tuple[int | None, int | None, int | None]:
    usage = payload.get("usage")

    if not isinstance(usage, dict):
        return None, None, None

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")

    return (
        prompt_tokens if isinstance(prompt_tokens, int) else None,
        completion_tokens if isinstance(completion_tokens, int) else None,
        total_tokens if isinstance(total_tokens, int) else None,
    )


def extract_stream_content(
    payload: dict[str, Any],
) -> tuple[str, tuple[int, ...]]:
    choices = payload.get("choices")

    if not isinstance(choices, list) or not choices:
        return "", ()

    first_choice = choices[0]

    if not isinstance(first_choice, dict):
        return "", ()

    delta = first_choice.get("delta")

    text = ""

    if isinstance(delta, dict):
        content = delta.get("content")

        if isinstance(content, str):
            text = content

    raw_token_ids = first_choice.get("token_ids")

    token_ids: tuple[int, ...] = ()

    if isinstance(raw_token_ids, list):
        token_ids = tuple(token_id for token_id in raw_token_ids if isinstance(token_id, int))

    return text, token_ids


async def generate_streaming_request(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    request_index: int,
    temperature: float = 0.0,
) -> AsyncRequestResult:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"

    request_payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "return_token_ids": True,
        "stream_options": {
            "include_usage": True,
        },
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    stream_events: list[StreamEvent] = []
    text_fragments: list[str] = []
    event_timestamps: list[float] = []

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    status_code: int | None = None

    start_time = time.perf_counter()

    try:
        async with client.stream(
            "POST",
            url,
            headers=headers,
            json=request_payload,
        ) as response:
            status_code = response.status_code
            response.raise_for_status()

            async for line in response.aiter_lines():
                parsed = parse_sse_line(line)

                if parsed is None:
                    continue

                if parsed == "[DONE]":
                    break

                usage = extract_usage(parsed)

                if usage != (None, None, None):
                    (
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                    ) = usage

                text, token_ids = extract_stream_content(parsed)

                # Ignore the initial empty role event.
                if not text and not token_ids:
                    continue

                timestamp_seconds = time.perf_counter() - start_time

                stream_events.append(
                    StreamEvent(
                        timestamp_seconds=timestamp_seconds,
                        text=text,
                        token_ids=token_ids,
                    )
                )
                event_timestamps.append(timestamp_seconds)

                if text:
                    text_fragments.append(text)

        latency_seconds = time.perf_counter() - start_time

        first_event_time = event_timestamps[0] if event_timestamps else None
        last_event_time = event_timestamps[-1] if event_timestamps else None

        decode_seconds = (
            last_event_time - first_event_time
            if (first_event_time is not None and last_event_time is not None)
            else None
        )

        post_stream_overhead_seconds = (
            latency_seconds - last_event_time if last_event_time is not None else None
        )

        inter_event_latencies = tuple(
            current - previous
            for previous, current in itertools.pairwise(
                event_timestamps,
            )
        )

        return AsyncRequestResult(
            request_index=request_index,
            success=True,
            status_code=status_code,
            generated_text="".join(text_fragments),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_seconds=latency_seconds,
            time_to_first_token_seconds=first_event_time,
            decode_seconds=decode_seconds,
            post_stream_overhead_seconds=(post_stream_overhead_seconds),
            stream_events=tuple(stream_events),
            inter_event_latencies_seconds=(inter_event_latencies),
            error=None,
        )

    except Exception as error:
        latency_seconds = time.perf_counter() - start_time

        return AsyncRequestResult(
            request_index=request_index,
            success=False,
            status_code=status_code,
            generated_text="",
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            latency_seconds=latency_seconds,
            time_to_first_token_seconds=None,
            decode_seconds=None,
            post_stream_overhead_seconds=None,
            stream_events=(),
            inter_event_latencies_seconds=(),
            error=f"{type(error).__name__}: {error}",
        )
