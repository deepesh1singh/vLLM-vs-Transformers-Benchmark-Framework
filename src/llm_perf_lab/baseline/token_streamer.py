from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from queue import Empty, Queue
from threading import Thread
from time import perf_counter
from typing import Any, Final

import torch
from transformers import PreTrainedTokenizerBase

from llm_perf_lab.types import CausalLMProtocol


@dataclass(frozen=True)
class TokenEvent:
    token_id: int
    elapsed_seconds: float


@dataclass(frozen=True)
class StreamingResult:
    generated_text: str
    generated_token_ids: tuple[int, ...]
    total_latency_seconds: float
    time_to_first_token_seconds: float
    decode_after_first_token_seconds: float
    token_events: tuple[TokenEvent, ...]
    inter_token_latencies_seconds: tuple[float, ...]

    @property
    def token_event_count(self) -> int:
        return len(self.token_events)


_STREAM_END: Final = object()


class TokenTimingStreamer:
    def __init__(
        self,
        *,
        start_time: float,
        skip_prompt: bool = True,
        timeout_seconds: float | None = 60.0,
    ) -> None:
        if start_time < 0:
            raise ValueError("start_time cannot be negative")

        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero or None")

        self.start_time = start_time
        self.skip_prompt = skip_prompt
        self.timeout_seconds = timeout_seconds

        self._prompt_received = False
        self._ended = False
        self._queue: Queue[TokenEvent | object] = Queue()

    def put(self, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError("streamed value must be a torch.Tensor")

        if value.ndim == 0:
            value = value.reshape(1)

        if value.ndim > 2:
            raise ValueError("streamed token tensor must have at most two dimensions")

        if value.ndim == 2:
            if value.shape[0] != 1:
                raise ValueError("TokenTimingStreamer supports only batch size 1")

            value = value[0]

        if self.skip_prompt and not self._prompt_received:
            self._prompt_received = True
            return

        self._prompt_received = True
        elapsed_seconds = perf_counter() - self.start_time

        for token_id in value.detach().cpu().tolist():
            self._queue.put(
                TokenEvent(
                    token_id=int(token_id),
                    elapsed_seconds=elapsed_seconds,
                )
            )

    def end(self) -> None:
        if not self._ended:
            self._ended = True
            self._queue.put(_STREAM_END)

    def __iter__(self) -> TokenTimingStreamer:
        return self

    def __next__(self) -> TokenEvent:
        try:
            item = self._queue.get(timeout=self.timeout_seconds)
        except Empty as error:
            raise TimeoutError("No generated token was received before the timeout") from error

        if item is _STREAM_END:
            raise StopIteration

        if not isinstance(item, TokenEvent):
            raise RuntimeError("Unexpected item received by token streamer")

        return item


def synchronize_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def generate_streaming(
    *,
    model: CausalLMProtocol,
    tokenizer: PreTrainedTokenizerBase,
    inputs: dict[str, torch.Tensor],
    device: str,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float = 1.0,
    timeout_seconds: float = 60.0,
) -> StreamingResult:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero")

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")

    if do_sample and temperature <= 0:
        raise ValueError("temperature must be greater than zero when do_sample=True")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    synchronize_device(device)
    start_time = perf_counter()

    streamer = TokenTimingStreamer(
        start_time=start_time,
        timeout_seconds=timeout_seconds,
    )

    generation_kwargs: dict[str, Any] = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if do_sample:
        generation_kwargs["temperature"] = temperature

    generation_errors: list[BaseException] = []

    def run_generation() -> None:
        try:
            with torch.inference_mode():
                model.generate(**generation_kwargs)
        except BaseException as error:
            generation_errors.append(error)
        finally:
            streamer.end()

    generation_thread = Thread(
        target=run_generation,
        daemon=True,
    )
    generation_thread.start()

    try:
        token_events = tuple(streamer)
    finally:
        generation_thread.join(timeout=timeout_seconds)

    if generation_thread.is_alive():
        raise TimeoutError("Generation did not finish within the configured timeout")

    if generation_errors:
        raise RuntimeError("Streaming generation failed") from generation_errors[0]

    if not token_events:
        raise RuntimeError("Generation completed without producing any tokens")

    synchronize_device(device)
    total_latency_seconds = perf_counter() - start_time

    generated_token_ids_list = [event.token_id for event in token_events]

    generated_text = tokenizer.decode(
        generated_token_ids_list,
        skip_special_tokens=True,
    )

    generated_token_ids = tuple(generated_token_ids_list)

    if not isinstance(generated_text, str):
        raise TypeError("Tokenizer decode() did not return a string")

    inter_token_latencies = tuple(
        current.elapsed_seconds - previous.elapsed_seconds
        for previous, current in pairwise(token_events)
    )

    first_token_time = token_events[0].elapsed_seconds
    decode_time = token_events[-1].elapsed_seconds - first_token_time

    return StreamingResult(
        generated_text=generated_text,
        generated_token_ids=tuple(generated_token_ids),
        total_latency_seconds=total_latency_seconds,
        time_to_first_token_seconds=first_token_time,
        decode_after_first_token_seconds=decode_time,
        token_events=token_events,
        inter_token_latencies_seconds=inter_token_latencies,
    )
