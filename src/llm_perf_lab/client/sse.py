from __future__ import annotations

import json
from typing import Any, Literal

type SSEPayload = dict[str, Any] | Literal["[DONE]"]


def parse_sse_line(line: str) -> SSEPayload | None:
    """
    Parse one Server-Sent Events line.

    Blank lines, comments, and non-data fields are ignored.
    """

    stripped = line.strip()

    if not stripped:
        return None

    if stripped.startswith(":"):
        return None

    if not stripped.startswith("data:"):
        return None

    data = stripped.removeprefix("data:").strip()

    if data == "[DONE]":
        return "[DONE]"

    payload = json.loads(data)

    if not isinstance(payload, dict):
        raise ValueError("SSE data payload must be a JSON object.")

    return payload
