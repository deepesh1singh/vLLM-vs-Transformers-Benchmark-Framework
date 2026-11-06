from llm_perf_lab.client.sse import parse_sse_line


def test_parse_sse_json_payload() -> None:
    result = parse_sse_line('data: {"choices": [{"delta": {"content": "Hello"}}]}')

    assert result == {
        "choices": [
            {
                "delta": {
                    "content": "Hello",
                }
            }
        ]
    }


def test_parse_sse_done() -> None:
    assert parse_sse_line("data: [DONE]") == "[DONE]"


def test_parse_sse_blank_line() -> None:
    assert parse_sse_line("") is None


def test_parse_sse_comment() -> None:
    assert parse_sse_line(": keep-alive") is None
