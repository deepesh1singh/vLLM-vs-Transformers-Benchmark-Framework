from __future__ import annotations

import argparse
import asyncio
import json

import httpx

from llm_perf_lab.client.async_client import (
    generate_streaming_request,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Send one measured streaming request to a vLLM server.")
    )

    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--api-key",
        default="local-vllm-key",
    )
    parser.add_argument(
        "--model",
        default="qwen25-3b",
    )
    parser.add_argument(
        "--prompt",
        required=True,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
    )

    return parser.parse_args()


async def main() -> None:
    arguments = parse_arguments()

    timeout = httpx.Timeout(
        connect=10.0,
        read=120.0,
        write=30.0,
        pool=10.0,
    )

    async with httpx.AsyncClient(
        timeout=timeout,
    ) as client:
        await wait_for_server(
            client=client,
            base_url=arguments.base_url,
            api_key=arguments.api_key,
            attempts=10,
        )

        result = await generate_streaming_request(
            client=client,
            base_url=arguments.base_url,
            api_key=arguments.api_key,
            model=arguments.model,
            prompt=arguments.prompt,
            max_tokens=arguments.max_tokens,
            request_index=1,
        )

    print(
        json.dumps(
            result.to_dict(),
            indent=2,
            ensure_ascii=False,
        )
    )


async def wait_for_server(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    attempts: int = 30,
    delay_seconds: float = 1.0,
) -> None:
    url = f"{base_url.rstrip('/')}/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    for _ in range(attempts):
        try:
            response = await client.get(
                url,
                headers=headers,
            )

            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass

        await asyncio.sleep(delay_seconds)

    raise RuntimeError(f"vLLM server is not ready at {base_url}.")


if __name__ == "__main__":
    asyncio.run(main())
