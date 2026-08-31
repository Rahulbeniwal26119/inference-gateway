"""Bursty load generator installed with the gateway package."""

import argparse
import asyncio
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(slots=True)
class Result:
    outcome: str
    status_code: int | None
    ttft_seconds: float | None
    duration_seconds: float


async def make_request(
    client: httpx.AsyncClient,
    *,
    url: str,
    model: str,
    streaming: bool,
    semaphore: asyncio.Semaphore,
    cancel_probability: float,
) -> Result:
    started = time.monotonic()
    ttft: float | None = None
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with five short words."}],
        "stream": streaming,
        "max_tokens": 32,
    }
    async with semaphore:
        try:
            if streaming:
                async with client.stream(
                    "POST", f"{url}/v1/chat/completions", json=payload
                ) as response:
                    if response.status_code >= 400:
                        await response.aread()
                        return Result(
                            "http_error", response.status_code, None, time.monotonic() - started
                        )
                    async for line in response.aiter_lines():
                        if line.startswith("data: ") and line != "data: [DONE]" and ttft is None:
                            ttft = time.monotonic() - started
                        if cancel_probability and random.random() < cancel_probability:
                            return Result(
                                "cancelled", response.status_code, ttft, time.monotonic() - started
                            )
                    return Result("success", response.status_code, ttft, time.monotonic() - started)

            response = await client.post(f"{url}/v1/chat/completions", json=payload)
            return Result(
                "success" if response.is_success else "http_error",
                response.status_code,
                None,
                time.monotonic() - started,
            )
        except httpx.HTTPError:
            return Result("transport_error", None, ttft, time.monotonic() - started)


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(results: list[Result], elapsed: float) -> dict[str, Any]:
    durations = [item.duration_seconds for item in results]
    ttfts = [item.ttft_seconds for item in results if item.ttft_seconds is not None]
    outcomes: dict[str, int] = {}
    for item in results:
        outcomes[item.outcome] = outcomes.get(item.outcome, 0) + 1
    return {
        "requests": len(results),
        "elapsed_seconds": elapsed,
        "throughput_requests_per_second": len(results) / elapsed if elapsed else None,
        "outcomes": outcomes,
        "duration_seconds": {
            "p50": percentile(durations, 0.50),
            "p95": percentile(durations, 0.95),
            "p99": percentile(durations, 0.99),
        },
        "ttft_seconds": {
            "p50": percentile(ttfts, 0.50),
            "p95": percentile(ttfts, 0.95),
            "p99": percentile(ttfts, 0.99),
        },
        "results": [asdict(item) for item in results],
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    headers = {"authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    limits = httpx.Limits(max_connections=args.concurrency)
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks: list[asyncio.Task[Result]] = []
    started = time.monotonic()
    async with httpx.AsyncClient(headers=headers, limits=limits, timeout=args.timeout) as client:
        for index in range(args.requests):
            tasks.append(
                asyncio.create_task(
                    make_request(
                        client,
                        url=args.url.rstrip("/"),
                        model=args.model,
                        streaming=args.stream,
                        semaphore=semaphore,
                        cancel_probability=args.cancel_probability,
                    )
                )
            )
            if (index + 1) % args.burst_size == 0 and index + 1 < args.requests:
                await asyncio.sleep(args.burst_size / args.rate)
        results = await asyncio.gather(*tasks)
    return summarize(results, time.monotonic() - started)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bursty traffic against the gateway.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="gateway-model")
    parser.add_argument("--api-key")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--burst-size", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=310.0)
    parser.add_argument("--cancel-probability", type=float, default=0.0)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.requests < 1 or args.rate <= 0 or args.burst_size < 1 or args.concurrency < 1:
        raise SystemExit("requests, rate, burst-size, and concurrency must be positive")
    if not 0 <= args.cancel_probability <= 1:
        raise SystemExit("cancel-probability must be between 0 and 1")
    report = asyncio.run(run(args))
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
