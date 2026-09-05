"""Bursty, multi-tenant load generator installed with the gateway package.

Tenants exist here before the gateway has any concept of tenancy: fairness work
in a later phase needs a baseline measured with the same traffic shape, and a
generator that only produces one anonymous stream of identical requests cannot
provide one.
"""

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

SHORT_PROMPT = "Reply with five short words."
LONG_PROMPT = "Explain how a streaming proxy applies backpressure, in several sentences."


@dataclass(slots=True)
class Result:
    tenant: str
    shape: str
    consumer: str
    outcome: str
    status_code: int | None
    ttft_seconds: float | None
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class RequestPlan:
    """One scheduled request. Built up front so a run is reproducible from a seed."""

    tenant: str
    shape: str
    consumer: str
    max_tokens: int
    prompt: str
    cancel: bool
    slow_delay: float


def build_plan(args: argparse.Namespace) -> list[RequestPlan]:
    rng = random.Random(args.seed)
    plans: list[RequestPlan] = []
    for index in range(args.requests):
        long_shape = rng.random() < args.long_share
        slow = args.stream and rng.random() < args.slow_consumer_share
        plans.append(
            RequestPlan(
                tenant=f"tenant-{index % args.tenants:02d}",
                shape="long" if long_shape else "short",
                consumer="slow" if slow else "fast",
                max_tokens=args.long_max_tokens if long_shape else args.short_max_tokens,
                prompt=LONG_PROMPT if long_shape else SHORT_PROMPT,
                cancel=args.stream and rng.random() < args.cancel_probability,
                slow_delay=args.slow_consumer_delay if slow else 0.0,
            )
        )
    return plans


async def make_request(
    client: httpx.AsyncClient,
    plan: RequestPlan,
    *,
    url: str,
    model: str,
    streaming: bool,
    semaphore: asyncio.Semaphore,
) -> Result:
    started = time.monotonic()
    ttft: float | None = None
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": plan.prompt}],
        "stream": streaming,
        "max_tokens": plan.max_tokens,
    }
    headers = {"x-tenant-id": plan.tenant}

    def result(outcome: str, status_code: int | None) -> Result:
        return Result(
            tenant=plan.tenant,
            shape=plan.shape,
            consumer=plan.consumer,
            outcome=outcome,
            status_code=status_code,
            ttft_seconds=ttft,
            duration_seconds=time.monotonic() - started,
        )

    async with semaphore:
        try:
            if streaming:
                async with client.stream(
                    "POST", f"{url}/v1/chat/completions", json=payload, headers=headers
                ) as response:
                    if response.status_code >= 400:
                        await response.aread()
                        return result("http_error", response.status_code)
                    saw_error_event = False
                    async for line in response.aiter_lines():
                        if line.startswith("event: error"):
                            saw_error_event = True
                        if line.startswith("data: ") and line != "data: [DONE]" and ttft is None:
                            ttft = time.monotonic() - started
                        if plan.cancel and ttft is not None:
                            return result("cancelled", response.status_code)
                        if plan.slow_delay:
                            await asyncio.sleep(plan.slow_delay)
                    outcome = "stream_error" if saw_error_event else "success"
                    return result(outcome, response.status_code)

            response = await client.post(
                f"{url}/v1/chat/completions", json=payload, headers=headers
            )
            return result("success" if response.is_success else "http_error", response.status_code)
        except httpx.HTTPError:
            return result("transport_error", None)


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


def latency_summary(results: list[Result]) -> dict[str, Any]:
    durations = [item.duration_seconds for item in results]
    ttfts = [item.ttft_seconds for item in results if item.ttft_seconds is not None]
    outcomes: dict[str, int] = {}
    for item in results:
        outcomes[item.outcome] = outcomes.get(item.outcome, 0) + 1
    return {
        "requests": len(results),
        "outcomes": dict(sorted(outcomes.items())),
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
    }


def group_by(results: list[Result], key: str) -> dict[str, Any]:
    groups: dict[str, list[Result]] = {}
    for item in results:
        groups.setdefault(str(getattr(item, key)), []).append(item)
    return {name: latency_summary(items) for name, items in sorted(groups.items())}


def summarize(
    results: list[Result], elapsed: float, args: argparse.Namespace, include_raw: bool
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "configuration": {
            "requests": args.requests,
            "rate_per_second": args.rate,
            "burst_size": args.burst_size,
            "concurrency": args.concurrency,
            "tenants": args.tenants,
            "stream": args.stream,
            "long_share": args.long_share,
            "slow_consumer_share": args.slow_consumer_share,
            "slow_consumer_delay": args.slow_consumer_delay,
            "cancel_probability": args.cancel_probability,
            "seed": args.seed,
        },
        "elapsed_seconds": elapsed,
        "throughput_requests_per_second": len(results) / elapsed if elapsed else None,
        "overall": latency_summary(results),
        "per_tenant": group_by(results, "tenant"),
        "per_shape": group_by(results, "shape"),
        "per_consumer": group_by(results, "consumer"),
    }
    if include_raw:
        report["results"] = [asdict(item) for item in results]
    return report


async def run(args: argparse.Namespace) -> dict[str, Any]:
    headers = {"authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    limits = httpx.Limits(max_connections=args.concurrency)
    semaphore = asyncio.Semaphore(args.concurrency)
    plans = build_plan(args)
    tasks: list[asyncio.Task[Result]] = []
    started = time.monotonic()

    async with httpx.AsyncClient(headers=headers, limits=limits, timeout=args.timeout) as client:
        for index, plan in enumerate(plans):
            tasks.append(
                asyncio.create_task(
                    make_request(
                        client,
                        plan,
                        url=args.url.rstrip("/"),
                        model=args.model,
                        streaming=args.stream,
                        semaphore=semaphore,
                    )
                )
            )
            if (index + 1) % args.burst_size == 0 and index + 1 < args.requests:
                await asyncio.sleep(args.burst_size / args.rate)
        results = await asyncio.gather(*tasks)

    return summarize(results, time.monotonic() - started, args, args.include_raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bursty traffic against the gateway.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="gateway-model")
    parser.add_argument("--api-key")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--burst-size", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--tenants", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=310.0)
    parser.add_argument("--cancel-probability", type=float, default=0.0)
    parser.add_argument("--long-share", type=float, default=0.0)
    parser.add_argument("--short-max-tokens", type=int, default=32)
    parser.add_argument("--long-max-tokens", type=int, default=256)
    parser.add_argument("--slow-consumer-share", type=float, default=0.0)
    parser.add_argument("--slow-consumer-delay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--include-raw", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.requests < 1 or args.rate <= 0 or args.burst_size < 1 or args.concurrency < 1:
        raise SystemExit("requests, rate, burst-size, and concurrency must be positive")
    if args.tenants < 1:
        raise SystemExit("tenants must be positive")
    for name in ("cancel_probability", "long_share", "slow_consumer_share"):
        if not 0 <= getattr(args, name) <= 1:
            raise SystemExit(f"{name.replace('_', '-')} must be between 0 and 1")

    report = asyncio.run(run(args))
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
