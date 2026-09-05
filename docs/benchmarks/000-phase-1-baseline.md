# Phase 1 baseline benchmark

Date: 2026-09-05. Gateway version 0.1.0.

## What was measured

A single gateway process proxying streaming chat completions to the controllable
fake upstream, driven by `inference-gateway-benchmark`. The fake upstream returns
five short chunks with no artificial delay, so the numbers below describe the
gateway's own overhead rather than any model's generation speed. That is the
point: this is the baseline that later phases must not silently regress.

The load generator is seeded, so a run is reproducible from its configuration
block alone.

## Environment

| Property | Value |
| --- | --- |
| Host | Linux 6.18.33.2-microsoft-standard-WSL2, 16 logical cores |
| Python | 3.12 |
| Gateway | one `uvicorn` worker, one event loop |
| Upstream | `tests.fake_upstream.app`, one `uvicorn` worker, same host |
| Client | `inference-gateway-benchmark`, same host |

Everything runs on one machine, so client and upstream CPU compete with the
gateway. The saturation point below is therefore a floor, not a published
capacity number.

## Reproducing

```bash
uv run uvicorn tests.fake_upstream.app:app --host 127.0.0.1 --port 9000 &
GATEWAY_PROVIDER_KIND=openai \
GATEWAY_UPSTREAM_BASE_URL=http://127.0.0.1:9000/v1 \
GATEWAY_UPSTREAM_MODEL=fake-success \
uv run uvicorn gateway.main:app --host 127.0.0.1 --port 8000 &

uv run inference-gateway-benchmark \
  --url http://127.0.0.1:8000 --model gateway-model \
  --requests 400 --rate "$RATE" --burst-size 10 --concurrency 64 \
  --tenants 5 --stream --long-share 0.3 --slow-consumer-share 0.2 \
  --cancel-probability 0.05 --seed 1
```

## Arrival-rate sweep

400 streaming requests per run, 5 tenants, 64 concurrent, 30% long generations,
20% deliberately slow consumers, 5% client cancellation, seed 1.

| Offered rate (req/s) | Achieved (req/s) | TTFT p50 | TTFT p95 | TTFT p99 | Duration p50 | Duration p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 24.0 | 26.6 ms | 46.2 ms | 56.2 ms | 31.7 ms | 944.8 ms |
| 50 | 45.3 | 24.3 ms | 31.8 ms | 46.7 ms | 28.0 ms | 939.8 ms |
| 100 | 80.9 | 25.7 ms | 36.5 ms | 55.3 ms | 29.8 ms | 942.3 ms |
| 200 | 134.0 | 25.5 ms | 44.0 ms | 61.6 ms | 29.5 ms | 942.1 ms |
| 400 | 181.8 | 158.5 ms | 229.3 ms | 238.8 ms | 180.4 ms | 1094.4 ms |
| 800 | 177.7 | 451.0 ms | 728.6 ms | 749.9 ms | 510.6 ms | 1448.2 ms |

## Where it turns upward, and why

Throughput climbs to roughly **180 requests per second and then stops**: raising
the offered rate from 400 to 800 buys no additional throughput (181.8 to 177.7)
while TTFT p50 nearly triples (158 ms to 451 ms). That is the signature of a
saturated server, not a slow one. Past the knee, added load becomes queue depth,
and queue depth becomes latency.

The p95 duration of roughly 940 ms at every rate below the knee is **not** a
latency problem. It is the 20% slow-consumer cohort, which sleeps 50 ms between
reads by construction. The per-consumer breakdown of the saturated run
separates them cleanly:

| Cohort | n | TTFT p50 | TTFT p95 | Duration p50 | Duration p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fast | 331 | 440.3 ms | 722.9 ms | 457.0 ms | 730.3 ms |
| slow | 69 | 468.6 ms | 740.1 ms | 1378.8 ms | 1654.3 ms |

Slow consumers pay in total duration and not in time to first token, which is the
expected shape when the client, not the gateway, sets the pace. Nothing is
buffered on their behalf: the gateway reads from the upstream only as fast as the
client drains it, so a slow reader stalls its own stream and no one else's.

### The saturated resource is one CPU core

Sampled from `/proc/<pid>/stat` during a saturating run:

| Process | CPU |
| --- | ---: |
| Gateway worker | **98.0% of one core** |
| Fake upstream worker | 20.5% of one core |
| Idle baseline (both) | 0.0% |

Fifteen of sixteen cores were idle. The gateway is a single-threaded event loop
in a single process, so its ceiling is one core's worth of Python executing SSE
parsing, JSON encoding, and per-chunk translation. Neither the upstream, the
network, nor the connection pool (`max_connections` 100, never approached) is the
constraint.

The consequence for later phases is that horizontal scale, not tuning, is the
lever here: the fix is more processes, which is Phase 7's subject. It also means
any per-chunk work added between now and then is spent from a budget that is
already the binding constraint.

## Per-tenant distribution

Five tenants sending identical traffic receive statistically identical service,
which is the expected — and unremarkable — result when no one misbehaves:

| Tenant | n | TTFT p50 | TTFT p95 | Duration p95 |
| --- | ---: | ---: | ---: | ---: |
| tenant-00 | 80 | 423.0 ms | 721.7 ms | 1405.8 ms |
| tenant-01 | 80 | 448.7 ms | 715.2 ms | 1428.4 ms |
| tenant-02 | 80 | 447.5 ms | 726.5 ms | 1412.1 ms |
| tenant-03 | 80 | 452.3 ms | 733.8 ms | 1452.4 ms |
| tenant-04 | 80 | 451.8 ms | 738.7 ms | 1474.4 ms |

This is the control case. It is worth recording precisely because Phase 3 has to
show the contrasting picture — one tenant misbehaving, the others unharmed — and
that claim is only meaningful against a measured "everyone is equal" baseline.
V1 has no fairness mechanism whatsoever; these tenants are equal because they
behave identically, not because anything protects them.

## A measurement artifact worth recording

The first sweep reported client-side cancellations but the gateway recorded
**zero**. That was not a bug. Against an upstream that returns five chunks with
no delay, the whole response is generated and written before the client's
disconnect can land, so "finished successfully" is the honest outcome.

Re-running against `fake-delay-between` (one second per chunk), so that streams
stay open long enough to be interrupted, gives an exact match:

| Signal | Value |
| --- | ---: |
| Client-side cancellations | 7 |
| `gateway_client_cancellations_total` | 7 |
| `gateway_requests_total{outcome="cancelled"}` | 7 |
| `gateway_active_requests` after the run | 0 |

Cancellation propagates through the real ASGI stack, and no request is left
active. The lesson for future benchmarks: cancellation behaviour cannot be
measured against an instantaneous upstream.

## Telemetry observed under load

Across 4,802 requests the gateway reported 19,208 prompt and 24,010 completion
tokens, 4,802 upstream `2xx` responses, and returned `gateway_active_requests` to
zero at rest. Time to first text delta is recorded separately from total
duration, which is what made the slow-consumer analysis above possible.

## Raw data

`benchmarks/phase-1-baseline-rate-100.json` (below the knee) and
`benchmarks/phase-1-baseline-rate-800.json` (saturated) hold the full reports,
including the `configuration` block needed to reproduce each run.
