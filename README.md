# Inference Gateway

A deliberately small OpenAI-compatible gateway for text chat completions. It
proxies every request through one configured upstream while providing bounded
timeouts, cancellation propagation, structured logs, health checks, and
Prometheus metrics.

## Supported API

`POST /v1/chat/completions` accepts `model`, `messages`, `stream`, `temperature`,
`top_p`, `max_tokens`, and `stop`. Messages may use the `system`, `user`, and
`assistant` roles and string content. Unsupported fields are rejected with an
OpenAI-shaped HTTP 400 response; they are never silently ignored.

Operational endpoints are `GET /health/live`, `GET /health/ready`, and
`GET /metrics`.

## Providers

One provider kind is active per process, chosen with `GATEWAY_PROVIDER_KIND`.
Clients always send the stable public alias from `GATEWAY_PUBLIC_MODEL`; the
upstream model name never leaks into the API.

| Kind | Adapter | Default base URL |
| --- | --- | --- |
| `ollama` | OpenAI-compatible | `http://127.0.0.1:11434/v1` |
| `openai` | OpenAI-compatible | `https://api.openai.com/v1` |
| `anthropic` | Anthropic Messages | `https://api.anthropic.com/v1` |

Adding a provider is one adapter module plus one entry in
`src/gateway/adapters/providers/registry.py`.

Providers differ in ways the public schema cannot hide, so those differences are
declared as capabilities and enforced before any upstream connection is opened.
The Anthropic Messages API requires `max_tokens` — `GATEWAY_DEFAULT_MAX_TOKENS`
is applied when a caller omits it — and requires the first non-system message to
come from the user, so a conversation opening with an `assistant` turn is
rejected with an HTTP 400 rather than being silently reshaped.

## Run locally

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
cp .env.example .env
uv run uvicorn gateway.main:app --host 127.0.0.1 --port 8000
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local-dev-key")
response = client.chat.completions.create(
    model="gateway-model",
    messages=[{"role": "user", "content": "Say hello."}],
)
print(response.choices[0].message.content)
```

If `GATEWAY_API_KEY` is set, clients must send it as a bearer token. Do not
expose the service beyond localhost without setting this value and adding TLS at
the ingress or reverse proxy.

## Quality checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
```

## Development against the fake upstream

For deterministic development without a paid provider, run the controllable
upstream in another terminal and point the gateway at it. It serves both wire
protocols, so every provider kind can be driven through the same scenarios:

```bash
uv run uvicorn tests.fake_upstream.app:app --host 127.0.0.1 --port 9000

GATEWAY_PROVIDER_KIND=openai \
GATEWAY_UPSTREAM_BASE_URL=http://127.0.0.1:9000/v1 \
GATEWAY_UPSTREAM_MODEL=fake-success \
uv run uvicorn gateway.main:app --host 127.0.0.1 --port 8000
```

Other fake model names deliberately produce `fake-http-400`, `fake-http-401`,
`fake-http-429`, `fake-http-500`, `fake-malformed`, `fake-close-early`,
`fake-delay-first`, `fake-delay-between`, `fake-hang`, `fake-omit-usage`, and
`fake-ignore-cancel` behaviours.

## Load generation

```bash
uv run inference-gateway-benchmark --url http://127.0.0.1:8000 \
  --model gateway-model --requests 400 --rate 100 --burst-size 10 \
  --concurrency 64 --tenants 5 --stream \
  --long-share 0.3 --slow-consumer-share 0.2 --cancel-probability 0.05 --seed 1
```

Traffic is bursty rather than a constant concurrency level, and mixes short and
long generations, fast and slow consumers, client cancellation, and several
named tenants. Runs are reproducible from `--seed` plus the `configuration`
block echoed in the report. Tenants are labelled with an `x-tenant-id` header
that the gateway currently ignores; they exist so that later fairness work has a
baseline measured with the same traffic shape.

Cancellation cannot be measured against an upstream that responds instantly —
the response completes before the disconnect lands. Use `fake-delay-between`.

## Design records

- [V1 streaming proxy design](docs/design/000-phase-1-streaming-proxy.md)
- [Phase 1 notes](docs/design/001-phase-1-notes.md)
- [ADR 1: provider registry and explicit capabilities](docs/adr/0001-provider-registry-and-explicit-capabilities.md)
- [Phase 1 baseline benchmark](docs/benchmarks/000-phase-1-baseline.md)

## Scope

This version intentionally has no Redis, distributed scheduling, automatic
retries, failover, caching, UI, or Kubernetes resources. It runs as a single
process against a single active upstream. See the design records above for the
compatibility and failure contract.
