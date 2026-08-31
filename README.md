# Inference Gateway

A deliberately small OpenAI-compatible gateway for text chat completions. V1
proxies every request through one configured OpenAI-compatible upstream while
providing bounded timeouts, cancellation propagation, structured logs, health
checks, and Prometheus metrics.

## Supported API

`POST /v1/chat/completions` accepts `model`, `messages`, `stream`, `temperature`,
`top_p`, `max_tokens`, and `stop`. Messages may use the `system`, `user`, and
`assistant` roles and string content. Unsupported fields are rejected with an
OpenAI-shaped HTTP 400 response; they are never silently ignored.

Operational endpoints are `GET /health/live`, `GET /health/ready`, and
`GET /metrics`.

## Run locally

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
cp .env.example .env
uv run uvicorn gateway.main:app --host 127.0.0.1 --port 8000
```

The default configuration targets a local OpenAI-compatible server at
`http://127.0.0.1:11434/v1`. Set `GATEWAY_UPSTREAM_BASE_URL`,
`GATEWAY_UPSTREAM_API_KEY`, and `GATEWAY_UPSTREAM_MODEL` for another provider.
Clients send the stable public model name from `GATEWAY_PUBLIC_MODEL`.

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

Run the lightweight load generator against a live gateway with:

```bash
uv run inference-gateway-benchmark --url http://127.0.0.1:8000 \
  --model gateway-model --requests 100 --rate 10 --stream
```

For deterministic development without a paid provider, run the controllable
upstream in another terminal and point the gateway at it:

```bash
uv run uvicorn tests.fake_upstream.app:app --host 127.0.0.1 --port 9000
GATEWAY_UPSTREAM_BASE_URL=http://127.0.0.1:9000/v1 \
GATEWAY_UPSTREAM_MODEL=fake-success \
uv run uvicorn gateway.main:app --host 127.0.0.1 --port 8000
```

Other fake model names deliberately produce `fake-http-429`, `fake-malformed`,
`fake-close-early`, `fake-delay-first`, `fake-delay-between`, `fake-hang`, and
`fake-ignore-cancel` behaviors.

V1 intentionally has no Redis, distributed scheduling, automatic retries,
failover, caching, UI, or Kubernetes resources. See
`docs/design/000-phase-1-streaming-proxy.md` for the compatibility and failure
contract.
