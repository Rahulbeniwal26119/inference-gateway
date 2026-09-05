# V1 streaming proxy design

Status: accepted for implementation. The scope statement below is superseded
in part by [ADR 0001](../adr/0001-provider-registry-and-explicit-capabilities.md),
which replaces the single OpenAI-compatible upstream with a provider registry.
See [Phase 1 notes](001-phase-1-notes.md) for the post-implementation review.

## Scope

V1 is a single-process gateway with one configured OpenAI-compatible upstream
and one public model alias. It implements only `POST /v1/chat/completions` for
text messages, plus liveness, readiness, and metrics endpoints. The fake
upstream is permanent test infrastructure.

The request fields are `model`, `messages`, `stream`, `temperature`, `top_p`,
`max_tokens`, and `stop`. Supported roles are `system`, `user`, and `assistant`.
All other fields and non-string message content receive HTTP 400. This explicit
boundary is safer than accepting parameters that an upstream might ignore.

## Chosen execution model

The application owns a `Provider` protocol returning an asynchronous stream of
typed events: `StreamStarted`, `TextDelta`, `UsageReported`, and
`StreamFinished`. The OpenAI HTTP schema is translated into domain objects at
the API boundary. The adapter translates domain objects to its upstream wire
format and translates SSE back into domain events.

Both API modes consume that event stream. Non-streaming responses accumulate
events; streaming responses translate each event to an OpenAI chunk. Separate
provider completion and streaming methods were rejected because their error,
usage, and translation behavior would drift.

## Stream commitment and failure semantics

The route awaits the first upstream domain event before constructing a
`StreamingResponse`. Connection, authentication, rate-limit, protocol, and
first-token failures can therefore return an honest HTTP error.

Once a chunk has been sent, the HTTP status cannot change. A later provider
failure is encoded as an SSE `error` event with an OpenAI-shaped error object,
the stream closes without `[DONE]`, and a failure metric records the phase.
The gateway never continues visible output with another model.

## Timeouts and cancellation

Four independent limits exist:

- connect timeout: TCP/TLS connection establishment;
- first-token timeout: response headers through first valid provider event;
- idle timeout: interval between subsequent provider events;
- total timeout: wall-clock lifetime of upstream streaming.

The provider generator owns the HTTP streaming context. Closing or cancelling
the generator exits that context and closes the upstream response. No queue or
background producer separates upstream reads from downstream writes, so client
backpressure is naturally bounded and disconnect cancellation propagates.

## Errors

Provider status codes and transport failures become stable gateway error
categories. Provider authentication is a gateway configuration failure and is
reported as 502 rather than forwarding a misleading 401 to the caller. Upstream
429 maps to 429, unavailable upstreams to 503, timeouts to 504, and malformed or
truncated protocols to 502.

Before commitment, errors use OpenAI-compatible JSON. After commitment, the
same error object is sent in an SSE `error` event. Upstream response bodies are
not exposed because they may contain provider or customer data.

## Logging and metrics

Every response carries an `x-request-id`; a safe caller-supplied ID is preserved
and otherwise a UUID is created. Structured logs contain request ID, public
model, provider, timing, and error category, but never messages or generated
text.

Metrics track completed requests, active requests, time to first text delta,
total duration, cancellations, and failures before versus during streaming.
Labels use configured provider and public model names to avoid customer-driven
cardinality.

## Security and deployment

An optional single bearer token protects all inference requests. It is not a
multi-tenant identity system. Deployments without the token must remain bound
to localhost. Secrets are environment configuration and never logged.

Readiness means configuration is internally valid; it deliberately does not
call the provider on every probe. V1 contains no retries: an automatic retry can
duplicate cost, and retry/failover policy requires explicit budgets in a later
version.
