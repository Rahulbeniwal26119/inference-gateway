# Phase 1 notes

Date: 2026-09-05. Written after implementation, reviewing
`000-phase-1-streaming-proxy.md` against what the code taught.

## What the design got right

The commitment boundary was the most valuable decision in the document.
Prefetching the first upstream event before constructing the `StreamingResponse`
means connection, authentication, rate-limit, and first-token failures still
become honest HTTP status codes, and the rule "once a chunk is sent, the status
cannot change" turned out to be easy to hold because it was decided before any
route existed.

Making the provider generator own the HTTP streaming context, with no queue
between upstream reads and downstream writes, paid off twice. Client backpressure
is bounded for free — a slow consumer stalls its own stream and nothing else, as
the benchmark's slow-consumer cohort confirms — and cancellation propagates by
closing the generator, with no bookkeeping.

One event stream shared by both API modes held up. Non-streaming accumulates the
same events that streaming translates to chunks, so error, usage, and finish
handling cannot drift between them.

## Where the design was wrong or incomplete

**The scope statement was too narrow.** "One configured OpenAI-compatible
upstream" conflicted with the plan's exit criteria and, worse, would have left
the `Provider` port validated against a single implementation. Resolved by
[ADR 0001](../adr/0001-provider-registry-and-explicit-capabilities.md).

**Capabilities were described but not designed.** The plan asked for "capability
representation" and the design document did not carry one. It became necessary
the moment a non-OpenAI provider existed: Anthropic requires `max_tokens` and
rejects a conversation whose first non-system turn is not the user's. Both are
now declared on the port and enforced in `ChatService`.

**Token estimation was dropped and is still missing.** The plan's Step 3 put a
token-estimation operation on the provider port; the design document removed it,
and nothing replaced it. Reported usage is now recorded as
`gateway_upstream_tokens_total`, but that arrives *after* the request. Phase 2
needs a *pre-request* estimate to reserve budget before the cost is known, and
there is currently no mechanism for one. This is the largest piece of debt
carried out of Phase 1, and it is on Phase 2's critical path.

## What the second protocol taught

Writing the Anthropic adapter surfaced differences that a second
OpenAI-compatible upstream would have hidden:

- **Thinking blocks are on by default on current Claude models.** An adapter that
  forwarded every content-block delta would stream reasoning to clients as if it
  were the answer. The adapter filters on `delta.type == "text_delta"`, so only
  visible assistant text crosses the boundary.
- **Usage arrives in two halves** — input tokens on `message_start`, output
  tokens on `message_delta` — so `UsageReported` is emitted at `message_stop`,
  and only when both halves arrived.
- **Truncation detection is protocol-specific.** OpenAI-compatible streams end
  with `data: [DONE]`; Anthropic streams end with a `message_stop` event. Both
  adapters must verify their own terminator, or a truncated stream reads as a
  complete one.
- **Stop reasons do not all map.** `tool_use` and `pause_turn` have no honest
  OpenAI equivalent, so they raise a protocol error rather than being reported as
  `"stop"`. Claiming an answer is complete when the model was waiting to call a
  tool is exactly the silent corruption the plan's mission statement warns about.

## What the measurements taught

The gateway saturates at roughly 180 requests per second on this host, and the
saturated resource is **one CPU core** — 98% of a single core with fifteen idle.
It is a single-threaded event loop, so per-chunk SSE parsing and JSON encoding
are the binding constraint. Full numbers in
[the baseline report](../benchmarks/000-phase-1-baseline.md).

Two consequences. Scaling this is Phase 7's multi-process work, not tuning. And
any per-chunk work added before then spends from the budget that is already
binding, so quota checks and routing decisions in Phases 2 and 4 belong on the
request path, not the chunk path.

A measurement lesson too: cancellation cannot be observed against an upstream
that responds instantly, because the response completes before the disconnect
lands. Verified against a delayed upstream instead, where client-side and
gateway-side cancellation counts match exactly.

## Exit criteria

| Criterion | Status |
| --- | --- |
| Standard OpenAI clients call the endpoint unmodified | Met |
| Streaming and non-streaming share one internal event path | Met |
| Three providers work, including one local option | Met — `openai`, `ollama`, `anthropic` |
| A fourth provider needs one adapter and one registry entry | Met |
| Client cancellation stops upstream generation | Met — unit, integration, and measured end to end |
| A test kills an upstream after several chunks and verifies honest failure | Met |
| Metrics expose request count, active streams, TTFT, duration, failures | Met, plus tokens and upstream status classes |
| The first reproducible benchmark report exists | Met |

## Carried into Phase 2

1. **Pre-request token estimation** has no home on the port. Quotas cannot be
   enforced before cost is known without it.
2. **No retries, by design.** Phase 1 deliberately has none; retry and failover
   need explicit budgets and belong with routing in Phase 4.
3. **One active provider per process.** The registry makes several reachable;
   choosing between them at request time is Phase 4.
4. **Readiness does not probe the upstream.** It validates configuration only, as
   designed. If a later phase wants real dependency health, that is a new
   decision with a new cost.
5. **Tenancy exists only in the load generator.** The `x-tenant-id` header is
   generated and reported on, but the gateway ignores it entirely. The per-tenant
   baseline is a control case, not evidence of fairness.
