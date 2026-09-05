# 1. Provider registry and explicit capabilities

Date: 2026-09-05

Status: accepted. Supersedes the "one configured OpenAI-compatible upstream"
scope statement in `docs/design/000-phase-1-streaming-proxy.md`.

## Context

Two accepted documents disagreed. The Phase 1 design document scoped V1 to "one
configured OpenAI-compatible upstream and one public model alias". The plan's
Phase 1 exit criteria require that "three providers work, including one local
option" and that "adding a fourth provider requires one adapter and one registry
entry".

The disagreement was not cosmetic. A gateway that has only ever spoken to
OpenAI-compatible upstreams cannot know which parts of its own design are a
provider abstraction and which are OpenAI's wire format wearing a domain-object
costume. The `Provider` port was written against exactly one implementation, so
nothing had tested the claim that it was provider-independent.

## Decision

Build the registry and a second wire protocol now, rather than deferring
multi-provider work to Phase 4 alongside routing and failover.

Three provider kinds are registered: `openai` and `ollama` (both served by the
OpenAI-compatible adapter, differing only in default base URL) and `anthropic`
(a native Messages API adapter). One kind is active per process, selected by
`GATEWAY_PROVIDER_KIND`; the single public model alias is unchanged.

Adding a fourth provider is one adapter module plus one `PROVIDERS` entry.
Nothing outside the registry branches on which upstream is configured.

Provider differences that the public schema cannot hide are declared as
`Capabilities` on the port and enforced by `ChatService` before any upstream
connection is opened, so a rejection is a gateway decision with an honest HTTP
status rather than a forwarded upstream error.

## Consequences

The port survived contact with a second protocol, which is the evidence the
design lacked. Writing the Anthropic adapter forced four real differences into
the open that a second OpenAI-compatible upstream would have hidden: system
prompts are a top-level parameter, `max_tokens` is mandatory, usage arrives in
two halves rather than one final chunk, and the stream terminates with a named
`message_stop` event rather than `data: [DONE]`.

`Capabilities` is deliberately small — two fields, both enforced. It is an
extension point, not a taxonomy, and a flag that nothing consumes should be
deleted rather than kept for symmetry.

Routing and failover across the registered providers remain Phase 4 work. This
decision makes several providers *reachable*; it does not make them
*simultaneously active*, and no retry or failover behaviour is implied.

The cost is a second wire protocol to maintain from Phase 1 onward, including
its share of contract tests, before any user has asked for it. That cost is
accepted because the alternative was declaring the port provider-independent on
the evidence of a single implementation.
