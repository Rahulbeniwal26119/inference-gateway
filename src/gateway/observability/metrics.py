import time

import structlog
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from gateway.domain.errors import GatewayError
from gateway.domain.events import ProviderEvent, TextDelta, UsageReported


class GatewayMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self.requests = Counter(
            "gateway_requests_total",
            "Completed gateway inference requests.",
            ("provider", "model", "stream", "outcome"),
            registry=self.registry,
        )
        self.active = Gauge(
            "gateway_active_requests",
            "Currently active gateway inference requests.",
            ("provider", "model", "stream"),
            registry=self.registry,
        )
        self.ttft = Histogram(
            "gateway_time_to_first_token_seconds",
            "Time from admission to the first text delta.",
            ("provider", "model", "stream"),
            registry=self.registry,
        )
        self.duration = Histogram(
            "gateway_request_duration_seconds",
            "Total gateway request duration.",
            ("provider", "model", "stream", "outcome"),
            registry=self.registry,
        )
        self.failures = Counter(
            "gateway_failures_total",
            "Gateway failures by commitment phase and stable category.",
            ("provider", "model", "phase", "code"),
            registry=self.registry,
        )
        self.cancellations = Counter(
            "gateway_client_cancellations_total",
            "Inference requests cancelled by a disconnected client.",
            ("provider", "model", "stream"),
            registry=self.registry,
        )
        self.tokens = Counter(
            "gateway_upstream_tokens_total",
            "Tokens reported by the upstream provider.",
            ("provider", "model", "direction"),
            registry=self.registry,
        )
        self.upstream_responses = Counter(
            "gateway_upstream_responses_total",
            "Upstream outcomes by HTTP status class, or 'transport' when no response arrived.",
            ("provider", "model", "status_class"),
            registry=self.registry,
        )

    def begin(self, *, provider: str, model: str, streaming: bool) -> "RequestObserver":
        return RequestObserver(self, provider=provider, model=model, streaming=streaming)


class RequestObserver:
    def __init__(
        self,
        metrics: GatewayMetrics,
        *,
        provider: str,
        model: str,
        streaming: bool,
    ) -> None:
        self.metrics = metrics
        self.provider = provider
        self.model = model
        self.stream = "true" if streaming else "false"
        self.started = time.monotonic()
        self.finished = False
        self.saw_text = False
        self.usage: tuple[int, int] | None = None
        self._active_labels = (self.provider, self.model, self.stream)
        self.metrics.active.labels(*self._active_labels).inc()
        self.log = structlog.get_logger().bind(
            provider=self.provider,
            model=self.model,
            stream=streaming,
        )
        self.log.info("inference_started")

    def observe(self, event: ProviderEvent) -> None:
        if isinstance(event, TextDelta) and not self.saw_text:
            self.saw_text = True
            self.metrics.ttft.labels(*self._active_labels).observe(time.monotonic() - self.started)
        elif isinstance(event, UsageReported):
            # Recorded, not accumulated: an upstream that repeats cumulative
            # usage on every chunk would multiply the token counters. The
            # counters are incremented once, from the last report, at the end.
            usage = event.usage
            self.usage = (usage.prompt_tokens, usage.completion_tokens)

    def succeed(self) -> None:
        self._record_upstream_response("2xx")
        self._finish("success")

    def fail(self, error: GatewayError, phase: str) -> None:
        if self.finished:
            return
        self.metrics.failures.labels(self.provider, self.model, phase, error.code).inc()
        self._record_upstream_response(_status_class(error.upstream_status))
        self.log.warning(
            "inference_failed", error_code=error.code, phase=phase, detail=error.detail
        )
        self._finish("failure")

    def cancel(self) -> None:
        if self.finished:
            return
        self.metrics.cancellations.labels(*self._active_labels).inc()
        self.log.info("inference_cancelled")
        self._finish("cancelled")

    def _record_upstream_response(self, status_class: str) -> None:
        self.metrics.upstream_responses.labels(self.provider, self.model, status_class).inc()

    def _finish(self, outcome: str) -> None:
        if self.finished:
            return
        self.finished = True
        duration = time.monotonic() - self.started
        if self.usage is not None:
            prompt_tokens, completion_tokens = self.usage
            self.metrics.tokens.labels(self.provider, self.model, "prompt").inc(prompt_tokens)
            self.metrics.tokens.labels(self.provider, self.model, "completion").inc(
                completion_tokens
            )
        self.metrics.active.labels(*self._active_labels).dec()
        self.metrics.requests.labels(self.provider, self.model, self.stream, outcome).inc()
        self.metrics.duration.labels(self.provider, self.model, self.stream, outcome).observe(
            duration
        )
        if outcome == "success":
            self.log.info(
                "inference_completed",
                duration_seconds=duration,
                prompt_tokens=self.usage[0] if self.usage else None,
                completion_tokens=self.usage[1] if self.usage else None,
            )


def _status_class(upstream_status: int | None) -> str:
    """Bucket upstream statuses. 'transport' means no HTTP response arrived at all."""
    if upstream_status is None:
        return "transport"
    return f"{upstream_status // 100}xx"
