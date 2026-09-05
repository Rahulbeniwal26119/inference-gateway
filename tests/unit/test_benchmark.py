import argparse

from gateway.benchmark import Result, build_plan, group_by, latency_summary, percentile


def args(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "requests": 12,
        "tenants": 3,
        "stream": True,
        "long_share": 0.5,
        "slow_consumer_share": 0.5,
        "slow_consumer_delay": 0.05,
        "cancel_probability": 0.0,
        "short_max_tokens": 32,
        "long_max_tokens": 256,
        "seed": 7,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def result(tenant: str, duration: float, ttft: float | None = None) -> Result:
    return Result(
        tenant=tenant,
        shape="short",
        consumer="fast",
        outcome="success",
        status_code=200,
        ttft_seconds=ttft,
        duration_seconds=duration,
    )


def test_requests_are_distributed_evenly_across_named_tenants() -> None:
    plans = build_plan(args())

    counts: dict[str, int] = {}
    for plan in plans:
        counts[plan.tenant] = counts.get(plan.tenant, 0) + 1

    assert sorted(counts) == ["tenant-00", "tenant-01", "tenant-02"]
    assert set(counts.values()) == {4}


def test_a_seed_makes_the_traffic_shape_reproducible() -> None:
    assert build_plan(args()) == build_plan(args())
    assert build_plan(args(seed=8)) != build_plan(args())


def test_long_requests_carry_a_larger_token_budget() -> None:
    plans = build_plan(args(long_share=1.0))

    assert {plan.shape for plan in plans} == {"long"}
    assert {plan.max_tokens for plan in plans} == {256}


def test_slow_consumers_only_exist_for_streaming_runs() -> None:
    assert all(plan.consumer == "fast" for plan in build_plan(args(stream=False)))
    assert any(plan.consumer == "slow" for plan in build_plan(args(slow_consumer_share=1.0)))


def test_percentiles_interpolate_between_samples() -> None:
    assert percentile([], 0.5) is None
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentile([1.0], 0.99) == 1.0


def test_summary_separates_ttft_from_total_duration() -> None:
    summary = latency_summary([result("a", 1.0, ttft=0.1), result("a", 3.0)])

    assert summary["requests"] == 2
    assert summary["outcomes"] == {"success": 2}
    assert summary["duration_seconds"]["p50"] == 2.0
    # Only the streamed request contributed a first-token sample.
    assert summary["ttft_seconds"]["p50"] == 0.1


def test_per_tenant_grouping_keeps_tenants_separate() -> None:
    grouped = group_by([result("a", 1.0), result("b", 5.0), result("b", 7.0)], "tenant")

    assert grouped["a"]["requests"] == 1
    assert grouped["b"]["requests"] == 2
    assert grouped["b"]["duration_seconds"]["p50"] == 6.0
