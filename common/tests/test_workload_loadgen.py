"""Phase-1 tests: workload sampling reproducibility, Poisson schedule, aggregation. No GPU needed."""
import random

from common.config import Config
from common.loadgen import RequestResult, build_schedule
from common.metrics import aggregate_run, summarize_repeats
from common.telemetry import DcgmSample, MetricsSample, TelemetrySampler
from common.workload import DEFAULT_PROFILES, LenDist, WorkloadMix, sample_request


def test_lendist_reproducible_and_bounded():
    d = LenDist("lognormal", lo=10, hi=200, mu=4.0, sigma=0.5)
    a = [d.sample(random.Random(7)) for _ in range(3)]
    b = [d.sample(random.Random(7)) for _ in range(3)]
    assert a == b  # same seed -> same draws
    assert all(10 <= x <= 200 for x in a)


def test_sample_request_deterministic():
    p = DEFAULT_PROFILES["coding"]
    r1 = sample_request(p, random.Random(1), seq=0)
    r2 = sample_request(p, random.Random(1), seq=0)
    assert r1.prompt == r2.prompt and r1.max_tokens == r2.max_tokens
    assert r1.spec_decode is True  # coding is the spec-decode class


def test_build_schedule_reproducible_and_in_window():
    mix = WorkloadMix(DEFAULT_PROFILES, {"short": 0.5, "coding": 0.5})
    s1 = build_schedule(rate=5.0, duration_s=10.0, mix=mix, seed=42)
    s2 = build_schedule(rate=5.0, duration_s=10.0, mix=mix, seed=42)
    assert [a.t for a in s1] == [a.t for a in s2]
    assert all(0 < a.t <= 10.0 for a in s1)
    assert all(a.t < b.t for a, b in zip(s1, s1[1:]))  # strictly increasing
    # ~50 arrivals expected at λ=5 over 10s; allow wide slack for randomness.
    assert 20 < len(s1) < 90


def _cfg():
    return Config({
        "_config_name": "t",
        "model": "m",
        "workload": {"duration_s": 10, "warmup_s": 4},
        "slo": {"ttft_ms": 500, "tpot_ms": 50},
    })


def _mk_result(profile, arrival_t, ttft, tpot, out, warmup, ok=True, seq=0):
    return RequestResult(
        profile=profile, spec_decode=(profile == "coding"), seq=seq, arrival_t=arrival_t,
        dispatch_t=arrival_t + 0.001, ttft_ms=ttft, tpot_ms=tpot, latency_ms=ttft + tpot * out,
        output_tokens_requested=out, output_tokens_actual=out, prompt_tokens=20,
        success=ok, is_warmup=warmup,
    )


def test_aggregate_run_discards_warmup_and_breaks_down_by_class():
    results = [
        _mk_result("short", 1.0, 30, 10, 40, warmup=True),    # warmup -> discarded
        _mk_result("short", 5.0, 40, 12, 50, warmup=False),
        _mk_result("coding", 6.0, 60, 8, 100, warmup=False),
        _mk_result("coding", 7.0, 500000, 9, 0, warmup=False, ok=False),  # failure
    ]
    sampler = TelemetrySampler(enable_dcgm=False)
    # measured window starts at t0+warmup=104; samples with t>=104 count.
    sampler.metrics_samples += [
        MetricsSample(103, 1, 0, 0.1, 60, 100, 200),   # before window
        MetricsSample(105, 2, 1, 0.5, 70, 160, 320),
        MetricsSample(109, 2, 0, 0.4, 68, 220, 440),
    ]
    sampler.dcgm_samples += [DcgmSample(105, 0.6, 0.5), DcgmSample(109, 0.8, 0.6)]

    agg = aggregate_run(_cfg(), results, sampler, t0=100.0, rate=3.0, seed=1, repeat=0)
    assert agg["n_requests_measured"] == 2  # 2 ok, non-warmup
    assert agg["n_requests_failed"] == 1
    assert set(agg["by_class"]) == {"short", "coding"}
    assert agg["by_class"]["coding"]["spec_decode"] is True
    # acceptance from windowed counters: (220-160)/(440-320) = 0.5
    assert abs(agg["acceptance_rate"] - 0.5) < 1e-9
    # SM active mean over windowed dcgm = (0.6+0.8)/2 *100 = 70
    assert abs(agg["sm_active_pct"] - 70.0) < 1e-9
    assert agg["ttft_ms"]["p50"] > 0


def test_summarize_repeats_reports_mean_std():
    runs = [
        {"config": "t", "rate": 2.0, "throughput_tok_s": 100.0, "ttft_ms": {"p50": 50, "p99": 90},
         "tpot_ms": {"p50": 10, "p99": 12}, "sm_active_pct": 70, "acceptance_rate": 0.4,
         "slo_attainment": 1.0},
        {"config": "t", "rate": 2.0, "throughput_tok_s": 110.0, "ttft_ms": {"p50": 54, "p99": 95},
         "tpot_ms": {"p50": 11, "p99": 13}, "sm_active_pct": 72, "acceptance_rate": 0.42,
         "slo_attainment": 1.0},
    ]
    s = summarize_repeats(runs)
    assert s["repeats"] == 2
    assert abs(s["throughput_tok_s"]["mean"] - 105.0) < 1e-9
    assert s["throughput_tok_s"]["std"] > 0
