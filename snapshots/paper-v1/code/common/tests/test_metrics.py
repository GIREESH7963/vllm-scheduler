"""Unit tests for parsing + aggregation that need no GPU (run with `pytest -q`)."""
from common.metrics import RequestRecord, _p50_p99, _acceptance_rate
from common.telemetry import MetricsSample, TelemetrySampler, parse_prometheus


def test_parse_prometheus_basic():
    text = """
# HELP vllm:num_requests_running Number of running requests.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="Q"} 3.0
vllm:gpu_cache_usage_perc{model_name="Q"} 0.42
vllm:spec_decode_num_accepted_tokens_total{model_name="Q"} 120
vllm:spec_decode_num_draft_tokens_total{model_name="Q"} 200
some_histogram_bucket{le="0.1"} 5
""".strip()
    m = parse_prometheus(text)
    assert m["vllm:num_requests_running"] == 3.0
    assert m["vllm:gpu_cache_usage_perc"] == 0.42
    assert "some_histogram_bucket" not in m  # buckets skipped


def test_parse_prometheus_sums_labelsets():
    text = "x_total{a=\"1\"} 2\nx_total{a=\"2\"} 3\n"
    assert parse_prometheus(text)["x_total"] == 5.0


def test_p50_p99():
    d = _p50_p99([float(i) for i in range(1, 101)])
    assert 49 <= d["p50"] <= 52
    assert d["p99"] >= 98


def test_acceptance_rate_from_counter_delta():
    s = TelemetrySampler(enable_dcgm=False)
    for acc, draft in [(100, 200), (160, 300)]:
        s.metrics_samples.append(
            MetricsSample(0, 0, 0, 0, 0, spec_accepted=acc, spec_draft=draft)
        )
    # (160-100)/(300-200) = 0.6
    assert abs(_acceptance_rate(s) - 0.6) < 1e-9


def test_acceptance_rate_none_when_absent():
    s = TelemetrySampler(enable_dcgm=False)
    s.metrics_samples.append(MetricsSample(0, 0, 0, 0, 0, spec_accepted=None, spec_draft=None))
    assert _acceptance_rate(s) is None


def test_request_record_fields():
    r = RequestRecord(ttft_ms=12.0, tpot_ms=8.0, output_tokens=50, success=True)
    assert r.success and r.output_tokens == 50
