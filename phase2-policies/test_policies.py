"""Phase-2 tests: policy ordering, adaptive feedback, length predictor. No GPU needed.

Run from repo root: `python -m pytest phase2-policies/test_policies.py -q`
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root (for `common`)
sys.path.insert(0, str(Path(__file__).resolve().parent))          # this dir (for scheduler/predictor)

from common.loadgen import RequestResult
from common.workload import RequestSpec
from scheduler import POLICIES, SchedRequest, SchedState, make_policy
from predictor import LengthPredictor, actuals_by_seq, oracle_size_fn


def _sr(seq, arrival, size, deadline, profile="short"):
    spec = RequestSpec(profile=profile, spec_decode=False, prompt="p", max_tokens=10,
                       prompt_words=int(size), seq=seq)
    return SchedRequest(spec=spec, arrival_t=arrival, seq=seq, size=size, deadline=deadline)


def _state():
    return SchedState(base_cap=4)


def test_registry_has_all_policies():
    assert set(POLICIES) == {"nocap", "fcfs", "srpt", "edf", "adaptive"}


def test_fcfs_picks_earliest_arrival():
    q = [_sr(0, 3.0, 100, 50), _sr(1, 1.0, 200, 40), _sr(2, 2.0, 10, 60)]
    pick = make_policy("fcfs").select(q, _state(), {}, now=5.0)
    assert pick.seq == 1 and len(q) == 2  # earliest arrival, removed from queue


def test_srpt_picks_smallest_size():
    q = [_sr(0, 3.0, 100, 50), _sr(1, 1.0, 200, 40), _sr(2, 2.0, 10, 60)]
    pick = make_policy("srpt").select(q, _state(), {}, now=5.0)
    assert pick.seq == 2  # smallest size


def test_edf_picks_earliest_deadline():
    q = [_sr(0, 3.0, 100, 50), _sr(1, 1.0, 200, 40), _sr(2, 2.0, 10, 60)]
    pick = make_policy("edf").select(q, _state(), {}, now=5.0)
    assert pick.seq == 1  # earliest deadline (40)


def test_nocap_admits_everything():
    assert make_policy("nocap").admission_cap(_state(), {}) >= 100_000


def test_adaptive_throttles_on_high_ttft_and_widens_on_idle_gpu():
    pol = make_policy("adaptive")
    st = SchedState(base_cap=6, min_cap=1, max_cap=16)
    # High recent TTFT p99 -> throttle (cap decreases).
    for _ in range(5):
        st.record_completion(RequestResult("short", False, 0, 0.0, 0.0, 5000.0, 5.0, 5000.0,
                                           10, 10, 10, True))
    cap_hi = pol.admission_cap(st, {"sm_active": 0.9, "kv_occupancy": 0.2})
    assert cap_hi < 6
    # Idle GPU + healthy TTFT -> widen.
    st2 = SchedState(base_cap=6, min_cap=1, max_cap=16)
    st2.record_completion(RequestResult("short", False, 0, 0.0, 0.0, 50.0, 5.0, 100.0,
                                        10, 10, 10, True))
    cap_lo = pol.admission_cap(st2, {"sm_active": 0.2, "kv_occupancy": 0.2})
    assert cap_lo > 6


def test_adaptive_deprioritizes_long_jobs_under_kv_pressure():
    pol = make_policy("adaptive")
    short = _sr(0, 1.0, 50, 100)
    long = _sr(1, 0.5, 400, 100)  # earlier arrival but large size
    snap = {"kv_occupancy": 0.95, "sm_active": 0.8}
    # Under KV pressure the long job is penalised, so the short (later) job wins despite later arrival.
    q = [short, long]
    pick = pol.select(q, _state(), snap, now=2.0)
    assert pick.seq == 0


def _rr(seq, profile, prompt_tok, out):
    return RequestResult(profile, False, seq, 0.0, 0.0, 10.0, 5.0, 100.0, out, out, prompt_tok, True)


def test_predictor_fits_and_oracle_lookup():
    recs = [_rr(i, "short", 30, 40) for i in range(5)] + [_rr(5 + i, "long", 600, 130) for i in range(5)]
    pred = LengthPredictor().fit(recs)
    # Class-aware: short ~ 40, long ~ 130.
    assert abs(pred.predict(30, "short") - 40) < 15
    assert abs(pred.predict(600, "long") - 130) < 20
    assert pred.mae(recs) < 20
    a = actuals_by_seq(recs)
    assert a[0] == 40 and a[5] == 130
    assert oracle_size_fn(a)(RequestSpec("long", False, "p", 10, 600, seq=5)) == 130
