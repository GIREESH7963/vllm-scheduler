"""Admission / ordering scheduler that sits IN FRONT of vLLM.

**This layer does NOT modify vLLM internals.** It controls only (a) *which* queued requests are
admitted to the single vLLM instance and (b) *in what order*, subject to an admission concurrency
cap (max in-flight). Intra-engine batch formation, KV management, and the continuous-batching
scheduler are entirely vLLM's — untouched (root rule). The cap is what makes ordering matter: it
keeps per-token latency low (avoids the 6x TPOT collapse measured in Phase 1) while the queue
absorbs overload, turning queue-wait into the tunable tradeoff.

Open-loop arrivals (Poisson, from `common.loadgen.build_schedule`) enqueue at their arrival time;
an admitter admits up to `cap` at once, picking the next by the active `Policy`. Multi-tenancy is
modelled as multiple request *streams/classes* into ONE engine — never multiple engines (16 GB T4).
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field

import httpx
import numpy as np

from common.loadgen import RequestResult, _send
from common.workload import RequestSpec, WorkloadMix
from common.loadgen import build_schedule


# --- request wrapper + feedback state ---------------------------------------------------


@dataclass
class SchedRequest:
    spec: RequestSpec
    arrival_t: float           # seconds since run start (scheduled)
    seq: int
    size: float                # SRPT size estimate (prompt-proxy / predicted / oracle)
    deadline: float            # EDF absolute deadline (seconds since run start)
    admit_t: float = 0.0


@dataclass
class SchedState:
    base_cap: int
    min_cap: int = 1
    max_cap: int = 16
    current_cap: int = 0
    inflight: int = 0
    _recent_ttft: deque = field(default_factory=lambda: deque(maxlen=64))

    def __post_init__(self):
        if self.current_cap == 0:
            self.current_cap = self.base_cap

    def record_completion(self, rr: RequestResult) -> None:
        if not (rr.ttft_ms != rr.ttft_ms):  # not NaN
            self._recent_ttft.append(rr.ttft_ms)

    def recent_ttft_p99(self) -> float:
        return float(np.percentile(self._recent_ttft, 99)) if self._recent_ttft else float("nan")


# --- policy interface + concrete policies -----------------------------------------------


class Policy:
    """Strategy interface. Lower ``priority`` = admitted sooner. ``admission_cap`` may vary."""

    name = "base"

    def admission_cap(self, state: SchedState, snap: dict) -> int:
        return state.base_cap

    def priority(self, req: SchedRequest, state: SchedState, snap: dict, now: float) -> float:
        raise NotImplementedError

    def select(self, queue: list[SchedRequest], state: SchedState, snap: dict, now: float) -> SchedRequest:
        idx = min(range(len(queue)), key=lambda i: self.priority(queue[i], state, snap, now))
        return queue.pop(idx)


class FCFS(Policy):
    name = "fcfs"

    def priority(self, req, state, snap, now):
        return req.arrival_t  # first-come first-served


class NoCap(FCFS):
    """No admission control: admit everything immediately (reproduces the Phase-1 no-scheduler
    baseline). Included in the matrix so plots show what the admission cap actually buys."""

    name = "nocap"

    def admission_cap(self, state, snap):
        return 100_000


class SRPT(Policy):
    """Shortest-remaining/size first. Size source is pluggable (prompt-proxy / prediction / oracle)."""

    def __init__(self, name: str = "srpt"):
        self.name = name

    def priority(self, req, state, snap, now):
        return req.size  # smallest job first


class EDF(Policy):
    name = "edf"

    def priority(self, req, state, snap, now):
        return req.deadline  # earliest deadline first


class Adaptive(Policy):
    """Rule-based feedback (NOT ML): widen cap when the GPU is idle, throttle when TTFT p99 rises,
    deprioritize long jobs when KV is under pressure."""

    name = "adaptive"

    def __init__(self, sm_low=0.55, ttft_throttle_ms=1500.0, kv_pressure=0.85, long_size=250.0,
                 long_penalty=1e6):
        self.sm_low = sm_low
        self.ttft_throttle_ms = ttft_throttle_ms
        self.kv_pressure = kv_pressure
        self.long_size = long_size
        self.long_penalty = long_penalty

    def admission_cap(self, state, snap):
        cap = state.current_cap
        p99 = state.recent_ttft_p99()
        sm = snap.get("sm_active", float("nan"))
        # Throttle first (safety): rising tail latency -> shrink cap.
        if p99 == p99 and p99 > self.ttft_throttle_ms:
            cap = max(state.min_cap, cap - 1)
        # Widen only if the GPU is demonstrably underused and tail latency is healthy.
        elif sm == sm and sm < self.sm_low:
            cap = min(state.max_cap, cap + 1)
        state.current_cap = cap
        return cap

    def priority(self, req, state, snap, now):
        base = req.arrival_t
        kv = snap.get("kv_occupancy", 0.0)
        if kv == kv and kv > self.kv_pressure and req.size >= self.long_size:
            return base + self.long_penalty  # push long jobs back under KV pressure
        return base


POLICIES = {"nocap": NoCap, "fcfs": FCFS, "srpt": SRPT, "edf": EDF, "adaptive": Adaptive}


def make_policy(name: str) -> Policy:
    if name not in POLICIES:
        raise KeyError(f"unknown policy {name!r}; have {list(POLICIES)}")
    return POLICIES[name]()


# --- size / deadline estimators ---------------------------------------------------------


def prompt_proxy_size(spec: RequestSpec) -> float:
    """Default SRPT size = prompt length proxy (known at arrival)."""
    return float(spec.prompt_words)


def make_deadline_fn(cfg):
    """Deadline = arrival + (ttft_slo + expected_output * tpot_slo). Class-aware via profile means,
    so short classes get tighter deadlines than long ones (makes EDF meaningful)."""
    slo = cfg.get("slo", {}) or {}
    ttft_s = float(slo.get("ttft_ms", 1000)) / 1000.0
    tpot_s = float(slo.get("tpot_ms", 75)) / 1000.0
    from common.workload import load_profiles

    profiles = load_profiles(cfg)

    def expected_out(name: str) -> float:
        p = profiles.get(name)
        if not p:
            return 128.0
        d = p.output_len
        return float(d.lo) if d.kind == "fixed" else (float(d.lo) + float(d.hi)) / 2.0

    def deadline_fn(spec: RequestSpec, arrival_t: float) -> float:
        return arrival_t + ttft_s + expected_out(spec.profile) * tpot_s

    return deadline_fn


# --- scheduled open-loop driver ---------------------------------------------------------


async def run_scheduled_load(cfg, rate: float, seed: int, policy: Policy, sampler,
                             size_fn=prompt_proxy_size) -> tuple[list[RequestResult], float]:
    """Open-loop arrivals -> scheduler queue -> admit <=cap by policy -> vLLM. Returns (results, t0)."""
    wl = cfg.get("workload", {}) or {}
    duration_s = float(wl.get("duration_s", 120))
    warmup_s = float(wl.get("warmup_s", 30))
    base_url = cfg["server"]["base_url"]
    model = cfg["model"]
    sched_cfg = cfg.get("scheduler", {}) or {}
    state = SchedState(
        base_cap=int(sched_cfg.get("max_in_flight", 4)),
        min_cap=int(sched_cfg.get("min_in_flight", 1)),
        max_cap=int(sched_cfg.get("max_cap", 16)),
    )
    # Load shedding: drop a request that has waited longer than this in the queue (counts as an SLO
    # miss). Bounds the queue/drain under overload AND differentiates policies (which jobs get shed).
    max_wait_s = sched_cfg.get("max_queue_wait_s")
    max_wait_s = float(max_wait_s) if max_wait_s is not None else None
    deadline_fn = make_deadline_fn(cfg)

    schedule = build_schedule(rate, duration_s, WorkloadMix.from_cfg(cfg), seed)
    queue: list[SchedRequest] = []
    results: list[RequestResult] = []
    wake = asyncio.Event()
    feeder_done = False

    limits = httpx.Limits(max_connections=1024, max_keepalive_connections=256)
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0), limits=limits) as client:
        t0 = time.monotonic()

        def on_done(sr: SchedRequest, task: asyncio.Task) -> None:
            state.inflight -= 1
            try:
                rr: RequestResult = task.result()
            except Exception:
                wake.set()
                return
            rr.admit_t = sr.admit_t
            rr.queue_wait_ms = max(0.0, (sr.admit_t - sr.arrival_t) * 1000.0)
            results.append(rr)
            state.record_completion(rr)
            wake.set()

        async def feeder():
            nonlocal feeder_done
            for arr in schedule:
                now = time.monotonic() - t0
                if arr.t - now > 0:
                    await asyncio.sleep(arr.t - now)
                queue.append(SchedRequest(
                    spec=arr.spec, arrival_t=arr.t, seq=arr.spec.seq,
                    size=float(size_fn(arr.spec)), deadline=deadline_fn(arr.spec, arr.t),
                ))
                wake.set()
            feeder_done = True
            wake.set()

        feeder_task = asyncio.create_task(feeder())

        def shed_expired(now: float) -> None:
            if max_wait_s is None or not queue:
                return
            keep = []
            for sr in queue:
                if now - sr.arrival_t > max_wait_s:
                    results.append(RequestResult(
                        profile=sr.spec.profile, spec_decode=sr.spec.spec_decode, seq=sr.seq,
                        arrival_t=sr.arrival_t, dispatch_t=now, ttft_ms=float("nan"),
                        tpot_ms=float("nan"), latency_ms=float("nan"),
                        output_tokens_requested=sr.spec.max_tokens, output_tokens_actual=0,
                        prompt_tokens=sr.spec.prompt_words, success=False,
                        is_warmup=(sr.arrival_t < warmup_s), error="dropped_timeout",
                        queue_wait_ms=(now - sr.arrival_t) * 1000.0,
                    ))
                else:
                    keep.append(sr)
            queue[:] = keep

        # Admitter loop.
        while True:
            shed_expired(time.monotonic() - t0)
            snap = sampler.latest() if sampler else {}
            cap = policy.admission_cap(state, snap)
            while state.inflight < cap and queue:
                sr = policy.select(queue, state, snap, time.monotonic() - t0)
                sr.admit_t = time.monotonic() - t0
                state.inflight += 1
                task = asyncio.create_task(
                    _send(client, base_url, model, sr.spec, sr.arrival_t, t0, warmup_s)
                )
                task.add_done_callback(lambda t, s=sr: on_done(s, t))
                snap = sampler.latest() if sampler else {}
                cap = policy.admission_cap(state, snap)
            if feeder_done and not queue and state.inflight == 0:
                break
            wake.clear()
            try:  # periodic re-eval so the adaptive cap reacts even without an event
                await asyncio.wait_for(wake.wait(), timeout=0.2)
            except asyncio.TimeoutError:
                pass

        await feeder_task
    return results, t0
