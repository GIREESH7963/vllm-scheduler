"""Open-loop Poisson load driver.

Requests arrive on a Poisson(λ) process — inter-arrival gaps are drawn from Exp(λ) with a fixed
seed — and each is dispatched at its scheduled time **regardless of how many are still in flight**
(open-loop, not closed-loop). Under saturation the in-flight count and queueing latency grow; that
is the signal we want to measure, not hide.

Each request is streamed so we can time TTFT and per-token latency client-side. We also record the
*scheduled* arrival vs the *actual* dispatch time: if the client can't keep up (CPU-bound at high
λ), that dispatch lag balloons and the run is untrustworthy — Phase-1 metrics surface it.
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field

import httpx

from .workload import Profile, RequestSpec, WorkloadMix


@dataclass
class RequestResult:
    profile: str
    spec_decode: bool
    seq: int                  # arrival index (stable under a fixed seed) — used for oracle lookup
    arrival_t: float          # scheduled arrival, seconds since run start
    dispatch_t: float         # actual dispatch, seconds since run start
    ttft_ms: float
    tpot_ms: float
    latency_ms: float         # end-to-end (dispatch -> last token)
    output_tokens_requested: int
    output_tokens_actual: int
    prompt_tokens: int
    success: bool
    is_warmup: bool = False
    error: str = ""
    # Phase-2 admission layer (0 in Phase-1 direct-dispatch runs): time spent waiting in the
    # scheduler queue before admission, and the total arrival->last-token latency.
    queue_wait_ms: float = 0.0
    admit_t: float = 0.0


@dataclass
class Arrival:
    t: float
    spec: RequestSpec


def build_schedule(rate: float, duration_s: float, mix: WorkloadMix, seed: int) -> list[Arrival]:
    """Pre-compute the arrival schedule: Exp(rate) gaps, profile per arrival from the mix.

    Using a fixed seed makes the exact arrival times AND the class of every request reproducible.
    """
    rng = random.Random(seed)
    arrivals: list[Arrival] = []
    t = 0.0
    seq = 0
    while True:
        t += rng.expovariate(rate)
        if t > duration_s:
            break
        profile: Profile = mix.pick(rng)
        from .workload import sample_request

        arrivals.append(Arrival(t=t, spec=sample_request(profile, rng, seq=seq)))
        seq += 1
    return arrivals


async def _send(client: httpx.AsyncClient, base_url: str, model: str, spec: RequestSpec,
                arrival_t: float, t0: float, warmup_s: float) -> RequestResult:
    dispatch_wall = time.monotonic()
    dispatch_t = dispatch_wall - t0
    body = {
        "model": model,
        "messages": [{"role": "user", "content": spec.prompt}],
        "max_tokens": spec.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t_first: float | None = None
    t_last = dispatch_wall
    content_chunks = 0
    usage_completion: int | None = None
    usage_prompt: int | None = None
    ok = True
    err = ""
    try:
        async with client.stream("POST", f"{base_url}/v1/chat/completions", json=body) as resp:
            if resp.status_code != 200:
                await resp.aread()
                ok, err = False, f"http_{resp.status_code}"
            else:
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        if delta.get("content"):
                            now = time.monotonic()
                            if t_first is None:
                                t_first = now
                            t_last = now
                            content_chunks += 1
                    if chunk.get("usage"):
                        usage_completion = chunk["usage"].get("completion_tokens")
                        usage_prompt = chunk["usage"].get("prompt_tokens")
    except Exception as e:  # noqa: BLE001
        ok, err = False, type(e).__name__

    out_actual = usage_completion if usage_completion is not None else content_chunks
    ttft_ms = (t_first - dispatch_wall) * 1000.0 if t_first is not None else float("nan")
    if t_first is not None and out_actual and out_actual > 1:
        tpot_ms = (t_last - t_first) * 1000.0 / (out_actual - 1)
    else:
        tpot_ms = float("nan")
    latency_ms = (t_last - dispatch_wall) * 1000.0
    if out_actual == 0 and ok:
        ok, err = False, "empty_output"
    return RequestResult(
        profile=spec.profile,
        spec_decode=spec.spec_decode,
        seq=spec.seq,
        arrival_t=arrival_t,
        dispatch_t=dispatch_t,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        latency_ms=latency_ms,
        output_tokens_requested=spec.max_tokens,
        output_tokens_actual=int(out_actual),
        prompt_tokens=int(usage_prompt) if usage_prompt is not None else spec.prompt_words,
        success=ok,
        is_warmup=(arrival_t < warmup_s),
        error=err,
    )


async def run_load(cfg, rate: float, seed: int, duration_override: float | None = None) -> tuple[list[RequestResult], float]:
    """Drive one open-loop run at ``rate`` req/s for ``workload.duration_s`` seconds.

    Returns ``(results, t0)`` where ``t0`` is the run-start ``time.monotonic()`` reference, so
    telemetry samples (same clock) can be filtered to the post-warmup measured window.
    ``duration_override`` shortens the run (used for the one-time pre-sweep warmup burst).
    """
    wl = cfg.get("workload", {}) or {}
    duration_s = float(duration_override) if duration_override else float(wl.get("duration_s", 120))
    warmup_s = float(wl.get("warmup_s", 30))
    base_url = cfg["server"]["base_url"]
    model = cfg["model"]

    mix = WorkloadMix.from_cfg(cfg)
    schedule = build_schedule(rate, duration_s, mix, seed)

    # Generous connection pool so the client — not the server — never becomes the bottleneck.
    limits = httpx.Limits(max_connections=1024, max_keepalive_connections=256)
    tasks: list[asyncio.Task] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0), limits=limits) as client:
        t0 = time.monotonic()
        for arr in schedule:
            now = time.monotonic() - t0
            wait = arr.t - now
            if wait > 0:
                await asyncio.sleep(wait)
            tasks.append(
                asyncio.create_task(_send(client, base_url, model, arr.spec, arr.t, t0, warmup_s))
            )
        # Let all in-flight requests drain.
        results = await asyncio.gather(*tasks) if tasks else []
    return list(results), t0
