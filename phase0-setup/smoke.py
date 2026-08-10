"""Phase-0 smoke run.

Fires a warmup burst (discarded), then a measured burst of streaming requests against a
running vLLM server, collects telemetry over the measured window, and writes one results
JSON in the project schema.

Usage (on the serving box, venv active):
    python -m phase0-setup.smoke --config configs/phase0_qwen1.5b.yaml

Note: the folder name has a hyphen, so it is not importable as a package path. Run this file
directly instead:  python phase0-setup/smoke.py --config configs/phase0_qwen1.5b.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

# Make the repo root importable when run as a script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_config  # noqa: E402
from common.metrics import RequestRecord, build_results, write_results  # noqa: E402
from common.telemetry import TelemetrySampler  # noqa: E402


async def wait_for_health(base_url: str, timeout_s: float = 180.0) -> None:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.monotonic() < deadline:
            try:
                r = await client.get(f"{base_url}/health")
                if r.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(1.0)
    raise TimeoutError(f"server at {base_url} not healthy within {timeout_s}s")


async def one_request(client: httpx.AsyncClient, cfg, base_url: str) -> RequestRecord:
    """Send one streaming chat request; time TTFT and mean TPOT client-side."""
    wl = cfg["workload"]
    body = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": wl["prompt"]}],
        "max_tokens": wl["max_tokens"],
        "temperature": wl["temperature"],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t_start = time.monotonic()
    t_first: float | None = None
    t_last: float = t_start
    content_chunks = 0
    usage_completion: int | None = None
    ok = True
    try:
        async with client.stream("POST", f"{base_url}/v1/chat/completions", json=body) as resp:
            if resp.status_code != 200:
                await resp.aread()
                return RequestRecord(float("nan"), float("nan"), 0, False)
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
    except Exception:
        ok = False

    out_tokens = usage_completion if usage_completion is not None else content_chunks
    ttft_ms = (t_first - t_start) * 1000.0 if t_first is not None else float("nan")
    # Mean per-output-token latency after the first token.
    if t_first is not None and out_tokens and out_tokens > 1:
        tpot_ms = (t_last - t_first) * 1000.0 / (out_tokens - 1)
    else:
        tpot_ms = float("nan")
    if out_tokens == 0:
        ok = False
    return RequestRecord(ttft_ms=ttft_ms, tpot_ms=tpot_ms, output_tokens=int(out_tokens), success=ok)


async def run_burst(cfg, base_url: str, n: int, concurrency: int) -> list[RequestRecord]:
    sem = asyncio.Semaphore(concurrency)
    results: list[RequestRecord] = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        async def guarded():
            async with sem:
                return await one_request(client, cfg, base_url)

        tasks = [asyncio.create_task(guarded()) for _ in range(n)]
        for t in asyncio.as_completed(tasks):
            results.append(await t)
    return results


async def main_async(cfg) -> None:
    base_url = cfg["server"]["base_url"]
    wl = cfg["workload"]
    tel = cfg.get("telemetry", {}) or {}

    print(f"[smoke] waiting for {base_url}/health ...")
    await wait_for_health(base_url)
    print("[smoke] server healthy.")

    # 1) Warmup — discarded, NOT sampled (keeps telemetry aligned to the measured window).
    print(f"[smoke] warmup: {wl['warmup_requests']} requests ...")
    await run_burst(cfg, base_url, wl["warmup_requests"], wl["concurrency"])

    # 2) Measured window — telemetry on.
    sampler = TelemetrySampler(
        metrics_url=f"{base_url}/metrics",
        interval_s=tel.get("interval_s", 0.5),
        gpu_index=tel.get("gpu_index", 0),
    )
    sampler.start()
    print(f"[smoke] measuring: {wl['measure_requests']} requests "
          f"(concurrency {wl['concurrency']}) ...")
    t0 = time.monotonic()
    records = await run_burst(cfg, base_url, wl["measure_requests"], wl["concurrency"])
    wall = time.monotonic() - t0
    sampler.stop()
    print(f"[smoke] measured window: {wall:.1f}s, "
          f"{sum(1 for r in records if r.success)}/{len(records)} ok, "
          f"{len(sampler.metrics_samples)} metric samples, "
          f"{len(sampler.dcgm_samples)} dcgm samples.")

    results = build_results(cfg, records, sampler, wall)
    path = write_results(results)
    print(f"[smoke] wrote {path}")
    print(json.dumps(results, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    asyncio.run(main_async(cfg))


if __name__ == "__main__":
    main()
