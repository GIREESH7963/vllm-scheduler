"""Experiment A — reproduce the CUDA OOM and characterise the state immediately before it.

The paper-v1 dataset contains exactly **one** OOM event. The three `λ=5` rows that look like
reproductions are not: the engine died once at 07-07 07:12:35 during `λ=4 rep=2`, and every
later request merely echoed the same cached `MQEngineDeadError`. A claim of the form "OOM occurs
reproducibly while KV occupancy stays low" cannot rest on n=1, so this probe rebuilds the
evidence properly:

  * **One fresh server per trial.** An OOM kills the vLLM engine process. Reusing a dead engine
    produces failures that look like OOMs but carry no memory information at all. Every trial
    therefore starts a new server and tears it down afterwards.
  * **Ramp until death, don't guess the rate.** Offered load escalates through stages until the
    engine dies, so the trial finds the boundary instead of assuming λ=4 hits it.
  * **A telemetry ring buffer.** Everything is sampled continuously; on death we keep the last
    ``--pre-window`` seconds. The interesting state is the state *just before* failure, which a
    post-hoc summary cannot recover.
  * **Allocator forensics from the traceback.** The failed allocation size, the allocator
    breakdown, and the *call site* come from the server log. The call site is the whole point:
    in paper-v1 it was ``spec_decode/batch_expansion.py:_contract_batch``, i.e. the speculative
    scorer's dense [batch x (k+1) x vocab] probability tensor — an allocation that lives outside
    the paged KV allocator and is invisible to vLLM's admission control.

Usage:
    python phase3-limits/oom_probe.py --config configs/oomprobe_qwen1.5b.yaml --trials 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_config  # noqa: E402
from common.loadgen import build_schedule  # noqa: E402
from common.workload import WorkloadMix  # noqa: E402

try:
    import pynvml

    _NVML = True
except Exception:  # pragma: no cover
    _NVML = False


# --------------------------------------------------------------------------------------
# Server lifecycle
# --------------------------------------------------------------------------------------


class VllmServer:
    """Launch a vLLM server as a subprocess and tear it down deterministically.

    The log file is per-trial: OOM forensics are parsed out of it, and a shared log would make
    it impossible to tell which trial a traceback belongs to.
    """

    def __init__(self, cfg: dict, log_path: Path, port: int):
        self.cfg = cfg
        self.log_path = log_path
        self.port = port
        self.proc: subprocess.Popen | None = None
        self.base_url = f"http://127.0.0.1:{port}"

    def argv(self) -> list[str]:
        s = self.cfg["server"]
        venv_bin = Path(sys.executable).parent
        argv = [
            str(venv_bin / "vllm"), "serve", self.cfg["model"],
            "--port", str(self.port),
            "--dtype", str(s.get("dtype", "float16")),
            "--gpu-memory-utilization", str(s.get("gpu_memory_utilization", 0.9)),
        ]
        if s.get("max_model_len"):
            argv += ["--max-model-len", str(s["max_model_len"])]
        if s.get("max_num_seqs"):
            argv += ["--max-num-seqs", str(s["max_num_seqs"])]
        spec = s.get("speculative_config")
        if spec:
            # vLLM 0.8.5 takes this as a single JSON string argument.
            argv += ["--speculative-config", json.dumps(spec)]
        if s.get("enforce_eager"):
            argv += ["--enforce-eager"]
        return argv

    def spawn(self) -> None:
        """Launch the server process without waiting for it to become healthy.

        Split out of ``start`` so a caller that only needs something vLLM prints during startup
        — the memory-profiling split, say — can read it and tear down without paying for
        cudagraph capture and warmup. ``start`` is this plus the health wait.
        """
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = self.log_path.open("w", encoding="utf-8")
        env = dict(os.environ)
        env.setdefault("VLLM_LOGGING_LEVEL", "INFO")
        self.proc = subprocess.Popen(
            self.argv(), stdout=fh, stderr=subprocess.STDOUT, env=env,
            # New process group: an OOM leaves worker children behind, and killing the group
            # is the only reliable way to free the GPU before the next trial.
            start_new_session=True,
        )

    def returncode(self) -> int | None:
        return self.proc.poll() if self.proc else None

    def start(self, timeout_s: float = 600.0) -> None:
        self.spawn()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"server exited during startup (rc={self.proc.returncode}); see {self.log_path}"
                )
            try:
                if httpx.get(f"{self.base_url}/health", timeout=2.0).status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(2.0)
        raise TimeoutError(f"server not healthy within {timeout_s}s; see {self.log_path}")

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        if not self.proc:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except Exception:
            pass
        try:
            self.proc.wait(timeout=45)
        except Exception:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:
                pass
            try:
                self.proc.wait(timeout=20)
            except Exception:
                pass
        self.proc = None


def wait_gpu_free(threshold_gib: float = 1.0, timeout_s: float = 180.0) -> float:
    """Block until the GPU is essentially empty again.

    Between trials the previous engine's memory must actually be reclaimed, otherwise trial N+1
    starts with less headroom than trial N and the OOM boundary drifts for the wrong reason.
    """
    if not _NVML:
        time.sleep(20)
        return float("nan")
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    deadline = time.monotonic() + timeout_s
    used = float("nan")
    while time.monotonic() < deadline:
        used = pynvml.nvmlDeviceGetMemoryInfo(h).used / 2**30
        if used < threshold_gib:
            break
        time.sleep(3.0)
    pynvml.nvmlShutdown()
    return used


def wait_cool(target_c: float, timeout_s: float = 420.0) -> float:
    """Wait for the (passively cooled) T4 to drop below ``target_c`` before a trial."""
    if not _NVML:
        return float("nan")
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    deadline = time.monotonic() + timeout_s
    t = float("nan")
    while time.monotonic() < deadline:
        t = float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
        if t <= target_c:
            break
        time.sleep(5.0)
    pynvml.nvmlShutdown()
    return t


# --------------------------------------------------------------------------------------
# Telemetry
# --------------------------------------------------------------------------------------

_PROM = re.compile(r"^(?P<name>[a-zA-Z_:][\w:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>[-\d.eE+]+|NaN)$")

# vLLM 0.8.5 exposes these unprefixed; keep both spellings so a version bump degrades to NaN
# rather than silently reporting zero.
_KV_KEYS = ("vllm:gpu_cache_usage_perc", "gpu_cache_usage_perc")
_RUN_KEYS = ("vllm:num_requests_running", "num_requests_running")
_WAIT_KEYS = ("vllm:num_requests_waiting", "num_requests_waiting")


def _parse_prom(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _PROM.match(line)
        if not m:
            continue
        name = m.group("name")
        if name.endswith("_bucket"):
            continue
        try:
            out[name] = out.get(name, 0.0) + float(m.group("value"))
        except ValueError:
            continue
    return out


def _pick(d: dict[str, float], keys) -> float:
    for k in keys:
        if k in d:
            return d[k]
    return float("nan")


class Sampler:
    """Continuous sampler writing into a bounded ring buffer.

    Captures, on one clock, every field Experiment A must report at the moment of failure:
    GPU memory (total/used/free, and this-process), KV occupancy, running/waiting concurrency,
    SM and DRAM utilisation, temperature and power.
    """

    def __init__(self, base_url: str, interval_s: float, window_s: float, gpu_index: int = 0):
        self.base_url = base_url
        self.interval_s = interval_s
        self.gpu_index = gpu_index
        self.buf: deque[dict] = deque(maxlen=max(64, int(window_s / interval_s) + 8))
        self.peak: dict = {"num_running": 0.0, "kv_occupancy": 0.0, "mem_used_gib": 0.0}
        self._client = httpx.Client(timeout=2.0)
        self._h = None
        if _NVML:
            pynvml.nvmlInit()
            self._h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    def sample(self) -> dict:
        rec: dict = {"t": time.time(), "mono": time.monotonic()}

        # --- vLLM /metrics ---
        try:
            prom = _parse_prom(self._client.get(f"{self.base_url}/metrics").text)
            rec["kv_occupancy"] = _pick(prom, _KV_KEYS)
            rec["num_running"] = _pick(prom, _RUN_KEYS)
            rec["num_waiting"] = _pick(prom, _WAIT_KEYS)
        except Exception as e:
            rec["kv_occupancy"] = rec["num_running"] = rec["num_waiting"] = float("nan")
            rec["metrics_error"] = type(e).__name__

        # --- NVML: memory / thermal / power / utilisation ---
        if self._h is not None:
            try:
                m = pynvml.nvmlDeviceGetMemoryInfo(self._h)
                rec["mem_total_gib"] = m.total / 2**30
                rec["mem_used_gib"] = m.used / 2**30
                rec["mem_free_gib"] = m.free / 2**30
                u = pynvml.nvmlDeviceGetUtilizationRates(self._h)
                rec["util_gpu_pct"] = float(u.gpu)
                rec["util_mem_pct"] = float(u.memory)
                rec["temp_c"] = float(pynvml.nvmlDeviceGetTemperature(self._h, pynvml.NVML_TEMPERATURE_GPU))
                rec["power_w"] = pynvml.nvmlDeviceGetPowerUsage(self._h) / 1000.0
                try:
                    reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(self._h)
                    rec["thermal_throttled"] = bool(reasons & (0x20 | 0x40 | 0x8))
                except Exception:
                    rec["thermal_throttled"] = None
                # Per-process attribution separates engine memory from anything else resident.
                procs = []
                for p in pynvml.nvmlDeviceGetComputeRunningProcesses(self._h):
                    procs.append({"pid": p.pid, "used_gib": (p.usedGpuMemory or 0) / 2**30})
                rec["procs"] = procs
            except Exception as e:
                rec["nvml_error"] = type(e).__name__

        for k in ("num_running", "kv_occupancy", "mem_used_gib"):
            v = rec.get(k)
            if v is not None and v == v and v > self.peak[k]:
                self.peak[k] = v

        self.buf.append(rec)
        return rec

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
        if self._h is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


# --------------------------------------------------------------------------------------
# OOM forensics
# --------------------------------------------------------------------------------------

_ALLOC = re.compile(r"Tried to allocate ([\d.]+) ([KMG])iB")
_CAP = re.compile(r"total capacity of ([\d.]+) GiB of which ([\d.]+) ([KMG])iB is free")
_INUSE = re.compile(r"this process has ([\d.]+) GiB memory in use")
_BYTORCH = re.compile(r"([\d.]+) GiB is allocated by PyTorch")
_RESERVED = re.compile(r"([\d.]+) ([KMG])iB is reserved by PyTorch but unallocated")
_FRAME = re.compile(r'File "([^"]+)", line (\d+), in (\S+)')

_UNIT = {"K": 1 / 2**20, "M": 1.0, "G": 1024.0}  # -> MiB


def parse_oom(log_path: Path) -> dict | None:
    """Extract the first OOM event: sizes, allocator breakdown, and the failing call site.

    Only the *first* event matters. Once the engine loop dies, vLLM replays the same exception
    to every outstanding request — in paper-v1 that produced 223 identical lines from a single
    failure, which is exactly the artefact that made n=1 look like n=3.
    """
    try:
        text = log_path.read_text(errors="replace")
    except Exception:
        return None
    if "OutOfMemoryError" not in text:
        return None

    lines = text.splitlines()

    # Scope to the ENGINE's own traceback. vLLM logs the failure twice: once from the engine
    # process (prefixed `[engine.py:NNN]`, carrying the real allocation stack) and then once per
    # outstanding request from the API layer (`serving_chat.py` / `client.py`) as the cached
    # MQEngineDeadError is replayed. Taking the first "OutOfMemoryError" line and reading
    # forwards lands in the replay region and reports `client.py:_process_request` as the
    # allocation site, which is wrong — the replay frames belong to the client, not the
    # allocator.
    engine_idx = [i for i, ln in enumerate(lines) if "[engine.py:" in ln]
    engine_lines = [lines[i] for i in engine_idx]
    engine_text = "\n".join(engine_lines)

    if "OutOfMemoryError" in engine_text:
        block = engine_text
        idx = next(i for i in engine_idx if "OutOfMemoryError" in lines[i])
        out_scope = "engine"
    else:
        # No engine-prefixed traceback (e.g. a worker crash logged elsewhere): fall back to a
        # forward window, and say so, rather than silently reporting a client frame.
        idx = next(i for i, ln in enumerate(lines) if "OutOfMemoryError" in ln)
        block = "\n".join(lines[idx : idx + 120])
        out_scope = "fallback_window"

    out: dict = {"first_oom_line_no": idx + 1, "traceback_scope": out_scope}

    ts = re.match(r"^\w+\s+([\d-]+\s[\d:]+)", lines[idx])
    out["timestamp"] = ts.group(1) if ts else None

    if (m := _ALLOC.search(block)):
        out["failed_alloc_mib"] = float(m.group(1)) * _UNIT[m.group(2)]
    if (m := _CAP.search(block)):
        out["total_capacity_gib"] = float(m.group(1))
        out["free_at_failure_mib"] = float(m.group(2)) * _UNIT[m.group(3)]
    if (m := _INUSE.search(block)):
        out["process_in_use_gib"] = float(m.group(1))
    if (m := _BYTORCH.search(block)):
        out["allocated_by_torch_gib"] = float(m.group(1))
    if (m := _RESERVED.search(block)):
        out["reserved_unallocated_mib"] = float(m.group(1)) * _UNIT[m.group(2)]

    # Deepest vLLM frame = the allocation site. This is the field that distinguishes a
    # KV-cache exhaustion from an activation-tensor blowup, so it is taken from the *last*
    # frame before the exception line rather than anywhere in the block.
    err_pos = block.find("OutOfMemoryError: CUDA out of memory")
    stack_text = block[:err_pos] if err_pos > 0 else block
    frames = [
        {"file": f, "line": int(n), "func": fn}
        for f, n, fn in _FRAME.findall(stack_text)
    ]
    out["traceback_frames"] = frames[-14:]
    vllm_frames = [f for f in frames if "/vllm/" in f["file"]]
    if vllm_frames:
        deep = vllm_frames[-1]
        out["alloc_site"] = f"{Path(deep['file']).name}:{deep['line']} in {deep['func']}"
        out["alloc_site_module"] = str(Path(deep["file"]).parent.name)
        out["in_spec_decode_path"] = any("spec_decode" in f["file"] for f in vllm_frames)
        out["in_kv_cache_path"] = any(
            ("block_manager" in f["file"] or "kv_cache" in f["file"] or "cache_engine" in f["file"])
            for f in vllm_frames
        )

    out["total_oom_lines_in_log"] = sum(1 for ln in lines if "OutOfMemoryError" in ln)
    out["engine_dead_errors"] = sum(1 for ln in lines if "MQEngineDeadError" in ln)
    return out


# --------------------------------------------------------------------------------------
# Load ramp
# --------------------------------------------------------------------------------------


async def _fire(client: httpx.AsyncClient, base_url: str, model: str, spec, state: dict) -> None:
    """Send one streamed request; record only whether it survived and why it didn't."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": spec.prompt}],
        "max_tokens": spec.max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    try:
        async with client.stream("POST", f"{base_url}/v1/chat/completions", json=body) as r:
            if r.status_code != 200:
                payload = (await r.aread()).decode(errors="replace")[:400]
                state["failed"] += 1
                state["last_error"] = f"http_{r.status_code}: {payload}"
                return
            async for line in r.aiter_lines():
                if line.startswith("data:") and line[5:].strip() == "[DONE]":
                    break
            state["ok"] += 1
    except Exception as e:
        state["failed"] += 1
        state["last_error"] = f"{type(e).__name__}: {e}"


async def ramp_to_failure(server: VllmServer, cfg: dict, sampler: Sampler, args) -> dict:
    """Escalate offered load through stages until the engine dies (or stages run out)."""
    mix = WorkloadMix.from_cfg(cfg)
    model = cfg["model"]
    stages = [float(x) for x in cfg["ramp"]["rates"]]
    stage_s = float(cfg["ramp"]["stage_s"])
    seed = int(cfg["ramp"].get("seed", 4242))

    state = {"ok": 0, "failed": 0, "last_error": ""}
    outcome = {"died": False, "death_stage": None, "death_reason": None, "stage_log": []}

    limits = httpx.Limits(max_connections=2048, max_keepalive_connections=512)
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0), limits=limits) as client:
        tasks: list[asyncio.Task] = []
        stop = asyncio.Event()

        async def sample_loop() -> None:
            while not stop.is_set():
                sampler.sample()
                await asyncio.sleep(sampler.interval_s)

        sampler_task = asyncio.create_task(sample_loop())

        try:
            for stage_i, rate in enumerate(stages):
                schedule = build_schedule(rate, stage_s, mix, seed + stage_i)
                t0 = time.monotonic()
                before_ok, before_fail = state["ok"], state["failed"]

                for arr in schedule:
                    wait = arr.t - (time.monotonic() - t0)
                    if wait > 0:
                        await asyncio.sleep(wait)
                    if not server.alive():
                        outcome.update(died=True, death_stage=rate, death_reason="process_exited")
                        break
                    tasks.append(asyncio.create_task(_fire(client, server.base_url, model, arr.spec, state)))

                if outcome["died"]:
                    break

                # Let the stage drain briefly so num_running reflects real steady state.
                await asyncio.sleep(min(5.0, stage_s * 0.1))

                last = sampler.buf[-1] if sampler.buf else {}
                stage_rec = {
                    "rate": rate,
                    "ok": state["ok"] - before_ok,
                    "failed": state["failed"] - before_fail,
                    "num_running": last.get("num_running"),
                    "kv_occupancy": last.get("kv_occupancy"),
                    "mem_used_gib": last.get("mem_used_gib"),
                    "temp_c": last.get("temp_c"),
                }
                outcome["stage_log"].append(stage_rec)
                print(f"    stage λ={rate:g}: ok={stage_rec['ok']} fail={stage_rec['failed']} "
                      f"running={stage_rec['num_running']} kv={stage_rec['kv_occupancy']} "
                      f"mem={stage_rec['mem_used_gib']:.2f}GiB temp={stage_rec['temp_c']}C"
                      if stage_rec["mem_used_gib"] is not None else f"    stage λ={rate:g}")

                # An engine that dies mid-stage shows up as a burst of failures.
                if not server.alive():
                    outcome.update(died=True, death_stage=rate, death_reason="process_exited")
                    break
                try:
                    if httpx.get(f"{server.base_url}/health", timeout=5.0).status_code != 200:
                        outcome.update(died=True, death_stage=rate, death_reason="health_failed")
                        break
                except Exception:
                    outcome.update(died=True, death_stage=rate, death_reason="health_unreachable")
                    break
        finally:
            stop.set()
            sampler_task.cancel()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    outcome["requests_ok"] = state["ok"]
    outcome["requests_failed"] = state["failed"]
    outcome["last_error"] = state["last_error"][:500]
    return outcome


# --------------------------------------------------------------------------------------
# Trial driver
# --------------------------------------------------------------------------------------


def run_trial(trial: int, cfg: dict, args, out_dir: Path) -> dict:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log_path = out_dir / "logs" / f"{ts}_trial{trial}_server.log"
    print(f"\n[trial {trial}] cooling / waiting for a free GPU ...")
    free_used = wait_gpu_free()
    temp = wait_cool(float(cfg.get("telemetry", {}).get("cooldown_c", 77)))
    print(f"[trial {trial}] GPU used={free_used:.2f}GiB temp={temp}C -> starting server")

    server = VllmServer(cfg, log_path, port=args.port)
    record: dict = {
        "experiment": "A_oom_reproduction",
        "trial": trial,
        "timestamp_utc": ts,
        "model": cfg["model"],
        "server_argv": server.argv(),
        "pre_start_gpu_used_gib": free_used,
        "pre_start_temp_c": temp,
        "ramp": cfg["ramp"],
    }

    t_start = time.monotonic()
    try:
        server.start()
    except Exception as e:
        record["error"] = f"server_start_failed: {e}"
        server.stop()
        return record
    record["startup_s"] = round(time.monotonic() - t_start, 1)

    sampler = Sampler(server.base_url, args.interval, args.pre_window)
    try:
        outcome = asyncio.run(ramp_to_failure(server, cfg, sampler, args))
        record.update(outcome)
        # Preserve the pre-failure window at full resolution — this is the payload.
        record["pre_failure_window"] = list(sampler.buf)
        record["peaks"] = dict(sampler.peak)
    finally:
        sampler.close()
        server.stop()

    # The engine writes its traceback as it dies; give the log a moment to flush.
    time.sleep(3.0)
    record["oom"] = parse_oom(log_path)
    record["server_log"] = str(log_path.relative_to(out_dir.parent)) if out_dir.parent in log_path.parents else str(log_path)

    # State at the last sample that still saw a live engine.
    live = [s for s in record.get("pre_failure_window", []) if s.get("num_running") == s.get("num_running")]
    if live:
        last = live[-1]
        record["at_failure"] = {
            "num_running": last.get("num_running"),
            "num_waiting": last.get("num_waiting"),
            "kv_occupancy": last.get("kv_occupancy"),
            "mem_total_gib": last.get("mem_total_gib"),
            "mem_used_gib": last.get("mem_used_gib"),
            "mem_free_gib": last.get("mem_free_gib"),
            "util_gpu_pct": last.get("util_gpu_pct"),
            "util_mem_pct": last.get("util_mem_pct"),
            "temp_c": last.get("temp_c"),
            "power_w": last.get("power_w"),
            "thermal_throttled": last.get("thermal_throttled"),
        }
        # Maximum concurrency/KV actually observed, which is what the boundary claim is about.
        record["at_failure"]["peak_num_running"] = record["peaks"]["num_running"]
        record["at_failure"]["peak_kv_occupancy"] = record["peaks"]["kv_occupancy"]

    path = out_dir / f"{ts}_expA_trial{trial}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(f"[trial {trial}] wrote {path.name}")
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--interval", type=float, default=0.25, help="telemetry sample period (s)")
    ap.add_argument("--pre-window", type=float, default=90.0, help="seconds of pre-failure telemetry to keep")
    ap.add_argument("--out-dir", default="results/expA")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for trial in range(1, args.trials + 1):
        records.append(run_trial(trial, cfg, args, out_dir))

    # --- summary ---
    print("\n" + "=" * 78)
    print("EXPERIMENT A SUMMARY")
    print("=" * 78)
    hdr = f"{'trial':>5} {'died':>5} {'alloc MiB':>10} {'site':>34} {'run':>6} {'kv%':>7} {'mem GiB':>8}"
    print(hdr)
    n_oom = 0
    for r in records:
        oom = r.get("oom") or {}
        af = r.get("at_failure") or {}
        if oom.get("failed_alloc_mib"):
            n_oom += 1
        kv = af.get("peak_kv_occupancy")
        print(f"{r['trial']:>5} {str(r.get('died')):>5} "
              f"{oom.get('failed_alloc_mib', float('nan')):>10.1f} "
              f"{str(oom.get('alloc_site'))[:34]:>34} "
              f"{af.get('peak_num_running', float('nan')):>6.0f} "
              f"{(kv * 100 if kv is not None else float('nan')):>7.2f} "
              f"{af.get('mem_used_gib', float('nan')):>8.2f}")
    print(f"\nreproduced OOM in {n_oom}/{len(records)} independent trials")

    summ = out_dir / "expA_summary.json"
    summ.write_text(json.dumps(records, indent=2, default=str) + "\n")
    print(f"wrote {summ}")


if __name__ == "__main__":
    main()
