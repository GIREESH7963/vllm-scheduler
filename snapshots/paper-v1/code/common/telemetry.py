"""Telemetry samplers for a serving run.

Three sources, per project convention:
  * vLLM ``/metrics`` (Prometheus)  -> live KV occupancy, queue depth, spec-decode counters.
  * DCGM ``dcgmi dmon``             -> SM-active / DRAM-active (streamed subprocess).
  * NVML (pynvml)                   -> board power in watts (our ONLY power source).

A ``TelemetrySampler`` runs a background thread that polls ``/metrics`` + NVML on a fixed
interval, and reads a streaming ``dcgmi dmon`` subprocess in parallel. Call ``start()`` before
the measured window and ``stop()`` after; raw samples are then handed to ``common.metrics``.
"""
from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import dataclass, field

import httpx

try:  # NVML is optional at import time so the module loads off-GPU (e.g. for tests).
    import pynvml

    _NVML_OK = True
except Exception:  # pragma: no cover
    _NVML_OK = False


# --- Prometheus parsing -----------------------------------------------------------------

# One exposition line: name{labels} value   (we ignore HELP/TYPE comment lines).
_PROM_LINE = re.compile(r"^(?P<name>[a-zA-Z_:][\w:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>[-\d.eE+]+|NaN)$")


def parse_prometheus(text: str) -> dict[str, float]:
    """Parse Prometheus exposition text into {metric_name: value}.

    Histogram buckets (``_bucket``) are skipped. When a metric appears with multiple label
    sets (rare here — one model), values are summed, which is correct for the ``_total``
    counters we care about.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _PROM_LINE.match(line)
        if not m:
            continue
        name = m.group("name")
        if name.endswith("_bucket"):
            continue
        try:
            val = float(m.group("value"))
        except ValueError:
            continue
        out[name] = out.get(name, 0.0) + val
    return out


# --- sample records ---------------------------------------------------------------------


@dataclass
class MetricsSample:
    t: float
    num_running: float
    num_waiting: float
    kv_occupancy: float  # gpu_cache_usage_perc, 0..1
    power_w: float
    spec_accepted: float | None  # cumulative counter, if exposed
    spec_draft: float | None
    temp_c: float = float("nan")   # GPU temperature (NVML)
    throttled: bool = False        # thermal clock-slowdown active (SW or HW) — data suspect if True
    raw: dict[str, float] = field(repr=False, default_factory=dict)


# Thermal bits of the NVML clocks-throttle-reasons bitmask.
_THERMAL_BITS = 0x0000000000000020 | 0x0000000000000040 | 0x0000000000000008  # SWThermal|HWThermal|HWSlowdown


@dataclass
class DcgmSample:
    t: float  # time.monotonic() at capture, so samples can be filtered to the measured window
    sm_active: float  # 0..1
    dram_active: float  # 0..1


# --- the sampler ------------------------------------------------------------------------


class TelemetrySampler:
    def __init__(
        self,
        metrics_url: str = "http://localhost:8000/metrics",
        interval_s: float = 0.5,
        gpu_index: int = 0,
        dcgm_fields=(1002, 1005),  # sm_active, dram_active
        enable_dcgm: bool = True,
    ):
        self.metrics_url = metrics_url
        self.interval_s = interval_s
        self.gpu_index = gpu_index
        self.dcgm_fields = dcgm_fields
        self.enable_dcgm = enable_dcgm

        self.metrics_samples: list[MetricsSample] = []
        self.dcgm_samples: list[DcgmSample] = []
        self.notes: list[str] = []

        self._stop = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._dcgm_thread: threading.Thread | None = None
        self._dcgm_proc: subprocess.Popen | None = None
        self._nvml_handle = None
        self._client: httpx.Client | None = None

    # -- lifecycle --
    def start(self) -> None:
        self._client = httpx.Client(timeout=2.0)
        if _NVML_OK:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            except Exception as e:  # pragma: no cover
                self.notes.append(f"NVML init failed: {e}")
                self._nvml_handle = None
        else:
            self.notes.append("pynvml not importable; power will be NaN")

        self._stop.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        if self.enable_dcgm:
            self._dcgm_thread = threading.Thread(target=self._dcgm_loop, daemon=True)
            self._dcgm_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._poll_thread:
            self._poll_thread.join(timeout=5)
        if self._dcgm_proc and self._dcgm_proc.poll() is None:
            self._dcgm_proc.terminate()
            try:
                self._dcgm_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:  # pragma: no cover
                self._dcgm_proc.kill()
        if self._dcgm_thread:
            self._dcgm_thread.join(timeout=5)
        if self._client:
            self._client.close()

    # -- live snapshot for feedback control (Phase-2 adaptive policy) --
    def latest(self) -> dict:
        """Most-recent telemetry snapshot: kv_occupancy, sm_active (0..1), queue depths.

        Returns NaNs for any source with no samples yet. Cheap; safe to call from the scheduler
        loop between admissions.
        """
        m = self.metrics_samples[-1] if self.metrics_samples else None
        d = self.dcgm_samples[-1] if self.dcgm_samples else None
        return {
            "kv_occupancy": m.kv_occupancy if m else float("nan"),
            "num_running": m.num_running if m else float("nan"),
            "num_waiting": m.num_waiting if m else float("nan"),
            "sm_active": d.sm_active if d else float("nan"),
        }

    # -- /metrics + NVML poll loop --
    def _read_power_w(self) -> float:
        if self._nvml_handle is None:
            return float("nan")
        try:
            return pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0  # mW -> W
        except Exception:  # pragma: no cover
            return float("nan")

    def _read_thermal(self) -> tuple[float, bool]:
        if self._nvml_handle is None:
            return float("nan"), False
        try:
            temp = float(pynvml.nvmlDeviceGetTemperature(self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU))
        except Exception:  # pragma: no cover
            temp = float("nan")
        try:
            reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(self._nvml_handle)
            throttled = bool(reasons & _THERMAL_BITS)
        except Exception:  # pragma: no cover
            throttled = False
        return temp, throttled

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            t = time.monotonic()
            m: dict[str, float] = {}
            try:
                resp = self._client.get(self.metrics_url)
                if resp.status_code == 200:
                    m = parse_prometheus(resp.text)
            except Exception:
                pass  # transient; skip this tick's /metrics
            temp_c, throttled = self._read_thermal()
            self.metrics_samples.append(
                MetricsSample(
                    t=t,
                    num_running=m.get("vllm:num_requests_running", float("nan")),
                    num_waiting=m.get("vllm:num_requests_waiting", float("nan")),
                    kv_occupancy=m.get("vllm:gpu_cache_usage_perc", float("nan")),
                    power_w=self._read_power_w(),
                    spec_accepted=_first(m, "spec_decode_num_accepted_tokens"),
                    spec_draft=_first(m, "spec_decode_num_draft_tokens"),
                    temp_c=temp_c,
                    throttled=throttled,
                    raw=m,
                )
            )
            self._stop.wait(self.interval_s)

    # -- DCGM streaming loop --
    def _dcgm_loop(self) -> None:
        fields = ",".join(str(f) for f in self.dcgm_fields)
        delay_ms = max(100, int(self.interval_s * 1000))
        cmd = ["dcgmi", "dmon", "-e", fields, "-d", str(delay_ms)]
        try:
            self._dcgm_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )
        except FileNotFoundError:
            self.notes.append("dcgmi not found; SM/DRAM active unavailable")
            return
        for line in self._dcgm_proc.stdout:  # type: ignore[union-attr]
            if self._stop.is_set():
                break
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("Entity") or "ID" in line[:4]:
                continue
            # e.g. "GPU 0     0.123    0.045"
            parts = line.split()
            if len(parts) < 2 + len(self.dcgm_fields):
                continue
            nums = parts[-len(self.dcgm_fields):]
            try:
                vals = [float(x) for x in nums]
            except ValueError:
                continue  # "N/A" etc.
            self.dcgm_samples.append(
                DcgmSample(t=time.monotonic(), sm_active=vals[0], dram_active=vals[1])
            )


def read_temp_c(gpu_index: int = 0) -> float:
    """One-shot GPU temperature read via NVML (for the cooldown gate). NaN if unavailable."""
    if not _NVML_OK:
        return float("nan")
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        return float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
    except Exception:  # pragma: no cover
        return float("nan")


def wait_for_cool(threshold_c: float = 75.0, gpu_index: int = 0, timeout_s: float = 300.0,
                  poll_s: float = 5.0, log=print) -> float:
    """Block until GPU temp <= threshold (passive T4 throttles ~85C, wrecking measurements).

    Returns the final temperature. Times out after ``timeout_s`` to avoid stalling forever.
    """
    import time as _time

    t = read_temp_c(gpu_index)
    if t != t:  # NaN -> can't read; don't block
        return t
    start = _time.monotonic()
    warned = False
    while t > threshold_c and _time.monotonic() - start < timeout_s:
        if not warned:
            log(f"[thermal] GPU {t:.0f}C > {threshold_c:.0f}C — cooling down before measuring ...")
            warned = True
        _time.sleep(poll_s)
        t = read_temp_c(gpu_index)
    if warned:
        log(f"[thermal] resumed at {t:.0f}C")
    return t


def _first(m: dict[str, float], needle: str) -> float | None:
    """Return the first metric value whose name contains ``needle`` (spec-decode names vary)."""
    for k, v in m.items():
        if needle in k:
            return v
    return None
