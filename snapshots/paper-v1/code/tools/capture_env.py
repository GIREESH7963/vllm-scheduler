"""Capture the full software/hardware environment for a measurement run.

Written for the paper-v1 freeze, but kept general so Experiments A and B can call it and
produce a directly comparable manifest.

Design note: NVML is treated as *optional and possibly broken*. On this box the userspace
driver was upgraded out from under the loaded kernel module (see ``driver.mismatch``), which
kills NVML while leaving the CUDA driver API working. Every field therefore records how it was
obtained, so a later reader can tell a real measurement from a hole in the telemetry.

Usage:
    python tools/capture_env.py                 # print JSON to stdout
    python tools/capture_env.py -o env.json     # write JSON
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str] | str, timeout: int = 15) -> str | None:
    """Run a command, returning stripped stdout, or None if it fails for any reason."""
    try:
        out = subprocess.run(
            cmd,
            shell=isinstance(cmd, str),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return None
    if out.returncode != 0:
        # nvidia-smi returns non-zero on the NVML mismatch but still prints a useful message.
        return (out.stdout + out.stderr).strip() or None
    return out.stdout.strip() or None


def capture_host() -> dict:
    return {
        "hostname": platform.node(),
        "kernel": platform.release(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
    }


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def capture_driver() -> dict:
    """Driver state, including an explicit mismatch check.

    ``/proc/driver/nvidia/version`` reports the *loaded kernel module*; the ``libnvidia-ml``
    soname on disk reports the *installed userspace*. When a package upgrade lands without a
    reboot these diverge, NVML refuses to initialise, and every NVML-sourced field (power,
    temperature, utilisation, memory) silently becomes unavailable.
    """
    info: dict = {}

    proc_ver = None
    try:
        proc_ver = Path("/proc/driver/nvidia/version").read_text().strip()
    except Exception:
        pass
    info["proc_driver_version_raw"] = proc_ver

    kernel_module_version = None
    if proc_ver:
        m = re.search(r"Kernel Module\s+([\d.]+)", proc_ver)
        if m:
            kernel_module_version = m.group(1)
    info["kernel_module_version"] = kernel_module_version

    # Installed userspace NVML: the versioned soname is the authoritative marker.
    userspace_version = None
    for libdir in ("/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib"):
        d = Path(libdir)
        if not d.is_dir():
            continue
        for so in d.glob("libnvidia-ml.so.*"):
            m = re.fullmatch(r"libnvidia-ml\.so\.(\d+\.\d+(?:\.\d+)?)", so.name)
            if m:
                userspace_version = m.group(1)
                break
        if userspace_version:
            break
    info["userspace_nvml_version"] = userspace_version

    mismatch = bool(
        kernel_module_version
        and userspace_version
        and not userspace_version.startswith(kernel_module_version.split(".")[0] + ".")
        or (
            kernel_module_version
            and userspace_version
            and kernel_module_version != userspace_version
        )
    )
    info["mismatch"] = mismatch
    info["mismatch_note"] = (
        f"loaded kernel module {kernel_module_version} != installed userspace "
        f"{userspace_version}; NVML unavailable until modules are reloaded or the host reboots"
        if mismatch
        else None
    )

    info["nvidia_smi"] = _run(["nvidia-smi"], timeout=20)
    return info


def capture_nvml() -> dict:
    """NVML-sourced GPU facts. Returns ``available: False`` plus the error when broken."""
    try:
        import pynvml
    except Exception as e:
        return {"available": False, "error": f"pynvml not importable: {e}"}

    try:
        pynvml.nvmlInit()
    except Exception as e:
        return {"available": False, "error": f"nvmlInit failed: {e}"}

    out: dict = {"available": True, "driver_version": None, "devices": []}
    try:
        out["driver_version"] = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(out["driver_version"], bytes):
            out["driver_version"] = out["driver_version"].decode()
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            name = pynvml.nvmlDeviceGetName(h)
            out["devices"].append(
                {
                    "index": i,
                    "name": name.decode() if isinstance(name, bytes) else name,
                    "memory_total_bytes": mem.total,
                    "memory_used_bytes": mem.used,
                    "memory_free_bytes": mem.free,
                    "temperature_c": pynvml.nvmlDeviceGetTemperature(
                        h, pynvml.NVML_TEMPERATURE_GPU
                    ),
                    "power_w": pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
                    "power_limit_w": pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0,
                }
            )
    except Exception as e:
        out["partial_error"] = str(e)
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return out


def capture_torch_cuda() -> dict:
    """CUDA facts via torch. Works even when NVML is dead, so this is the fallback source."""
    try:
        import torch
    except Exception as e:
        return {"available": False, "error": str(e)}

    out: dict = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
    }
    if not out["cuda_available"]:
        return out

    out["device_count"] = torch.cuda.device_count()
    devices = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        devices.append(
            {
                "index": i,
                "name": p.name,
                "compute_capability": f"{p.major}.{p.minor}",
                "multi_processor_count": p.multi_processor_count,
                "memory_total_bytes": total,
                "memory_total_gib": round(total / 2**30, 4),
                "memory_free_bytes": free,
                "memory_free_gib": round(free / 2**30, 4),
                # Non-zero free-vs-total gap means another process (e.g. a live vLLM server)
                # already holds device memory; record it so runs are not compared blindly.
                "memory_in_use_by_others_gib": round((total - free) / 2**30, 4),
            }
        )
    out["devices"] = devices
    return out


def capture_dcgm() -> dict:
    """DCGM state. DCGM survives some NVML breakage, so it is captured independently."""
    out: dict = {"dcgmi_path": _run("command -v dcgmi")}
    if not out["dcgmi_path"]:
        out["available"] = False
        return out
    disc = _run(["dcgmi", "discovery", "-l"], timeout=20)
    out["discovery"] = disc
    out["available"] = bool(disc and "GPU" in disc)
    # A 3-sample probe of the two profiling fields the harness relies on.
    out["dmon_probe_sm_dram"] = _run("timeout 10 dcgmi dmon -e 1002,1005 -c 3", timeout=20)
    return out


def capture_packages() -> dict:
    """Exact resolved versions of every installed package (the real dependency record)."""
    out: dict = {}
    for mod in ("vllm", "torch", "transformers", "numpy", "pandas", "matplotlib", "httpx"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception as e:
            out[mod] = f"<not importable: {e}>"
    out["_pip_freeze"] = _run([sys.executable, "-m", "pip", "freeze"], timeout=60)
    return out


def capture_server(base_url: str = "http://localhost:8000") -> dict:
    """Whether a vLLM server is live, and what it is serving."""
    out: dict = {"base_url": base_url, "reachable": False}
    try:
        import httpx

        r = httpx.get(f"{base_url}/v1/models", timeout=5.0)
        out["reachable"] = r.status_code == 200
        if out["reachable"]:
            out["models"] = r.json()
    except Exception as e:
        out["error"] = str(e)

    # The exact argv of the running server is the ground truth for serving configuration.
    ps = _run("pgrep -af 'vllm serve' || true")
    out["server_process_argv"] = ps
    return out


def capture(base_url: str = "http://localhost:8000") -> dict:
    return {
        "captured_at_utc": _run("date -u +%Y-%m-%dT%H:%M:%SZ"),
        "host": capture_host(),
        "driver": capture_driver(),
        "nvml": capture_nvml(),
        "torch_cuda": capture_torch_cuda(),
        "dcgm": capture_dcgm(),
        "packages": capture_packages(),
        "server": capture_server(base_url),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", type=Path, help="write JSON here instead of stdout")
    ap.add_argument("--base-url", default="http://localhost:8000")
    args = ap.parse_args()

    env = capture(args.base_url)
    text = json.dumps(env, indent=2, sort_keys=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
        print(f"wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
