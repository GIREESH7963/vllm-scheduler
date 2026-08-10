"""Freeze a self-contained, read-only experiment snapshot.

Motivation: ``results/`` is a live directory — every later sweep writes new JSON into it and
overwrites the summaries and figures derived from it. A git tag alone does not protect the
baseline, because a future run can still clobber the working tree. So this tool takes a
*physical* copy of code + configs + raw outputs + plots + environment, checksums every file,
and drops write permission on the result.

Usage:
    python tools/freeze_snapshot.py paper-v1
    python tools/freeze_snapshot.py paper-v1 --force   # re-freeze, replacing an existing one
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Source trees copied verbatim into the snapshot so it stands alone without the git history.
CODE_DIRS = ("common", "configs", "phase0-setup", "phase1-harness", "phase2-policies",
             "phase3-limits", "tools")
CODE_FILES = ("requirements.txt", "status.sh")
# Prose that interprets the results — the derivations and analyses a reader needs in order to
# know what the numbers in results/ mean. Frozen alongside them so the two cannot drift.
DOC_DIRS = ("docs",)
# Logs are compressed: server.log is ~45 MB of vLLM stdout and holds the only record of the
# OOM event, so it must be preserved, but not at full size.
LOG_FILES = (
    "server.log",
    "install.log",
    "model_dl.log",
    "modelval.log",
    "kvprobe_8k.log",
    "phase1_mixed.log",
    "phase1_validation.log",
    "phase2_all.log",
    "phase2_mixed.log",
)
EXCLUDE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".venv")


def _make_writable(path: Path) -> None:
    """Restore write permission across a tree so a previous frozen snapshot can be replaced."""
    if not path.exists():
        return
    for p in [path, *path.rglob("*")]:
        try:
            p.chmod(p.stat().st_mode | stat.S_IWUSR)
        except Exception:
            pass


def _freeze_tree(path: Path) -> int:
    """Drop write permission for everyone. Returns the number of paths altered."""
    n = 0
    for p in sorted(path.rglob("*"), reverse=True):
        try:
            mode = p.stat().st_mode
            p.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
            n += 1
        except Exception:
            pass
    try:
        path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        n += 1
    except Exception:
        pass
    return n


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=120
        )
    except Exception:
        return None
    return (r.stdout or "").strip() if r.returncode == 0 else None


def _readme(name: str, env: dict, dest: Path, supersedes: str | None = None) -> str:
    """Generate the provenance document from the captured environment.

    The freeze-time environment is not necessarily the measurement-time environment — this box
    had its userspace NVIDIA driver upgraded weeks after the runs — so the document states
    which is which rather than implying the two are the same.
    """
    gpu = (env.get("torch_cuda", {}).get("devices") or [{}])[0]
    drv = env.get("driver", {})
    pkg = env.get("packages", {})
    git = env.get("git", {})
    nvml = env.get("nvml", {})

    n_results = len(list((dest / "results").glob("*.json"))) if (dest / "results").is_dir() else 0
    n_figs = len(list((dest / "results" / "figures").glob("*.png"))) if (dest / "results" / "figures").is_dir() else 0
    n_paper = len(list((dest / "results" / "figures" / "paper").glob("*.pdf")))
    docs = sorted(p.name for d in DOC_DIRS if (dest / d).is_dir()
                  for p in (dest / d).glob("*.md"))
    doc_row = (f"| `docs/` | {', '.join(f'`{d}`' for d in docs)} |\n" if docs else "")
    sup = ""
    if supersedes:
        sup = (f"\nSupersedes **`{supersedes}`**, which remains frozen and valid for the results "
               f"it contains.\nWhere the two disagree, this snapshot is the corrected one — see "
               f"`docs/experiment_b.md` §2.\n")

    warn = ""
    if drv.get("mismatch"):
        warn = (
            "\n> **Driver state at freeze time is degraded.** "
            f"{drv.get('mismatch_note')}\n>\n"
            "> The CUDA driver API still works (compute runs fine), but every NVML-sourced\n"
            "> field — power, temperature, utilisation, per-process memory — is unavailable.\n"
            "> Results inside this snapshot were captured *earlier*, while NVML was healthy,\n"
            "> which is why they contain valid `mean_power_w` and `max_temp_c` values.\n"
            "> Do not read the driver block below as the environment that produced them.\n"
        )

    return f"""# Snapshot `{name}`

Frozen at **{env.get('captured_at_utc')}** from commit
**`{git.get('commit_short')}`** (`{git.get('branch')}`{', **dirty working tree**' if git.get('dirty') else ''}).

This directory is **read-only by design**. It is a physical copy, not a reference: later sweeps
write into the live `results/` tree and regenerate summaries and figures in place, so a git tag
alone would not protect this baseline.
{sup}{warn}
## Contents

| Path | What it holds |
|---|---|
| `code/` | Harness, policies, probes, configs and `requirements.txt` as of the commit above |
{doc_row}| `results/` | {n_results} per-run JSON files, raw traces, sweep/policy summaries |
| `results/figures/` | {n_figs} generated plots, {n_paper} of them manuscript PDFs |
| `logs/` | gzipped stdout from the server and every sweep driver |
| `ENVIRONMENT.json` | Full machine-readable environment capture |
| `MANIFEST.sha256` | SHA-256 of every file above |

## Environment at freeze time

| | |
|---|---|
| Host | `{env.get('host', {}).get('hostname')}` — {env.get('host', {}).get('cpu_model')} |
| Kernel | `{env.get('host', {}).get('kernel')}` |
| GPU | {gpu.get('name')} — {gpu.get('memory_total_gib')} GiB, CC {gpu.get('compute_capability')}, {gpu.get('multi_processor_count')} SMs |
| Driver (kernel module) | `{drv.get('kernel_module_version')}` |
| Driver (userspace NVML) | `{drv.get('userspace_nvml_version')}` |
| NVML usable | `{nvml.get('available')}`{' — ' + str(nvml.get('error')) if nvml.get('error') else ''} |
| DCGM usable | `{env.get('dcgm', {}).get('available')}` |
| CUDA (torch) | `{pkg.get('torch')}` / CUDA `{env.get('torch_cuda', {}).get('torch_cuda_version')}` / cuDNN `{env.get('torch_cuda', {}).get('cudnn_version')}` |
| vLLM | `{pkg.get('vllm')}` |
| transformers | `{pkg.get('transformers')}` |
| Python | `{env.get('host', {}).get('python')}` |

Exact resolved versions of every installed package are in
`ENVIRONMENT.json` under `packages._pip_freeze`.

## Verifying integrity

```bash
cd {dest.relative_to(REPO)}
sha256sum -c MANIFEST.sha256
```

## Restoring the code

```bash
git checkout {git.get('commit_short')}      # or: git checkout {name}
uv pip install -r requirements.txt
```
"""


def build(name: str, force: bool, supersedes: str | None = None) -> Path:
    dest = REPO / "snapshots" / name
    if dest.exists():
        if not force:
            sys.exit(f"error: {dest} already exists (use --force to replace)")
        _make_writable(dest)
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    # --- code + configs ------------------------------------------------------------------
    code_dst = dest / "code"
    code_dst.mkdir()
    for d in CODE_DIRS:
        src = REPO / d
        if src.is_dir():
            shutil.copytree(src, code_dst / d, ignore=EXCLUDE)
    for f in CODE_FILES:
        if (REPO / f).is_file():
            shutil.copy2(REPO / f, code_dst / f)

    # --- interpretive documents ----------------------------------------------------------
    for d in DOC_DIRS:
        src = REPO / d
        if src.is_dir():
            shutil.copytree(src, dest / d, ignore=EXCLUDE)

    # --- results: per-run JSON, raw traces, summaries, figures ---------------------------
    res_src = REPO / "results"
    res_dst = dest / "results"
    if res_src.is_dir():
        shutil.copytree(res_src, res_dst, ignore=EXCLUDE)

    # --- logs (compressed) ---------------------------------------------------------------
    log_dst = dest / "logs"
    log_dst.mkdir()
    for f in LOG_FILES:
        src = REPO / f
        if not src.is_file():
            continue
        with src.open("rb") as fi, gzip.open(log_dst / (f + ".gz"), "wb", compresslevel=9) as fo:
            shutil.copyfileobj(fi, fo)

    # --- environment ---------------------------------------------------------------------
    sys.path.insert(0, str(REPO / "tools"))
    import capture_env  # noqa: E402  (import after path setup, by design)

    env = capture_env.capture()
    env["git"] = {
        "commit": _git("rev-parse", "HEAD"),
        "commit_short": _git("rev-parse", "--short", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "describe": _git("describe", "--tags", "--always"),
        # A dirty tree at freeze time means the snapshot does not correspond to any commit.
        "dirty": bool(_git("status", "--porcelain")),
        "status_porcelain": _git("status", "--porcelain"),
    }
    (dest / "ENVIRONMENT.json").write_text(json.dumps(env, indent=2) + "\n")

    # --- provenance README ----------------------------------------------------------------
    (dest / "README.md").write_text(_readme(name, env, dest, supersedes))

    # --- checksums over everything written so far ----------------------------------------
    manifest_path = dest / "MANIFEST.sha256"
    lines, total_bytes, count = [], 0, 0
    for p in sorted(dest.rglob("*")):
        if p.is_file() and p != manifest_path:
            rel = p.relative_to(dest)
            lines.append(f"{_sha256(p)}  {rel}")
            total_bytes += p.stat().st_size
            count += 1
    manifest_path.write_text("\n".join(lines) + "\n")

    return dest, env, count, total_bytes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name", help="snapshot name, e.g. paper-v1")
    ap.add_argument("--force", action="store_true", help="replace an existing snapshot")
    ap.add_argument("--no-freeze", action="store_true", help="skip dropping write permission")
    ap.add_argument("--supersedes", help="name of the snapshot this one replaces, e.g. paper-v1")
    args = ap.parse_args()

    dest, env, count, total_bytes = build(args.name, args.force, args.supersedes)
    if not args.no_freeze:
        _freeze_tree(dest)

    print(f"snapshot   : {dest}")
    print(f"files      : {count}")
    print(f"size       : {total_bytes / 2**20:.1f} MiB")
    print(f"git commit : {env['git']['commit_short']} (dirty={env['git']['dirty']})")
    print(f"nvml       : available={env['nvml']['available']}")
    print(f"read-only  : {not args.no_freeze}")


if __name__ == "__main__":
    main()
