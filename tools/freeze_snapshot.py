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
CODE_DIRS = ("common", "configs", "phase0-setup", "phase1-harness", "phase2-policies", "tools")
CODE_FILES = ("requirements.txt",)
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


def build(name: str, force: bool) -> Path:
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
    args = ap.parse_args()

    dest, env, count, total_bytes = build(args.name, args.force)
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
