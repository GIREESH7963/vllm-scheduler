# Snapshot `paper-v2`

Frozen at **2026-08-10T13:52:20Z** from commit
**`e3d485a`** (`master`).

This directory is **read-only by design**. It is a physical copy, not a reference: later sweeps
write into the live `results/` tree and regenerate summaries and figures in place, so a git tag
alone would not protect this baseline.

Supersedes **`paper-v1`**, which remains frozen and valid for the results it contains.
Where the two disagree, this snapshot is the corrected one — see `docs/experiment_b.md` §2.

## Contents

| Path | What it holds |
|---|---|
| `code/` | Harness, policies, probes, configs and `requirements.txt` as of the commit above |
| `docs/` | `experiment_b.md`, `queueing_model.md` |
| `results/` | 143 per-run JSON files, raw traces, sweep/policy summaries |
| `results/figures/` | 27 generated plots, 10 of them manuscript PDFs |
| `logs/` | gzipped stdout from the server and every sweep driver |
| `ENVIRONMENT.json` | Full machine-readable environment capture |
| `MANIFEST.sha256` | SHA-256 of every file above |

## Environment at freeze time

| | |
|---|---|
| Host | `zeronsec-ProLiant-DL360-Gen9` — Intel(R) Xeon(R) CPU E5-2696 v4 @ 2.20GHz |
| Kernel | `6.8.0-124-generic` |
| GPU | Tesla T4 — 14.5622 GiB, CC 7.5, 40 SMs |
| Driver (kernel module) | `580.173.02` |
| Driver (userspace NVML) | `580.173.02` |
| NVML usable | `True` |
| DCGM usable | `True` |
| CUDA (torch) | `2.6.0+cu124` / CUDA `12.4` / cuDNN `90100` |
| vLLM | `0.8.5.post1` |
| transformers | `4.51.3` |
| Python | `3.11.15` |

Exact resolved versions of every installed package are in
`ENVIRONMENT.json` under `packages._pip_freeze`.

## Verifying integrity

```bash
cd snapshots/paper-v2
sha256sum -c MANIFEST.sha256
```

## Restoring the code

```bash
git checkout e3d485a      # or: git checkout paper-v2
uv pip install -r requirements.txt
```
