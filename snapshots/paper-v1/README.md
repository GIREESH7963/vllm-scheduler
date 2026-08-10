# Snapshot `paper-v1`

Frozen at **2026-08-10T03:11:44Z** from commit
**`89c06c6`** (`master`, **dirty working tree**).

This directory is **read-only by design**. It is a physical copy, not a reference: later sweeps
write into the live `results/` tree and regenerate summaries and figures in place, so a git tag
alone would not protect this baseline.

> **Driver state at freeze time is degraded.** loaded kernel module 580.159.03 != installed userspace 580.173.02; NVML unavailable until modules are reloaded or the host reboots
>
> The CUDA driver API still works (compute runs fine), but every NVML-sourced
> field — power, temperature, utilisation, per-process memory — is unavailable.
> Results inside this snapshot were captured *earlier*, while NVML was healthy,
> which is why they contain valid `mean_power_w` and `max_temp_c` values.
> Do not read the driver block below as the environment that produced them.

## Contents

| Path | What it holds |
|---|---|
| `code/` | Harness, policies, configs and `requirements.txt` as of the commit above |
| `results/` | 101 per-run JSON files, raw traces, sweep/policy summaries |
| `results/figures/` | 21 generated plots |
| `logs/` | gzipped stdout from the server and every sweep driver |
| `ENVIRONMENT.json` | Full machine-readable environment capture |
| `MANIFEST.sha256` | SHA-256 of every file above |

## Environment at freeze time

| | |
|---|---|
| Host | `zeronsec-ProLiant-DL360-Gen9` — Intel(R) Xeon(R) CPU E5-2696 v4 @ 2.20GHz |
| Kernel | `6.8.0-124-generic` |
| GPU | Tesla T4 — 14.5622 GiB, CC 7.5, 40 SMs |
| Driver (kernel module) | `580.159.03` |
| Driver (userspace NVML) | `580.173.02` |
| NVML usable | `False` — nvmlInit failed: RM has detected an NVML/RM version mismatch. |
| DCGM usable | `True` |
| CUDA (torch) | `2.6.0+cu124` / CUDA `12.4` / cuDNN `90100` |
| vLLM | `0.8.5.post1` |
| transformers | `4.51.3` |
| Python | `3.11.15` |

Exact resolved versions of every installed package are in
`ENVIRONMENT.json` under `packages._pip_freeze`.

## Verifying integrity

```bash
cd snapshots/paper-v1
sha256sum -c MANIFEST.sha256
```

## Restoring the code

```bash
git checkout 89c06c6      # or: git checkout paper-v1
uv pip install -r requirements.txt
```
