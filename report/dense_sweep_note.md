# Dense model-validation sweep — outcome note

**What it was.** A fine-grained, no-scheduler (admit-all) arrival-rate sweep of the *mixed* workload —
`λ ∈ {0.5, 1, 1.5, 2, 3, 4, 5}`, 3 repeats each, 55 s windows — run to get a denser
throughput/TPOT-vs-load curve across the saturation knee than the two rate-points the Phase-2 matrix
provided, with error bars. Config: `configs/modelval_mixed.yaml`. Ran on the GPU server via the
original `phase1-harness/run.py` path. Clean per-run data: `model/modelval_runs.csv` (17 valid runs).

**Honest outcome: it did *not* firm up the model fit, but it hardened two qualitative results and
surfaced a third.**

### Why it was not folded into the queueing fit
The sweep is genuinely noisy at the resolution needed for a knee fit. At a *fixed* arrival rate the
repeats scatter hard — e.g. at λ = 0.5 the three reps landed at running-population `nrun` = {0.4, 5.0,
2.5} with TPOT p50 = {18, 88, 76} ms and throughput = {24, 64, 42} tok/s. That is short-window,
few-requests-per-rate sampling noise (the harness's own summary reported throughput ±20 tok/s at
λ = 0.5). Fitting `throughput(N) = min(N·r0, μmax)` to it yields `r0 ≈ 7.5, μmax ≈ 380, N* ≈ 51` with a
poor RMSE ≈ 52 — an artifact of the noise, **not** a real contradiction of the Phase-2 fit
(`r0 ≈ 49, μmax ≈ 495, N* ≈ 10`), which is physically anchored by the measured ~21 ms single-stream
TPOT (→ 48 tok/s/seq). Folding these noisy points into `measured_runs.csv` would corrupt a clean,
anchored fit, so we keep them as a **separate recorded artifact** and do not add a `modelval_mixed`
family to `configs/phase3_model.yaml`.

(The Phase-2 numbers came through the scheduler harness's `nocap` path; this sweep came through the
original direct-harness path. The two report `num_running` on a somewhat different basis at low load —
another reason not to cross-fit them. Reconciling the two harnesses' concurrency accounting would need
the server and is out of scope here.)

### What it *did* establish (both used in report §5.1)
1. **KV stays slack far past the knee.** The admit-all sweep drove the running population to **~117
   concurrent sequences** (6× the Phase-2 range) with `num_waiting` = 0 throughout, and `kv_occupancy`
   still peaked at only **~13 %**. This is a much stronger version of the KV-slack result than the
   Phase-2 data (which reached `nrun` ≈ 18) could give, and it is consistent with the linear
   `C_kv ≈ 10³` extrapolation.
2. **The real failure mode of unbounded admission is an activation-memory OOM, not KV blocking.**
   Pushed to λ ≥ 4 sustained, admit-all let the batch exceed ~140 sequences and the vLLM engine
   **OOM-crashed** — `CUDA out of memory` on a 642 MiB allocation with the KV pool still ~13 % used, so
   the exhausted resource was per-step activation/scratch memory, not KV. With the server's default
   `max_num_seqs = 256` and no admission cap, the engine keeps admitting until activations blow the
   non-KV headroom on the 16 GB card. This is a concrete **stability** argument for the concurrency cap
   an admission layer provides — one the aggregate-SLO parity result (§4.1) does not capture. Single
   observation, unambiguous mechanism, not repeated.

### Invalid runs (excluded, kept under `results/modelval_raw/rejected/`)
- λ = 4 rep2: crash-onset run (TPOT p50 = 2981 ms, `nrun` = 141) — the engine was already dying.
- λ = 5 (all 3 reps): throughput 0, all requests failed against the dead server (post-OOM).
