# A concurrency-dependent service-rate model for continuously-batched LLM inference

*Draft for Section 7. Every numeric parameter in this document is empirically fitted from the
runs in `results/`; the derivation is separated from the fit so a reader can see which is which.*

---

## 1. Why the casual label "M/G/1-PS" is wrong

It is tempting to describe a continuously-batched LLM server as an M/G/1 processor-sharing
queue. Three of the four Kendall symbols do hold:

| Symbol | Meaning here | Holds? |
|---|---|---|
| **M** | Arrivals are Poisson(λ) | **Yes** — `common/loadgen.py` draws inter-arrival gaps from Exp(λ) with a fixed seed, open-loop |
| **G** | Service requirement is generally distributed | **Yes** — a request's work is its output length, drawn lognormal/uniform per workload class |
| **1** | One server | **Yes** — a single vLLM engine on one T4 |
| **PS** | Processor sharing | **No, not in general** |

The failure is in **PS**. Classical M/G/1-PS assumes a *fixed* total service capacity μ shared
among the N jobs in service, so each receives μ/N and the aggregate is constant at μ. A
continuously-batched LLM engine does not behave that way at low concurrency. Every engine step
advances *all* N running sequences by one token, and at small N the step time is dominated by
streaming the model weights — a cost paid once per step regardless of N. Adding a second
sequence therefore costs almost nothing: each still generates at roughly its solo rate, and the
**aggregate rate grows with N** rather than staying constant.

Only once N is large enough that the step time itself starts growing with N does the system
begin to behave like processor sharing. So the correct description is:

> An **M/G/1 queue with a concurrency-dependent service rate** μ(N), which is *batch-parallel*
> below a knee N\* and *processor-sharing* above it, subject to a hard multiprogramming limit
> N_max. In the queueing literature this is **limited processor sharing (LPS)**, not PS.

The multiprogramming limit is not a modelling convenience — it is vLLM's `max_num_seqs`,
default **256**, and Experiment A shows the engine reaching exactly that value.

## 2. The model

$$\mu(N) \;=\; \min\!\left(N\,r_0,\ \mu_{\max}\right), \qquad 0 \le N \le N_{\max}$$

with the knee

$$N^\* \;=\; \frac{\mu_{\max}}{r_0}$$

### 2.1 What *N* represents

**N is the number of sequences resident in the engine's running batch** — vLLM's
`num_requests_running` gauge. It is *not*:

- the number of requests in the system (that includes vLLM's own `num_requests_waiting` queue
  and, in Phase 2, the external admission queue);
- the offered concurrency implied by λ.

The distinction matters because our scheduling layer controls admission, so it directly
manipulates N while λ stays fixed. N is bounded above by `max_num_seqs` = 256.

### 2.2 What *r₀* represents

**r₀ is the per-sequence token generation rate in the uncontended regime**, in tokens per second
per sequence. It is the reciprocal of time-per-output-token when the batch is small enough that
adding one more sequence does not slow the others:

$$r_0 = \lim_{N \to 1^+} \frac{\mu(N)}{N} = \frac{1000}{\mathrm{TPOT}_{\text{ms}}\big|_{N \text{ small}}}$$

Physically r₀ is set by the *sequential* dependency of autoregressive decoding: one forward pass
per emitted token. At small N that pass is latency-bound — dominated by reading weights from
HBM — so per-sequence rate is nearly independent of N.

r₀ is **workload-dependent**, not a hardware constant. A workload of long generations spends
proportionally more of its life in memory-bound decode than one of short generations that keeps
re-entering the compute-bound prefill phase. This is why we fit r₀ per workload (§4) and why a
single pooled r₀ is meaningless (§5.2).

### 2.3 What μ_max represents

**μ_max is the saturated aggregate service rate** in tokens per second — the ceiling the engine
approaches once step time grows in proportion to batch size, so that adding sequences
redistributes throughput instead of adding it.

## 3. How the parameters are measured

Both are **empirically fitted**. Neither is derived from hardware specifications.

### 3.1 r₀ — two independent estimators

1. **Direct**: `r₀ = 1000 / TPOT_p50` measured at the lowest arrival rate, where N is smallest.
2. **Regression**: the slope of the fitted sub-knee segment (`tools/fit_queueing_model.py`).

Reporting both is deliberate. They rest on different assumptions — the direct estimator assumes
the lowest-λ runs really are uncontended, the regression estimator assumes the sub-knee segment
is truly linear — and their disagreement bounds how much of r₀ is an artefact of either.

### 3.2 μ_max — fitted plateau, cross-checked against a roofline

μ_max is the fitted asymptote. It is then sanity-checked against the bandwidth roofline. On a
T4 (320 GB/s HBM), a decode step must stream every weight: for Qwen2.5-1.5B in float16,

$$t_{\text{step}}^{\min} = \frac{1.54\times10^9 \times 2\,\text{B}}{320\times10^9\,\text{B/s}} \approx 9.6\ \text{ms}$$

i.e. at most ~104 steps/s. **The measured μ_max sits far below what that roof permits at large
batch**, which is itself a finding: the system is not bandwidth-saturated when it saturates.
Candidate causes — per-step Python/scheduler overhead, the speculative scorer's own cost, and
thermal throttling (§5.5) — are separable only with step-rate telemetry we do not yet collect.
The roofline is therefore used as an *upper bound that is not attained*, not as a prediction.

### 3.3 Fitting procedure

Non-linear least squares (Trust Region Reflective) on the two-parameter model, over run-level
`(N, X)` pairs where N is `num_running_mean` and X is `throughput_tok_s`. Confidence intervals
come from a **non-parametric bootstrap over runs** (2000 resamples), not over residuals, so the
run-to-run variance structure — the dominant noise source — is preserved. Runs with any failed
request are excluded: a run whose engine died mid-window reports throughput averaged over a
period when the engine was partly dead, which is not a point on the service curve at all.

## 4. Fitted values

From `tools/fit_queueing_model.py` over `results/` (95% bootstrap CIs):

| Workload | n | N range | r₀ (tok/s/seq) | μ_max (tok/s) | N\* | R² | plateau slope |
|---|---|---|---|---|---|---|---|
| `phase1_mixed` | 9 | 1.4 – 60.6 | **12.60** [11.93, 14.86] | **247.5** [240.3, 253.0] | **19.7** [16.7, 20.8] | **0.983** | −0.61 |
| `expB_qwen3b` | 18 | 0.9 – 129.2 | 4.06 [3.85, 4.59] | *470.8* [438.9, 487.6] | *116.1* [101.0, 126.2] | 0.950 | +1.73 |
| `phase2_mixed` | 19 | 3.6 – 18.2 | 50.57 [45.46, 54.35] | *451.9* [398.7, 549.3] | *8.9* [7.7, 11.8] | 0.862 | +16.7 |
| `modelval_mixed` | 17 | 0.4 – 116.8 | 8.65 [6.61, 12.31] | *355.7* [283.0, 435.3] | *41.1* [28.3, 61.0] | 0.825 | +3.01 |
| `expB_qwen1.5b` | 17 | 0.4 – 117.0 | 8.52 [6.36, 12.21] | *355.2* [281.5, 436.5] | *41.7* [28.0, 62.6] | 0.813 | +3.06 |
| `phase2_chat` | 23 | 3.6 – 11.8 | 58.23 [56.05, 61.01] | *349.2* [308.5, 387.8] | *6.0* [5.3, 6.7] | 0.479 | +30.5 |
| `phase1_validation` | 6 | 1.2 – 32.6 | 63.27 [—] | *141.2* [112.6, 170.6] | *2.2* [1.8, 2.7] | 0.372 | +3.44 |
| *pooled* | 109 | 0.4 – 129.2 | — | — | 6.9 [4.6, 8.6] | **0.229** | +1.65 |

*Italicised* μ_max and N\* are extrapolations, not measurements — see §5.1. Only `phase1_mixed`
has a plateau flat enough to identify them.

**`phase1_mixed` is the reference fit.** It is the only workload that both spans the knee
generously (N from 1.4 to 60.6, 6 points below and 3 above) and has a genuinely flat plateau
(slope −0.61 tok/s/seq, indistinguishable from zero at this noise level). R² = 0.983 and mean
absolute percentage error 12.7%.

**Two corrections to the numbers previously reported in this table.** Both follow from a run
exclusion added after Experiment B (`docs/experiment_b.md` §2). When a vLLM engine dies mid-run,
requests already streaming receive a *cleanly terminated* response — `success=True`, no error,
but a handful of tokens instead of the requested hundred. Such a run reports a plausible
concurrency with a throughput an order of magnitude too low, and the failed-request filter of
§3.3 does not catch it. Three runs across the campaign carry this signature; the discriminator
is bimodal rather than tuned, since every healthy run completes 75–100% of its requested output
and these complete 2–6%.

- `modelval_mixed` previously read R² = 0.558 with MAPE 77.3%, and its N range extended to 140.8.
  That endpoint was one such run. Removing it gives R² = 0.825, MAPE 29.1%.
- The *pooled* row previously read n = 75; it now covers 109 observations including both
  Experiment B arms. Its R² remains near 0.2, which is the point of §5.2.

## 5. Where the model breaks down

This section is the point of the exercise. The fits above are not uniformly good, and the
failures are informative.

### 5.1 μ_max is unidentified where the sweep never saturated

`phase2_mixed` and `phase2_chat` have plateau slopes of **+16.7 and +30.5 tok/s/seq** — their
"saturated" regions are still rising steeply. Those sweeps only reached N ≈ 12–18, well short of
the knee. The fitted μ_max for them is an extrapolation constrained by the functional form, not
a measurement, and the fitted N\* (6.0, 8.9) is an artefact of the N range rather than a
property of the system. **Do not quote μ_max or N\* for the Phase-2 workloads.**

Experiment B was designed partly to fix this: it sweeps λ up to the point of failure and reaches
N ≈ 117 and 129, an order of magnitude past the Phase-2 range. It does not fix it. Fitting the
local slope in the lowest and highest third of each arm's N range:

| Arm | slope, low third | slope, high third | ratio |
|---|---|---|---|
| `expB_qwen1.5b` | 9.69 | 3.06 | 0.32 |
| `expB_qwen3b` | 5.12 | 3.99 | 0.78 |

A saturated system would show a high-third slope near zero. Both are still climbing when the
sweep ends, the 3B arm having barely bent at all. **No workload in this campaign except
`phase1_mixed` reaches its plateau**, and the reason Experiment B does not is the subject of
§5.3: the engine runs out of memory before it runs out of throughput. On this device, under
speculative decoding, μ_max is not merely unmeasured — it is unreachable.

### 5.2 r₀ is not workload-invariant, so the pooled fit is meaningless

Pooling every workload gives R² = **0.216**. Fitted r₀ ranges over 8.65 – 63.27 tok/s/seq — a
factor of 7 — across workloads that differ only in prompt/output length mix. The model is a
*per-workload* law; there is no single system-wide (r₀, μ_max). Any figure or claim that pools
workloads is reporting an average of incompatible regimes.

### 5.3 The model has no memory dimension, so it cannot predict failure

This is the most important limitation. μ(N) describes *rate*, and says nothing about *survival*.
Experiment A shows the engine dying at N = 256 with a 742 MiB allocation in the speculative
decoding scorer, while KV occupancy sat at 28.4% ± 0.3%. The service-rate model predicts that
N = 256 simply yields μ = μ_max; it has no term that goes to infinity, no failure mode, and no
representation of the dense `[contracted_bs, k+1, vocab]` probability tensor whose size grows
linearly in N and is invisible to the model.

**The queueing model predicts throughput. It does not predict the boundary at which the system
stops existing.** Those are separate results and the paper must not let one imply the other. A
memory-augmented model would need a second constraint of the form

$$M_{\text{weights}} + M_{\text{KV}}(N, L) + \underbrace{c \cdot N (k{+}1) V}_{\text{spec-decode scorer}} \le M_{\text{device}}$$

whose third term — not the second — is what binds on this hardware.

**Experiment B measures c.** Across seven independent engine deaths spanning two model sizes,
two experiments and two distinct allocation sizes, the failing allocation is predicted to within
0.1% by taking c = 4 bytes — the scorer materialises its probability tensor in fp32:

$$M_{\text{scorer}} = N \cdot (k{+}1) \cdot V \cdot 4\ \text{bytes} = N \times 2.898\ \text{MiB}$$

with k+1 = 5 and V = 151,936. At vLLM's default `max_num_seqs` = 256 that is 742 MiB, which is
exactly the allocation Experiment A observed. The 1.5B sweep died at N = 220 and asked for
638 MiB, which is the same law evaluated at a different concurrency — the part that makes this a
test of the model rather than a fit to it.

Two consequences follow, and they are the reason this term deserves its own constraint rather
than a footnote. First, the allocation is **independent of model size**: both checkpoints share
V = 151,936 and the same k, so the wall sits at the same N for a 1.5B and a 3B model. Second, it
is **linear in N and invisible to μ(N)** — the service-rate model sees N = 256 and predicts
μ = μ_max, with no term that fails. The full analysis is in `docs/experiment_b.md`.

### 5.4 Jensen bias from fitting on run-averaged N

Each `(N, X)` pair uses `num_running_mean`, a time-average of an N that varies substantially
within a run. Because μ(·) is concave (a `min` of two linear functions),

$$\mathbb{E}[\mu(N)] \;\le\; \mu(\mathbb{E}[N])$$

so fitting the mean throughput against the mean concurrency **systematically overestimates**
the service rate near the knee, by an amount that grows with within-run variance of N. Runs
whose N swings widely — precisely the high-λ runs — are the most affected. Fitting against the
full telemetry time series rather than run means would remove this; the raw traces needed for
that exist for only 15 runs (`results/raw/`) and now for the Experiment B sweeps.

### 5.5 μ(N) is not stationary — the device throttles

All three Experiment A trials recorded **85–86 °C with the NVML thermal-throttle flag set**. A
throttled T4 has a lower μ_max than an unthrottled one, so μ(N) drifts *within* a run and
*across* repeats in a cell that ran back to back. The model assumes a time-invariant service
rate; the hardware does not provide one. Any cell whose `thermal_throttled` flag is set should
be treated as sampling a different (slower) service curve.

### 5.6 "Service rate" is ambiguous under speculative decoding

With n-gram speculation at k = 4 and acceptance ≈ 0.71, one engine step can emit between 1 and 5
tokens per sequence. Tokens per second, steps per second, and *accepted* tokens per second are
three different service rates with three different ceilings. This document fits **emitted output
tokens per second**, because that is what the client observes, but the memory-bandwidth roofline
of §3.2 is naturally expressed per *step*. The two are related by a factor that is itself
load-dependent, which is why the roofline comparison is stated as a bound rather than a fit.

### 5.7 Open-loop arrivals mean N is endogenous

N is not a control input in the Phase-1 runs — it is the *result* of λ, the service rate, and
admission. Regressing X on N therefore fits an equilibrium relation, not a causal response.
Phase-2's admission cap is the only place N is manipulated directly, and that is exactly where
the sweeps are too narrow to identify the plateau (§5.1). Establishing μ(N) causally would need
a closed-loop experiment that pins N and measures X.

## 6. What is theoretical and what is fitted

| Claim | Status |
|---|---|
| Arrivals are Poisson | **By construction** — the generator draws Exp(λ) gaps |
| Service requirement is generally distributed | **By construction** — output-length distributions per class |
| μ(N) is non-decreasing and concave | **Theoretical** — follows from batching amortising a fixed per-step cost |
| μ(N) has the specific form min(N·r₀, μ_max) | **Modelling assumption** — a piecewise-linear approximation; supported at R² = 0.983 on `phase1_mixed`, not supported pooled (R² = 0.216) |
| N ≤ 256 | **Architectural** — vLLM `max_num_seqs` |
| r₀ = 12.60 tok/s/seq, μ_max = 247.5 tok/s, N\* = 19.7 | **Empirically fitted** (`phase1_mixed`, bootstrap CIs in §4) |
| ~104 decode steps/s bandwidth ceiling | **Derived from hardware specs**, and *not attained* — an unmet upper bound |
| The system OOMs at N = 256 | **Empirical** (Experiment A, 3/3), and **outside this model** |

## 7. Reproducing

```bash
python tools/fit_queueing_model.py --results-dir results   # writes results/model/queueing_model_fit.json
```
