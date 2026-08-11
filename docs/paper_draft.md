# KV-cache capacity is not serving capacity

*Working title. A memory allocation outside the paged KV allocator bounds continuously-batched
LLM inference, and admission control governs aggregate capacity as well as differentiation.*

---

## Status of this draft

This is a **manuscript draft assembled from the analysis documents in this repository**, not a
new analysis. Every number below is transcribed from a generated artefact and cited to it:

| Source | What it supplies |
|---|---|
| `docs/queueing_model.md` | The service-rate model, its fitted parameters, and §5's limitations |
| `docs/experiment_b.md` | Experiments A, B and F — the memory boundary and the scorer law |
| `results/stats/STATISTICS.md` | Phase-2 policy comparison and Experiment D |
| `results/expF/expF_analysis.json` | Experiment F's per-arm agreement |
| `snapshots/paper-v2/ENVIRONMENT.json` | Environment capture at freeze time |

Two conventions, because they are the difference between a defensible paper and a retracted one:

- **Measured, asserted, and derived are labelled separately.** §10 lists every claim by status.
- **Numbers are not restated from memory.** Where a figure appears here it exists in a generated
  file; where a figure does *not* exist, this draft says so rather than estimating it.

**Format.** Markdown, because this machine has no LaTeX or pandoc toolchain (`pdflatex`,
`xelatex`, `latexmk`, `tectonic`, `pandoc` all absent) and the rest of `docs/` is Markdown.
Converting to a venue template is mechanical once a venue is chosen; no venue or format was
specified, so this draft is venue-neutral. See §13 for the open items that block submission.

---

## Abstract

Continuously-batched LLM serving systems are routinely provisioned as though the paged KV cache
were the binding resource: admission control, capacity planning and autoscaling all key on KV
occupancy. We show on a single-GPU vLLM deployment that this assumption is not merely imprecise
but **inverted**. Across 16 independent engine deaths spanning two model sizes, three
experiments and two speculative depths, every failure occurred in the same allocation — a dense
`[N, k+1, V]` fp32 tensor in the speculative-decoding scorer, which lives outside the paged
allocator and is invisible to every admission signal the engine exposes. We give a closed-form
law for its size, `M = N·(k+1)·V·4` bytes, and test it on two of its three axes: it predicts the
failing allocation to within 0.1% at k = 4 and k = 7, and to within 0.5% out at N ≈ 410, while
correctly predicting that k = 2 does not fail at all at the default sequence cap. The failure is
not explained by occupancy: engines died with 37–56% of the KV
pool free, and in matched trials *survived* a full load ramp at 82.6% occupancy having *died* at
57.4% under an identical configuration. Disabling speculative decoding removes the failure
entirely at a *higher* occupancy than the one that killed the speculative arm. We further show
that the binding resource can be flipped between the scorer and the KV pool by changing a single
scheduler parameter, `max_num_seqs`, on the same model, GPU and workload — because raising the
sequence cap enlarges vLLM's activation reservation and shrinks the KV pool that the same
setting is presumed to govern. Finally, at n = 10 repeats, admission control is shown to govern
**aggregate throughput as well as service differentiation** (+178 tok/s and +0.888 SLO
attainment for an uncapped queue over FCFS, both surviving Holm correction across a
207-comparison family), contradicting the common framing in which admission policy trades
latency for fairness at fixed capacity.

> **TODO before submission.** The abstract claims a general property from a single GPU, a single
> vLLM version and a single model family. §10.1 and §13 state the scope honestly; the abstract
> must be narrowed to match before it goes out, or the generalisation experiments in §13 must be
> run. Do not submit the abstract as written above without reading §10.

---

## 1. Introduction

A continuously-batched LLM server admits requests into a running batch, advances every resident
sequence by one token per engine step, and evicts sequences as they finish. The dominant memory
consumer in this design is the paged KV cache, and the dominant operational assumption is that
KV capacity *is* serving capacity: if the pool has room, the system can take the work.

That assumption is load-bearing. It is why vLLM exposes KV occupancy as a first-class gauge, why
admission controllers throttle on it, and why capacity planning reduces to "how many sequences
fit in the pool". This paper reports that on the system we measured, it is wrong in a specific
and reproducible way, and that the failure it hides is not an edge case but the *only* way our
engine ever died.

### 1.1 Contributions

1. **A closed-form law for the binding allocation** (§7). The speculative-decoding scorer
   materialises a dense `[N, k+1, V]` fp32 probability tensor per engine step. Its size,
   `N·(k+1)·V·4` bytes, predicts the observed failing allocation to within 0.1%. The law is
   tested on N (220 → 410) and on k (2, 4, 7); the V term is read from the source, not measured,
   and we say so.
2. **A refutation of occupancy as the capacity signal** (§7.4). Deaths occurred with 37–56% of
   the KV pool free. In matched trials the engine survived at 82.6% occupancy having died at
   57.4% under an identical configuration — the relationship is not merely weak but
   non-monotonic.
3. **A mechanism control** (§7.5). With speculative decoding disabled the same ramp survives
   3/3 at 99.91–99.96% occupancy, higher than the 98.66–98.96% at which the speculative arm died
   3/3. Removing the allocation removes the failure.
4. **A demonstrated regime flip** (§8). Changing `max_num_seqs` from 256 to 1024 moves the
   binding resource from the scorer to the KV pool on the same model, GPU and workload, because
   the raised cap costs 3.1 GiB of activation reservation taken out of the KV pool.
5. **Admission control governs aggregate capacity** (§9). At n = 10, an uncapped queue beats FCFS
   by +178 tok/s *and* +0.888 SLO attainment, both surviving Holm correction over 207
   comparisons — where the n = 3 campaign had yielded a single surviving contrast in 171, and not
   a capacity one.
6. **A service-rate model and an honest account of where it fails** (§5, §6). μ(N) = min(N·r₀,
   μ_max) fits one workload at R² = 0.983 and the pooled data at R² ≈ 0.22. We report the second
   number as prominently as the first, because the model's failure to generalise is itself the
   finding that motivates §7.

### 1.2 What this paper does not claim

The failure we characterise is specific to vLLM's V0 engine (§10.2), to a single T4 (§10.1) and
to a model family with one vocabulary size (§10.3). The *mechanism* — a scorer allocation
scaling with `N × (k+1) × V`, outside the paged allocator and invisible to admission control —
is a design property whose persistence in other engines is an open empirical question that
nothing here settles.

---

## 2. Background

**Continuous batching.** Requests join and leave a running batch between engine steps rather
than being served in fixed batches. `N`, the number of resident sequences, is vLLM's
`num_requests_running` gauge, and is bounded by `max_num_seqs` (default 256).

**Paged KV cache.** Attention keys and values are stored in fixed-size blocks from a pool sized
at startup as whatever remains of the memory budget after weights and a profiled activation
peak. Occupancy of this pool is the signal operators watch.

**Speculative decoding.** A cheap draft proposes `k` tokens; the target model scores all `k+1`
positions in one pass and accepts a prefix. Our runs use vLLM's prompt-lookup n-gram drafter, so
the proposal distribution depends on the prompt rather than on a draft model. The scoring step
is where the allocation of §7 lives.

> **TODO — Related work.** This repository contains no bibliography and this draft fabricates no
> citations. The sections that must be written and cited before submission: continuous batching
> and paged attention (vLLM and successors); speculative decoding and its acceptance-rate
> literature; LLM serving admission control and SLO-aware scheduling; queueing models for
> batch-parallel servers, in particular limited processor sharing. §5.1's LPS framing in
> particular needs to be positioned against the existing LPS literature rather than presented as
> novel.

---

## 3. Experimental setup

Single-node, single-GPU. Full capture in `snapshots/paper-v2/ENVIRONMENT.json`.

| | |
|---|---|
| GPU | Tesla T4, 14.56 GiB usable, CC 7.5, 40 SMs, 70 W cap |
| Host | Intel Xeon E5-2696 v4, 88 threads, Linux 6.8.0-124-generic |
| Driver | 580.173.02 (kernel module and userspace NVML agree) |
| Stack | vLLM 0.8.5.post1, PyTorch 2.6.0+cu124, CUDA 12.4, Python 3.11.15 |
| Models | Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, float16 |
| Speculation | n-gram prompt-lookup, `k` = 2/4/7, `prompt_lookup` 2–5 |

Arrivals are Poisson with a fixed seed, open-loop (`common/loadgen.py`); service requirement is
the output length, drawn per workload class. Telemetry is sampled per stage: `num_running`,
`num_waiting`, KV occupancy, memory, temperature, power and the NVML throttle flag.

**The device throttles.** Nearly every run in the campaign sat at 85–89 °C with the NVML
thermal-throttle flag set. This is a passively cooled T4 in a dense chassis. Every absolute rate
in this paper is therefore depressed relative to an unthrottled device, and **only paired and
ratio comparisons should be read as hardware-independent**. This is stated once here and again
in §10.4 because it conditions every throughput number in §5.

---

## 4. Why "M/G/1-PS" is the wrong label

Three of the four Kendall symbols hold: arrivals are Poisson by construction, service
requirements are generally distributed by construction, and there is one server. **Processor
sharing does not hold.** Classical M/G/1-PS assumes a fixed capacity μ shared among N jobs, each
receiving μ/N. A continuously-batched engine does not behave that way at low concurrency: every
step advances all N sequences, and at small N the step time is dominated by streaming weights, a
cost paid once per step regardless of N. Adding a sequence therefore costs almost nothing and
the **aggregate rate grows with N**.

The correct description is an M/G/1 queue with a concurrency-dependent service rate, batch-
parallel below a knee and processor-sharing above it, subject to a hard multiprogramming limit —
**limited processor sharing**, not PS. The multiprogramming limit is not a modelling convenience:
it is `max_num_seqs`, and §8 shows it is also, unexpectedly, a memory-allocation parameter.

---

## 5. A concurrency-dependent service-rate model

$$\mu(N) = \min(N\,r_0,\ \mu_{\max}), \qquad N^* = \mu_{\max}/r_0, \qquad 0 \le N \le N_{\max}$$

`r₀` is the per-sequence generation rate in the uncontended regime; `μ_max` is the saturated
aggregate rate. Both are **empirically fitted**, neither derived from hardware specifications.
Fitting is non-linear least squares over run-level `(N, X)` pairs, with 95% CIs from a
non-parametric bootstrap **over runs** (2000 resamples), so the run-to-run variance structure —
the dominant noise source — is preserved.

From `docs/queueing_model.md` §4:

| Workload | n | N range | r₀ (tok/s/seq) | μ_max (tok/s) | N\* | R² |
|---|---|---|---|---|---|---|
| `phase1_mixed` | 9 | 1.4 – 60.6 | **12.60** [11.93, 14.86] | **247.5** [240.3, 253.0] | **19.7** [16.7, 20.8] | **0.983** |
| `expB_qwen3b` | 18 | 0.9 – 129.2 | 4.06 [3.85, 4.59] | *470.8* | *116.1* | 0.950 |
| `phase2_mixed` | 19 | 3.6 – 18.2 | 50.57 [45.46, 54.35] | *451.9* | *8.9* | 0.862 |
| `modelval_mixed` | 17 | 0.4 – 116.8 | 8.65 [6.61, 12.31] | *355.7* | *41.1* | 0.825 |
| `expB_qwen1.5b` | 17 | 0.4 – 117.0 | 8.52 [6.36, 12.21] | *355.2* | *41.7* | 0.813 |
| `phase2_chat` | 23 | 3.6 – 11.8 | 58.23 [56.05, 61.01] | *349.2* | *6.0* | 0.479 |
| `phase1_validation` | 6 | 1.2 – 32.6 | 63.27 | *141.2* | *2.2* | 0.372 |
| *pooled* | 109 | 0.4 – 129.2 | — | — | 6.9 | **0.229** |

*Italicised μ_max and N\* are extrapolations, not measurements (§6.1).* Only `phase1_mixed`
both spans the knee generously and has a flat plateau (slope −0.61 tok/s/seq, indistinguishable
from zero at this noise level; MAPE 12.7%). It is the reference fit.

---

## 6. Where the rate model breaks

This section is the point of the exercise, and it motivates §7.

**6.1 μ_max is unidentified wherever the sweep did not saturate.** `phase2_mixed` and
`phase2_chat` have plateau slopes of +16.7 and +30.5 tok/s/seq — their "saturated" regions are
still climbing steeply, because those sweeps reached only N ≈ 12–18. Their fitted μ_max and N\*
are artefacts of the N range. Experiment B was designed to fix this by sweeping to failure at
N ≈ 117 and 129, and **it does not**: high-third slopes are 3.06 and 3.99 tok/s/seq, still
rising. The reason is §7 — *the engine runs out of memory before it runs out of throughput.* On
this device, under speculative decoding, μ_max is not merely unmeasured but unreachable.

**6.2 r₀ is not workload-invariant, so the pooled fit is meaningless.** Fitted r₀ spans
8.65–63.27 tok/s/seq — a factor of 7 — across workloads differing only in prompt/output length
mix. Pooling gives R² ≈ 0.22. The model is a *per-workload* law; any claim that pools workloads
averages incompatible regimes.

**6.3 The model has no memory dimension, so it cannot predict failure.** μ(N) describes rate and
says nothing about survival. It sees N = 256, predicts μ = μ_max, and has no term that fails.
The remainder of this paper is about the term it is missing.

Additional limitations — Jensen bias from fitting on run-averaged N, non-stationarity from
thermal throttling, the ambiguity of "service rate" under speculation, and the endogeneity of N
under open-loop arrivals — are carried in `docs/queueing_model.md` §5.4–5.7 and summarised in
§10.

---

## 7. The memory boundary

### 7.1 The allocation

Every OOM in this campaign — **16 independent engine deaths across two model sizes and three
experiments** — failed at the same source line: `batch_expansion.py:227` in `_contract_batch`,
reached via `spec_decode_worker.py:794` `score_proposals`. At that site the scorer materialises

```python
226:  all_probs    = target_probs.new_zeros(*all_tokens.shape, self._vocab_size)
227:  all_logprobs = target_logprobs.new_full(size=all_probs.shape, ...)
```

two dense `[N, k+1, V]` fp32 tensors back to back, where `all_tokens.shape` is
`(contracted_bs, k+1)`. This yields the law

$$M_{\text{scorer}} = N \cdot (k{+}1) \cdot V \cdot 4\ \text{bytes}$$

**Every observed OOM reports line 227**, so line 226 had already succeeded and the *second*
allocation failed. Transient peak demand at this site is therefore **twice** the reported failed
allocation. This does not affect the law as tested — `M_scorer` predicts the size of the tensor
that failed, and our test statistic is MiB per sequence computed from that — but it does affect
any prediction of the *concurrency* at which death occurs, which depends on total headroom. This
is a source reading, not an experiment, and is reported as such.

### 7.2 The law on the N axis

With k+1 = 5 and V = 151,936, the law gives **2.898 MiB per sequence**. Against the ten deaths
of Experiments A and B (`docs/experiment_b.md` §5):

| Arm | N at failure | Predicted | Reported | Error |
|---|---|---|---|---|
| `1.5b` sweep | 220 | 638 MiB | 638 MiB | −0.1% |
| `3b` sweep | 256 | 742 MiB | 742 MiB | −0.0% |
| 8 further probe trials | 256 | 742 MiB | 742 MiB | −0.0% |

The 1.5B sweep died at lower concurrency and asked for correspondingly *less* memory, which is
what makes this a test rather than a fit: the failing allocation tracks N, not model size.

### 7.3 The law on the k axis (Experiment F)

The N-axis test cannot identify the scorer specifically — any per-sequence allocation would look
linear in N. The `(k+1)` factor is the discriminating term, and the one most likely to have
taken another form, since how many speculative tokens are actually scored per step is a
scheduler decision rather than simply the configured `k`. Experiment F varies it at fixed model,
GPU, ramp and seed (`results/expF/expF_analysis.json`, **Figure 11**):

| Arm | k | Trials | Died | MiB/seq measured | Predicted | Error |
|---|---|---|---|---|---|---|
| Experiment B anchor | 4 | — | — | 2.898 | 2.898 | (anchor) |
| `k2_cap256` | 2 | 3 | 0 | — (no death to measure) | 1.739 | — |
| `k7_cap256` | 7 | 3 | 3 | 4.642 (range 4.640–4.645) | 4.637 | **+0.1%** |

The ratio to the k = 4 anchor is **1.602 observed against 1.600 predicted** — 8/5, fixed by the
tensor's shape before the arm ran. The k = 2 arm is the cheap falsification the law had to
survive: at 1.739 MiB/seq its boundary lies past the sequence cap, so the engine cannot reach it,
and it did not — surviving 3/3 while reaching N = 256 and 83.1% occupancy. A law that
over-predicted would have killed k = 2; one that under-predicted would have spared k = 7.

### 7.4 Occupancy does not explain the failures

The k = 7 deaths came at N = 186, 187 and 256 against a cap of 256, with KV occupancy at the
failing step of **43.6%, 43.6% and 63.3%**, against an estimated KV ceiling of N ≈ 363 at that
cap. Up to 37% of the pool was free at the moment of death, so neither the cap nor exhaustion
accounts for it.

We state this on occupancy rather than concurrency deliberately: one of the three deaths lands
on the cap itself, so a concurrency-only argument would not carry.

The stronger form comes from the reproduction trials (`docs/experiment_b.md` §8). The 3B
boundary is stochastic — 5 of 10 probe trials died, a 50% rate with a Wilson 95% interval of
[24%, 76%] — and the survivors invert the ordering that any occupancy-based account requires:

| Arm | Peak KV when it died | Peak KV when it survived |
|---|---|---|
| `1.5b` | 28.2%, 28.5%, 28.7% | — |
| `3b` | 57.4%, 59.1%, 59.5%, 59.6%, 60.9% | 80.7%, 80.8%, 80.9%, 81.3%, 82.6% |

The engine **survived a full ramp at 82.6% occupancy having died at 57.4%** under an identical
configuration. Occupancy is not merely a poor predictor of failure here; it is not monotonically
related to it. This needs no model — two runs of one configuration, one dead at low occupancy
and one alive at high.

The stochasticity is consistent with the mechanism: the tensor is requested once per engine
step, and whether a request of that size succeeds depends on the state of the caching allocator
at that instant. Reported free memory at failure sits close to, but not below, the allocation
size.

### 7.5 The mechanism control

Experiment F's fourth step re-ran the speculative-decoding contrast at `max_num_seqs` = 512, a
cap chosen from a calibration sweep rather than guessed. Both arms share model, cap, ramp and
seed; only `--speculative-config` differs.

| Arm | Speculative decoding | Trials | Died | Peak N | Peak KV |
|---|---|---|---|---|---|
| spec on | on | 3 | 3 | 410 | 98.96% |
| spec off | off | 3 | **0** | 402 | 99.96% |

The spec-on arm died 3/3 at **2.884 MiB/seq against 2.898 predicted (−0.5%)**, at allocations of
1167–1188 MiB — larger than anything in §7.2's table — and at N ≈ 410, extending the law's
tested range from 220–256 out to 410.

The spec-off arm **died 0/3**, riding the ramp to λ = 8 at 99.91–99.96% occupancy — *higher* than
the 98.66–98.96% at which the spec-on arm died — and kept serving. Remove the allocation the law
names and the failure disappears, under a load that drives the same engine to a fuller KV pool
than the one that killed it. The mechanism is not merely consistent with the deaths; it is
necessary for them.

> **These particular deaths are not low-occupancy ones.** At cap 512 the spec-on arm reaches
> N ≈ 410 and a nearly full pool before the scorer allocation fails, so this pair does not
> demonstrate the low-KV OOM of §7.2 and §7.4. It demonstrates the *mechanism*, against a matched
> control. The two arms do different jobs: k = 7 shows the failure arriving with a third of the
> pool free; this pair shows it not arriving at all once the scorer is gone.

### 7.6 What moves with model size, and what does not

Fitting occupancy = a·N through the origin (bootstrap over runs, 2000 resamples): KV cost per
sequence is 0.1103% for the 1.5B and 0.2501% for the 3B, a **ratio of 2.27× [2.17, 2.34]**. That
ratio is predicted rather than merely observed, from two numbers neither of which came from the
sweep: per-token KV cost rises 1.29× (36 layers against 28) and the pool it lands in shrinks
1.72× (fp16 weights of 5.79 GiB against 2.89 GiB leave 4.74 GiB against 8.15 GiB of KV). Together
they predict **2.21×**, inside the measured interval; vLLM's own reported maximum-concurrency
ratio (2.21×) agrees independently.

The scorer allocation, meanwhile, does not move with model size at all. **This is the bottleneck
moving**: the wall sits at the same N for both models, but the 3B engine arrives there having
spent twice as much of a smaller pool on KV. Extrapolating, the 3B arm would exhaust its pool at
N ≈ 400, beyond the cap of 256 — which is why the scorer wall still arrives first. The margin is
about 1.6× and closes from both ends at once. A modestly larger model on this device crosses
over, at which point the failure changes identity and becomes a genuine KV exhaustion. **The
claim is specific to a regime, and Experiment B locates its edge.**

Separately, speculative-decoding acceptance falls from 0.723 (1.5B, n = 17) to 0.589 (3B,
n = 18), Hedges' g = −5.66, Welch p = 1.33e-17. The draft is prompt-lookup n-gram in both arms,
so the proposal distribution is identical; the larger target model simply agrees less often. The
cost is paid twice — fewer accepted tokens per step, and the same scorer tensor allocated to
score them.

---

## 8. The regime flip: `max_num_seqs` is a memory parameter

The binding resource can be moved between the scorer and the KV pool by changing one scheduler
parameter, on the same model, GPU and workload. vLLM sizes its memory-profiling pass at
`max_num_seqs` and reserves the resulting activation peak, leaving the KV pool the remainder
(`results/expF/calibration.json`):

| `max_num_seqs` | Activation reserve | KV pool | Estimated KV-bound N | Regime |
|---|---|---|---|---|
| 256 | 2.52 GiB | 4.74 GiB | ≈ 363 | **scorer-bound** — dies at N = 256, KV 57–60% |
| 512 | 2.89 GiB | 4.38 GiB | ≈ 336 | scorer-bound |
| 768 | 4.26 GiB | 3.00 GiB | ≈ 230 | — |
| 1024 | 5.64 GiB | 1.63 GiB | ≈ 125 | **KV-bound** — survives, KV ≈ 99%, N ≈ 125 |

Raising the cap by 4× costs 3.1 GiB of activation reservation, taken directly out of the pool the
same setting is presumed to govern. **Raising the admission limit therefore halves the
concurrency the engine can actually sustain**, and flips which resource binds.

This has a methodological consequence we report because it cost us two experiments: an earlier
attempt at the speculative-decoding contrast used `max_num_seqs` = 1024 and measured the
KV-bound row rather than the intended question. The cap in §7.5 was chosen from the calibration
sweep above rather than guessed, for exactly this reason.

---

## 9. Admission control governs aggregate capacity, not only differentiation

Phase 2 compares five admission policies (`nocap`, `fcfs`, `srpt`, `edf`, `adaptive`) across
workloads and arrival rates. Intervals are two-sided 95% t-intervals; effect sizes are Hedges' g
with the small-sample correction; contrasts are Welch's t, Holm-adjusted across the **whole
family of 207 comparisons** (`results/stats/STATISTICS.md`).

**At n = 3 the comparison had almost no resolving power.** Before Experiment D, with every cell
at n = 3, exactly **1 of 171** contrasts survived Holm — and it was a queue-wait result, not a
capacity one (`results/stats/STATISTICS.md` at commit `d9602c6`). Repeating the decisive cell at
n = 10 (`phase2_mixed_n10`, λ = 4) changed that: of 207 comparisons, 32 are significant before
correction and **9 survive Holm**. The surviving contrasts include:

| Contrast | Metric | Effect | g | p_holm |
|---|---|---|---|---|
| `nocap` vs `fcfs` | SLO attainment | +0.888 | +9.24 | 0.0000 |
| `nocap` vs `fcfs` | Queue wait p99 | −5.78e3 ms | −15.52 | 0.0000 |
| `nocap` vs `fcfs` | **Throughput** | **+178 tok/s** | +2.67 | 0.0088 |
| `edf` vs `fcfs` | SLO attainment | +0.412 | +3.69 | 0.0000 |
| `srpt` vs `fcfs` | SLO attainment | +0.397 | +3.78 | 0.0000 |

The throughput row is the one that matters for framing. Admission policy is commonly described
as trading latency or fairness *at fixed capacity*. Here an uncapped queue gains **both** +0.888
SLO attainment **and** +178 tok/s (287 → 464) over FCFS, with both effects surviving the
conservative correction. **Admission control governs aggregate capacity and differentiation
together**, and the two would decouple only above the saturation knee — which §6.1 shows this
hardware cannot reach before failing.

---

## 10. Threats to validity

**10.1 Single device, single version, single model family.** One passively cooled T4, vLLM
0.8.5.post1, and two Qwen2.5 checkpoints. This is the weakest axis of the paper and the first
thing a referee will raise. Nothing here establishes that the constants generalise; the
*mechanism* is a design property, but its magnitude on other hardware is unmeasured.

**10.2 The failing code path does not exist in vLLM V1.** Read from the installed source:
`vllm/v1/spec_decode/` contains `eagle.py`, `ngram_proposer.py`, `metadata.py`, `metrics.py` and
`utils.py` — and no `batch_expansion.py`. V1's equivalent, `vllm/v1/sample/rejection_sampler.py`,
operates on a flattened 2-D `(num_tokens, vocab_size)` tensor rather than a padded 3-D
`[N, k+1, V]` expansion, and shows no sign of the doubled probs/logprobs pair. The honest framing
is that **the specific failure is V0-specific, while the mechanism is a design property whose
persistence in V1 is an open empirical question**: `num_tokens` in V1 is still ≈ N·(k+1) in the
worst case, so the magnitude may well survive the refactor. Nothing here settles that, and only
a V1 run would. This was a source read, not an experiment.

**10.3 The V term of the law is asserted, not measured.** Every death in the campaign ran a
Qwen2.5 checkpoint at V = 151,936. The vocabulary dimension is read off the allocation site —
`new_zeros(*all_tokens.shape, self._vocab_size)` — which is a direct reading of the code that
fails, not an inference from the failure, but it is **not an independent measurement and must
not be reported as one**. Consequently "the allocation is model-size independent" is
demonstrated on two models that agree on precisely the parameter which would have made it
size-*dependent*. Measuring V requires a model from a different family, which changes layer
count, KV cost per token and activation profile simultaneously — so a disagreement could not be
attributed to V. §13 gives the concrete blocked experiment.

**10.4 Thermal throttling.** 85–89 °C with the throttle flag set through nearly the whole
campaign. Absolute rates are depressed; only paired and ratio comparisons are safe. Repeats
within a cell ran back to back, so a thermal excursion is shared within a cell rather than
averaged out.

**10.5 Statistical scope.** Holm correction assumes the family is the one reported. Adding
metrics later without re-running the correction reintroduces the multiplicity it removes. Phase 2
outside the n = 10 cell remains at n = 3, where intervals are wide and effect sizes imprecise.

**10.6 Model-fitting caveats.** Jensen bias from fitting on run-averaged N systematically
overestimates the service rate near the knee (μ is concave); N is endogenous under open-loop
arrivals, so regressing X on N fits an equilibrium relation rather than a causal response; and
"service rate" is ambiguous under speculation, where emitted tokens/s, steps/s and accepted
tokens/s are three different quantities with three different ceilings.

**10.7 A silent data-corruption artefact, and a corrected published number.** When the engine
dies mid-window, requests already streaming receive a **cleanly terminated** response —
`success=True`, HTTP 200, no error — carrying roughly 5 tokens instead of the ~134 requested.
Such runs report a plausible concurrency with a throughput an order of magnitude too low and are
silently admitted by any filter that checks only the failed-request count. The output-length
histogram spikes at multiples of `k+1`, because the stream ends on a speculative chunk boundary.
Runs are now excluded on `actual_over_requested < 0.4`; the separation is bimodal rather than
tuned, with every healthy run in the campaign in [0.75, 1.00] and every affected one in
[0.02, 0.06]. **This affected a previously published number**: the same artefact is present in
the frozen `paper-v1` campaign, in `modelval_mixed` λ=4 rep2, and was included in that fit. With
it removed, that workload's fit improves from R² = 0.558 to 0.825 and its MAPE from 77.3% to
29.1%. Any reader comparing against `paper-v1` should use the corrected figure.

---

## 11. Claim status table

Because the distinction is easy to lose in prose, every substantive claim is labelled:

| Claim | Status |
|---|---|
| Arrivals Poisson; service requirement generally distributed | **By construction** |
| N ≤ `max_num_seqs` | **Architectural** |
| μ(N) non-decreasing and concave | **Theoretical** — batching amortises a fixed per-step cost |
| μ(N) = min(N·r₀, μ_max) | **Modelling assumption** — R² = 0.983 on `phase1_mixed`, ≈ 0.22 pooled |
| r₀ = 12.60, μ_max = 247.5, N\* = 19.7 | **Empirically fitted** (`phase1_mixed`) |
| ~104 decode steps/s bandwidth ceiling | **Derived from specs, and not attained** |
| The engine OOMs in the scorer at low KV occupancy | **Empirical** — 16 deaths, and outside the rate model |
| `M = N·(k+1)·V·4` predicts the failing allocation | **Empirical on N and (k+1)**; V asserted (§10.3) |
| Peak demand at the site is 2× the reported allocation | **Source reading**, not measured |
| Removing speculation removes the failure | **Empirical** — 0/3 vs 3/3 at matched cap and higher occupancy |
| Raising `max_num_seqs` flips the binding resource | **Empirical** — calibration sweep + both regimes observed |
| Admission control governs aggregate capacity | **Empirical** — n = 10, survives Holm over 207 comparisons |
| The mechanism persists in vLLM V1 | **Open** — source suggests the path is gone; unmeasured |

---

## 12. Reproducibility

Two frozen snapshots (`snapshots/paper-v1`, `snapshots/paper-v2`) are physical copies, not git
tags, because later sweeps regenerate summaries and figures in place. Each carries code,
results, figures, gzipped driver logs, a full `ENVIRONMENT.json` and a `MANIFEST.sha256` over
every file. Where the two disagree, `paper-v2` is the corrected one (§10.7).

```bash
python tools/fit_queueing_model.py --results-dir results   # service-rate fits
python tools/analyze_expB.py                               # experiments A, B and F -> docs/experiment_b.md
python tools/analyze_expF.py                               # experiment F arms
python tools/analyze_stats.py                              # phase-2 policy statistics
python tools/make_paper_figures.py                         # figure set, including Figure 11
```

`docs/experiment_b.md` is generated, not hand-edited: it regenerates byte-identically from the
committed results, and degrades cleanly to its pre-F content if `results/expF` is absent.

---

## 13. Open items before submission

**Blocking:**

1. **Related work and bibliography do not exist** (§2). No citations have been written, and none
   should be invented.
2. **Narrow the abstract** to the scope §10 actually supports, or run the generalisation
   experiments below.
3. **Choose a venue and format.** No LaTeX toolchain is installed on this machine. Prior
   assessment of realistic venues: MLSys (competitive), IEEE TPDS, ACM TOMPECS, Performance
   Evaluation, JPDC, FGCS, Cluster Computing, IEEE Access; a workshop such as EuroMLSys or
   HotInfra first is reasonable. Not OSDI/SOSP/NSDI/ASPLOS — single GPU, no system built, narrow
   mechanism.

**Would most raise the ceiling, cheapest first:**

4. **A different-vocabulary model** closes §10.3, the last untested term of the law. The intended
   arm is `meta-llama/Llama-3.2-1B-Instruct` (V = 128,256, predicting 2.446 MiB/seq and 656.7 MiB
   at N = 256 against Qwen's 741.9). **Currently blocked**: the repo is gated, `hf_hub_download`
   returns `GatedRepoError` 401, and there is no HF token on the machine. Accepting the licence
   and exporting `HF_TOKEN` unblocks it. Llama-3.2-1B is the right choice because the V test needs
   a model that *reaches* high concurrency — a cached alternative with a larger vocabulary
   (Phi-4-mini, V = 200,064) costs 128 KiB/token, plateaus near N ≈ 61, goes KV-bound and never
   reaches the scorer at all.
5. **A second GPU class** kills the T4-specific objection (§10.1).
6. **A current vLLM version** kills the stale-bug objection and answers §10.2 directly.

**Non-blocking discrepancy noted while assembling this draft:** `docs/queueing_model.md` reports
the pooled R² as **0.229** in its §4 table and **0.216** in its §5.2 text. Both round to ≈ 0.22
and neither changes any conclusion, but one of them is stale and they should be reconciled before
either is quoted.
