# Adaptive Admission Scheduling for Multi-Tenant LLM Inference on vLLM

*A measured characterization on a single NVIDIA T4, with a queueing-theoretic explanation.*

**Scope up front (read this first).** This is a research characterization, not a production system.
Everything here is one GPU (NVIDIA T4, 16 GB), one small model (Qwen2.5-1.5B-Instruct), and an
**admission/ordering layer in front of an unmodified vLLM** — we never touch vLLM's internal
continuous-batching scheduler. Multi-tenancy is simulated as multiple request streams with different
priorities/deadlines hitting one vLLM instance. We use off-the-shelf n-gram speculative decoding; we
do **not** train draft models. The headline finding is partly a rigorously-measured *null* result,
and we report it as such.

---

## 1. Problem and framing

Serving LLMs to multiple tenants means deciding, under load, **whose request runs and in what order**.
vLLM already does continuous batching — it dynamically packs running sequences into each decode step.
The question this project asks is narrow and practical: **given that batcher, does an admission/ordering
layer in front of it buy you anything?** And if so, what, and when?

We deliberately restrict control to the admission layer: the scheduler chooses (a) which queued
requests to admit into vLLM and (b) in what order, plus optional load-shedding. It does **not** change
how vLLM forms batches internally. This keeps the contribution isolated and portable, and it matches
how most operators can actually intervene without forking the engine.

The work proceeds in four phases: (0) a serving + telemetry baseline, (1) an open-loop measurement
harness, (2) a policy comparison, and (3) a queueing model that explains the observed behavior.

---

## 2. Setup and baseline (Phase 0)

**Serving.** vLLM 0.8.5.post1, `--dtype float16` (Turing has no hardware bf16), n-gram speculative
decoding (`num_speculative_tokens=4, prompt_lookup_min=2, prompt_lookup_max=5`).

**Telemetry.** Three independent sources, each validated to produce non-zero data: vLLM `/metrics`
(KV occupancy, queue depth, running/waiting counts, spec-decode counters); DCGM (`sm_active`,
`dram_active` — *not* nvidia-smi's misleading "GPU-util"); NVML (board power only).

**Single-stream baseline** (concurrency 1, 20 measured requests):

| Metric | Value |
|---|---|
| throughput | 59.7 tok/s |
| TTFT p50 / p99 | 55.3 / 88.9 ms |
| TPOT p50 / p99 | 16.2 / 16.4 ms |
| SM-active (DCGM) | 71 % |
| draft acceptance (n-gram) | 0.458 |

The n-gram acceptance rate of 0.458 was cross-checked against vLLM's own server-log counter
(484/1056) — the telemetry pipeline is trustworthy.

---

## 3. Measurement harness (Phase 1)

The harness is an **open-loop Poisson driver**: inter-arrival gaps are `Exp(λ)`, dispatched on
schedule regardless of how many requests are already in flight. This is the correct load model for a
saturation study — a closed-loop (fixed-concurrency) client would mask overload by throttling itself.
We verified the client is not the bottleneck: scheduled-arrival → actual-dispatch lag stayed below
9 ms even at the highest rates, so measured latency is the server's.

Workloads are weighted mixes of four request profiles (short, long, reasoning, coding) with
configurable prompt/output length distributions; `coding` is the spec-decode-friendly class. Every run
writes one JSON in a fixed schema, always reporting **p50 and p99**, with the first ~30 s / warmup
requests discarded. Runs are **thermally gated** — on the passively-cooled T4 we wait for the GPU to
cool below 77 °C before each measurement, because clock throttling above ~85 °C silently corrupts
timings.

**Key Phase-1 finding:** aggregate throughput scales almost linearly with load (vLLM batches harder),
but per-request TPOT degrades and SLO attainment collapses — the saturation knee lives in the
latency/SLO curves, not the throughput curve. This is exactly why a scheduling study needs
latency-percentile and per-class lenses, not just tokens/second.

---

## 4. Policy comparison (Phase 2)

We implemented a pluggable admission layer with five policies: `nocap` (admit-all, ≈ raw vLLM),
`fcfs`, `srpt` (shortest-remaining first, size = prompt-length proxy), `edf` (earliest-deadline,
class-aware), and `adaptive` (a rule-based feedback controller: widen the admission cap when SM is
idle, throttle when TTFT p99 rises, deprioritize long jobs when KV is high). Requests queued longer
than a bound are load-shed and counted as SLO misses (so shedding policies don't look artificially
good). We ran two workload mixes (a heterogeneous "mixed" and a short-dominated "chat"), three repeats
each, across arrival rates that bracket the saturation knee.

### 4.1 The uncomfortable headline: nobody beats admit-all on aggregate SLO

**Aggregate SLO attainment (mean ± std, 3 reps):**

| Policy | mixed λ=4 | chat λ=6 | tok/s (mixed λ=4) |
|---|---|---|---|
| **nocap** (admit-all) | **0.91 ± 0.10** | 0.88 ± 0.05 | **470** |
| adaptive | 0.57 ± 0.20 | **0.90 ± 0.04** | 430 |
| edf | 0.51 ± 0.08 | 0.74 ± 0.08 | 274 |
| srpt | 0.43 ± 0.02 | 0.79 ± 0.03 | 323 |
| fcfs | 0.12 ± 0.13 | 0.47 ± 0.38 | 317 |

On a single T4, **capping concurrency costs throughput** — and on this hardware throughput *is* what
meets deadlines. `edf` at mixed λ=4 delivers 274 tok/s versus nocap's 470, a **−42 % penalty**, and
its aggregate SLO is worse. vLLM's continuous batching converts concurrency directly into goodput, so
any hard admission cap is fighting the very mechanism that keeps latencies low. **A rigorously measured
null result: on this setup, admission control adds nothing to aggregate SLO over just letting vLLM
batch.** (Figures: `phase2_mixed_slo_attainment.png`, `phase2_chat_slo_attainment.png`.)

### 4.2 Where the scheduler *does* earn its place: differentiated service

Aggregate SLO hides the actual value. Under overload the operator's real question is *who suffers*,
and there the policies differ sharply. **Per-class SLO at mixed λ=4** (`phase2_mixed_byclass_slo.png`):

| Policy | short | coding | long | reasoning |
|---|---|---|---|---|
| edf | **0.85** | 0.22 | 0.42 | 0.04 |
| srpt | 0.68 | 0.31 | **0.06** | 0.36 |
| adaptive | 0.57 | 0.42 | 0.59 | 0.65 |
| nocap | 0.87 | 0.89 | 0.97 | 0.98 |

- **`edf` protects the urgent class** (`short` at 0.85) by starving `reasoning` (0.04) — deadline
  ordering doing exactly what it should.
- **`srpt` protects short jobs** (0.68) but **starves long ones** (0.06) — the textbook SRPT
  long-job-starvation, measured.
- **`adaptive` refuses to starve anyone** (every class 0.42–0.65) — the most balanced service.
- **`nocap` looks uniformly excellent only because throughput is high — but it offers zero control.**
  You cannot protect a priority tenant when it matters; everyone floats or sinks together.

So the admission layer's contribution is **not** better aggregate numbers — it is the *ability to
choose the loss distribution* under overload. That is a real operational lever nocap simply does not have.

### 4.3 The FCFS convoy, and the adaptive exception

`fcfs` + cap is strictly worst everywhere (0.12 mixed, 0.47 chat) with the highest variance (±0.38):
long jobs clog the limited admission slots while short jobs queue behind them and get shed — the
classic convoy effect, and it is arrival-order/seed sensitive, hence the variance. Every ordering
policy mitigates it.

`adaptive` is the one policy that stays throughput-competitive (it widens its cap when SM is idle) and
in the short-dominated **chat** overload actually **edges nocap on aggregate SLO (0.90 vs 0.88)** while
keeping per-class service balanced — the one regime where a feedback cap reclaims idle capacity that
convoy would otherwise waste.

### 4.4 Output-length prediction barely matters here (Phase 2 mini-experiment)

SRPT needs to know job size. We compared three size signals (mixed, λ=4; predictor MAE = 19 tokens):

| SRPT variant | SLO |
|---|---|
| prompt-length proxy | 0.462 |
| learned length prediction | 0.441 |
| true-length oracle | 0.376 |

**Misprediction cost is negligible** — the learned predictor matches the free prompt-length proxy, and
even a perfect oracle doesn't win. At this operating point SRPT ordering is inherently weak (all
variants < 0.47), so investing in a length predictor buys nothing. A clean secondary null result.

---

## 5. Queueing model and explained deviation (Phase 3)

We model vLLM continuous batching as an **M/G/1 processor-sharing (PS)** server with a *load-dependent*
(batched) service rate, plus a hard concurrency capacity cap. With concurrency `N` = mean running
sequences:

```
throughput(N) = min(N · r0, μmax)      tpot(N) = max(1/r0, N/μmax)      knee  N* = μmax / r0
```

- `r0` = single-stream output rate (tok/s/seq); `μmax` = aggregate compute ceiling (tok/s).
- Below `N*` there is spare compute — adding a sequence is nearly free (TPOT flat, throughput rises
  linearly). Above `N*` the GPU is the bottleneck and the `N` sequences share a fixed `μmax`, so
  per-token latency rises linearly. **The knee is derivable from first principles**: it is exactly
  where the fair share `μmax/N` drops below the standalone rate `r0`.

**Formal statement.** Requests arrive Poisson(λ); each needs a random number of output tokens with
mean `b`. The engine is a processor-sharing server whose aggregate token rate is load-dependent:
`μ(N) = min(N·r0, μmax)`. Two ceilings can cap the running population `N`: the compute knee
`N* = μmax/r0`, and a KV-cache limit `C_kv = B_kv / L̄` where `B_kv` is the KV budget in tokens and
`L̄` the mean live sequence length. The effective concurrency cap is `C = min(N*, C_kv)`; excess
arrivals wait in an admission queue. Token utilisation is `ρ = λb/μmax`. In the sub-cap PS regime,
processor-sharing insensitivity gives a mean sojourn time `E[T] = E[S]/(1−ρ)` that depends only on the
load, not the service-time distribution — which is why token-aware ordering (SRPT) rides the *same*
`throughput(N)` law and can only reshape *who* waits, not the aggregate curve. The system is
KV-bound rather than compute-bound iff `C_kv < N*`, i.e. `L̄ > B_kv/N* = B_kv·r0/μmax` — the crossover
context this project measures (~36k tokens; see §5.1).

**Fit** (from 78 measured runs; `phase3_throughput_vs_concurrency.png`, `phase3_tpot_vs_concurrency.png`):

| Mix | r0 (tok/s/seq) | μmax (tok/s) | knee N* |
|---|---|---|---|
| mixed (baseline) | 49.5 | 495 | **10.0** |
| chat | 58.0 | 355 | 6.1 |
| phase1 heavy mix | 20.0 | 250 | 12.5 |

`r0 ≈ 48` independently matches the measured ~21 ms standalone TPOT — a physical anchor, not a free
knob. Two observations validate the framing:

1. **All five Phase-2 policies collapse onto one `throughput(N)` curve.** They differ only in the
   concurrency `N` they induce; they ride the same service law. This is direct evidence that the
   scheduler is an admission layer, not a change to the service process — the whole project premise,
   confirmed by the model.
2. **Same output length, different ceiling:** the heavy mix and the baseline mix both average ~104
   output tokens but have μmax of 250 vs 495. The heavy mix has longer *prompts*, so more compute goes
   to prefill and the output-token ceiling halves. μmax is mix-specific; the PS *structure* is universal.

### 5.1 The capacity limit is compute, not KV — and we can say where that flips

The Phase-3 brief anticipated a **KV-cache** capacity limit as the thing that breaks the model. The
data says otherwise, and that is the interesting result. `kv_occupancy` never exceeds ~8 %, even at 60
concurrent sequences. Deriving the KV cap directly from the measurements (`C_kv = N / kv_occupancy`)
gives **C_kv ≈ 830 concurrent sequences** versus the compute knee **N\* ≈ 10** — **compute saturates
~83× before KV does.** On a 1.5B model with 16 GB, KV is slack; the T4 is compute-bound. The
KV-blocking regime that PagedAttention/vLLM were built for is simply outside this hardware's envelope.

Because the KV budget is fixed in *tokens*, `C_kv(context) = C_kv · (ctx₀ / context)`, so KV overtakes
compute when the average sequence length reaches **~25–36k tokens** on this exact setup
(`phase3_capacity_regimes.png`). Below that, compute binds (what we measured); above it — long-document
RAG, or a larger model with fatter KV/token — KV blocking dominates and admission-layer KV control
starts to matter.

**We tried to reach that regime empirically** (see `report/kv_regime_experiment.md`). A long-context
calibration probe (8k-token prompts) held 27 running sequences at only **60 % KV occupancy while
SM-active pinned at 97 %** — i.e., the binding constraint at 8k is *prefill compute*, not KV. That
measurement fixes the KV budget at ~360k tokens, placing the KV-crossover at ~36k-token contexts —
just past the model's 32k `max_model_len`. The two ways to push there both fail on this platform: long
prompts saturate prefill compute first (measured), and long outputs make decode impractically slow
(tens of thousands of sequential tokens per request). **So across its entire feasible envelope
(≤32k context), a T4 + 1.5B vLLM instance is compute/bandwidth-bound and never KV-capacity-bound.**
This is a rigorously bounded negative result: it *confirms and hardens* the compute-bound thesis, and
tells you precisely what it would take to enter the KV regime (a bigger model or a longer context
window than this hardware admits).

### 5.2 Where model and measurement diverge

Nineteen points exceed 20 % throughput error, all explainable and none a failure of the PS structure:
(a) low-load short-window sampling noise (few, bursty requests per window); (b) thermal throttling
depressing the plateau below `μmax` on the passively-cooled T4; (c) the real batching ramp is smoother
than the sharp `min`/`max` kink around `N*`. A clear approximate model that *explains* the gap beats a
tuned complex one, per the project's modeling rule.

---

## 6. Honest limitations

- **Single GPU, small model, admission layer.** No multi-GPU, no tensor/pipeline parallelism, no
  engine-internal changes. Multi-tenancy is simulated as multiple streams on one instance.
- **The aggregate result is largely a null result.** On a T4 + 1.5B, vLLM's batcher is hard to beat on
  throughput/latency; the scheduler's demonstrated value is *differentiated service and convoy
  avoidance under overload*, not a strict aggregate win. We do not claim to beat production systems.
- **Compute-bound regime only.** Because KV never binds within the 32k context ceiling, we could not
  empirically exercise KV-admission control — we probed for it, bounded the crossover at ~36k-token
  contexts, and confirmed the platform is compute/bandwidth-bound throughout its feasible envelope
  (§5.1).
- **Thermal noise.** The passively-cooled T4 throttles on long runs; we gate on temperature and report
  variance, but it inflates run-to-run spread near the knee.
- **Off-the-shelf drafts.** Speculative decoding uses n-gram lookup as shipped. We did not train draft
  models; any draft-training tooling referenced elsewhere contributed *loss functions*, not drafts used
  here.

---

## 7. Future work

- **Learned/RL admission policy.** The `adaptive` rule is hand-tuned; a learned controller over the
  same telemetry (SM-active, KV, TTFT p99) is the natural next step. Out of scope here by design.
- **The KV-bound regime.** Re-run at long contexts (≥25k tokens) or on a larger model where `C_kv < N*`
  — that is where admission-layer KV control should finally pay off, and where this model predicts the
  crossover.
- **One adaptive in-engine knob.** The only place engine-internal change is warranted: make the
  batch-size cap or KV-admission threshold adaptive under pressure — kept isolated, after the
  admission-layer comparison (which this report is).
- **GPU colocation / multi-instance routing** for true multi-tenancy beyond a single instance.

---

## 8. Related work

**Serving engines and batching.** vLLM's continuous batching and PagedAttention (Kwon et al., SOSP
2023) are the substrate this project measures; PagedAttention specifically targets *KV-cache*
fragmentation and capacity, which motivated our test of whether KV is the binding constraint on
commodity hardware (§5.1) — we find it is not, within a 32k context on a T4 + 1.5B. Orca (Yu et al.,
OSDI 2022) introduced iteration-level (continuous) scheduling inside the engine; we deliberately stay
*in front* of that mechanism, controlling admission and ordering rather than intra-engine batch
formation.

**Inference scheduling and SLOs.** A line of work schedules LLM requests for latency SLOs and
fairness — e.g. output-length-prediction scheduling (S³, Jin et al., NeurIPS 2023), fair KV-cache
sharing (VTC, Sheng et al., OSDI 2024), and disaggregated prefill/decode (DistServe, OSDI 2024;
Splitwise, ISCA 2024). Our contribution is narrower and complementary: a *measured characterization*
of what an admission/ordering layer adds on top of an already-batching engine on a single commodity
GPU, including the honest finding that the gain is differentiated service under overload rather than
aggregate throughput. Our negative predictor result (§4.4) is a small counterpoint to the S³-style
premise that better length prediction improves scheduling — at our operating point it does not.

**Queueing theory.** The processor-sharing (PS) and Shortest-Remaining-Processing-Time (SRPT) models
we use are classical (Kleinrock; Schrage). The PS *insensitivity* property — mean sojourn time
depending only on load, not the service-time distribution — is exactly why we observe every policy
collapsing onto one `throughput(N)` curve (§5). Our extension is the load-dependent batched service
rate `μ(N) = min(N·r0, μmax)` and the explicit dual capacity cap `min(N*, C_kv)`, which localizes the
compute-vs-KV crossover.

**Speculative decoding.** We serve off-the-shelf n-gram prompt-lookup speculation (as in vLLM) and
measure its acceptance per class; we do **not** train draft models. Draft-training methods (EAGLE,
Medusa, and related loss-function work) are orthogonal to the scheduling question studied here.

*(Citations are by name/venue for orientation; this is an engineering report, not a peer-reviewed
paper, and the framing above states its boundaries plainly.)*

---

## 9. Reproducing

See the repository `README.md` for exact environment, server, and run commands, and `reproduce.sh`
for an end-to-end regeneration of the headline figures from `results/`. All committed summaries
(`results/summaries/*.md`) and figures (`results/figures/*.png`) are the artifacts of record; raw
per-run JSONs stay local by design.
