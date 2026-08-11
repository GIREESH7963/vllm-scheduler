# Related work — scaffold

*Scaffolding for §2 of `docs/paper_draft.md`. This is a structure and a verified starter
bibliography, **not** a finished literature review.*

**The rule for this file: no citation is written from memory.** Every entry in §6 was checked
against a primary or near-primary source in August 2026, and each carries the URL it was checked
against. Entries that could not be verified sit in §7 under "to find", with what is missing
stated explicitly. Nothing in this file should be pasted into a manuscript without a human
opening the paper — a verified title and venue is not the same as a verified *claim about what
the paper says*.

---

## 1. The positioning problem, stated first

A literature check turned up something that changes how the paper must be framed, and it should
be settled before any prose is written.

**The memory overhead of vLLM V0's batch-expansion scorer is not an unknown.** Practitioner
writing on speculative decoding already describes V0's batch expansion as "resource-intensive"
and as carrying a **K× memory overhead**. (That writing further attributes V1's redesign to this
overhead; that attribution could **not** be corroborated from any primary source and must not be
repeated — see §7 item G, which documents what was checked.) Independently, it is well
established that speculative decoding *degrades* at large batch sizes, because at high
concurrency the target model is already compute-efficient and the extra `k` verification
positions are pure overhead.

So the paper cannot claim to have discovered that the scorer is expensive. What it can claim,
and what the evidence in `docs/experiment_b.md` actually supports:

1. **An exact closed form, tested.** "K× overhead" is a scaling statement. `N·(k+1)·V·4` bytes is
   a prediction, and it holds to 0.1% on the k axis and 0.5% out at N ≈ 410. Nobody in the
   material surveyed gives or tests a closed form.
2. **That this allocation is the *binding* constraint** — the thing that ends the engine's life —
   rather than a tax on throughput. The existing literature discusses spec-decode overhead as a
   *speedup* problem; this paper reports it as a *survival* problem.
3. **That the standard capacity signal cannot see it.** The occupancy inversion (survived at
   82.6%, died at 57.4%) is the novel and defensible core, and it is an argument about
   observability, not about speculative decoding.
4. **The `max_num_seqs` regime flip**, which is a scheduler-parameter result and appears
   unrelated to the spec-decode literature entirely.

**Recommended framing:** lead with (3) and (4). The paper is about *what serving systems cannot
see*, with the scorer as the worked example — not a paper about speculative decoding being
expensive, which is known. Reviewers who know the spec-decode literature will otherwise read the
contribution as a re-discovery, and §1.1 of the draft currently invites that reading.

---

## 2. Section structure for §2 of the draft

Four subsections, each ending in the sentence that does the positioning work.

### 2.1 Continuous batching and paged memory
Establish the mechanism the paper measures. Orca introduced iteration-level scheduling — the
scheduler runs a single iteration over the batch rather than a whole request — and vLLM/
PagedAttention introduced the paged KV pool that makes occupancy the natural capacity signal.
**Close on:** the paged allocator manages KV and only KV; allocations made outside it by other
parts of the engine are invisible to the signal operators actually watch. That gap is this
paper's subject.

### 2.2 Speculative decoding and its costs
Leviathan et al. and Chen et al. for the algorithm and its distribution-preserving guarantee.
Then the batch-size interaction: speculation helps at low concurrency and hurts at high, where
the target is already compute-bound. Then implementation: V0's batch expansion versus V1's
flattened rejection sampler.
**Close on:** this literature treats the cost as a throughput tax measured in speedup. We
measure it as an allocation, give its closed form, and show it terminates the engine.

### 2.3 Admission control and SLO-aware scheduling for LLM serving
Sarathi-Serve (chunked prefill, stall-free scheduling), Llumnix (migration and rescheduling),
QLM (queue management with waiting-time estimation), FastServe (preemptive MLFQ).
**Close on:** these systems schedule against latency and SLO targets under an assumed capacity.
§9 of the draft reports that admission control moves *aggregate capacity itself* (+178 tok/s at
n=10, surviving Holm) — which is a different claim from the differentiation results these
systems report, and it should be stated as complementary rather than contradictory.

### 2.4 Queueing models for batch-parallel servers
This is the subsection with the largest novelty risk. **Limited processor sharing is established
theory** — Zhang & Zwart give steady-state heavy-traffic approximations, and Zhang, Dai & Zwart
give fluid and diffusion limits. §4 of the draft currently argues its way to LPS as though
arriving somewhere new.
**Close on:** the contribution is not the LPS framing but the empirical finding that the
multiprogramming limit and the memory constraint are *coupled through the implementation* (§8):
raising the LPS limit shrinks the resource the limit governs. That coupling is not in the
queueing literature because it is an artefact of how the engine reserves activation memory.

---

## 3. Claim-to-citation map

Each row is a claim already in `docs/paper_draft.md` that will not survive review uncited.

| Draft § | Claim needing support | Citation |
|---|---|---|
| §1, §2 | Requests join/leave a running batch between steps | Orca |
| §1, §2 | KV stored in fixed-size blocks; occupancy is the capacity signal | vLLM/PagedAttention |
| §2 | Draft proposes k tokens, target scores k+1, prefix accepted | Leviathan et al.; Chen et al. |
| §2 | Distribution-preserving property of speculative sampling | Chen et al. |
| §4 | LPS is the right queueing description, not PS | Zhang & Zwart; Zhang, Dai & Zwart |
| §4, §8 | `max_num_seqs` is the multiprogramming limit | vLLM docs (primary source: the code) |
| §7.3 | Speculation degrades at high batch size (motivates why k matters) | **§7 item F — to find** |
| §7.1, §10.2 | V0 batch expansion vs V1 rejection sampler | Own source read + **§7 item G** |
| §9 | Admission control framed as differentiation at fixed capacity | Sarathi-Serve; Llumnix; QLM; FastServe |
| §9 | Holm correction over a comparison family | **§7 item H — statistics reference** |
| §10.4 | Thermal throttling changes the service curve | **§7 item I — to find** |

---

## 4. What this paper is *not* claiming, for the reviewer's benefit

Worth a short paragraph in §2, because it pre-empts the three likeliest misreadings:

- Not claiming speculative decoding is bad. It is a net win in the regime it was designed for.
- Not claiming to have found a bug. The allocation is doing what it was written to do; the
  finding is that its size is unbounded in a dimension nothing admits against.
- Not claiming novelty for the LPS model. See §2.4.

---

## 5. Effort estimate

| Task | Estimate |
|---|---|
| Read the 9 verified works closely enough to cite claims, not titles | 1–2 days |
| Resolve §7's five open items | half a day |
| Write §2 (≈1200 words) | 1 day |
| Rework draft §1.1 and §4 for the positioning in §1 above | half a day |

---

## 6. Verified bibliography

*Checked August 2026. "Verified" means title, authors, venue and year were read off the linked
source. It does **not** mean the papers' contents have been read closely — do that before citing
any specific claim.*

**Serving systems**

- **Kwon, W. et al.** "Efficient Memory Management for Large Language Model Serving with
  PagedAttention." *SOSP 2023* (ACM SIGOPS 29th Symposium on Operating Systems Principles).
  Reported 2–4× throughput over FasterTransformer and Orca.
  <https://en.wikipedia.org/wiki/PagedAttention>, <https://docs.vllm.ai/en/latest/design/paged_attention/>
  *Full author list still to be taken from the ACM DL entry.*

- **Yu, G.-I., Jeong, J. S., Kim, G.-W. et al.** "Orca: A Distributed Serving System for
  Transformer-Based Generative Models." *OSDI 2022* (16th USENIX Symposium on Operating Systems
  Design and Implementation). Iteration-level scheduling and selective batching.
  <https://www.semanticscholar.org/paper/9d7a75601e0e50dd68d40cfb8ef0e891dad797a6>
  *Full author list still to be taken from the USENIX proceedings.*

- **Agrawal, A., Kedia, N., Panwar, A., Mohan, J., Kwatra, N., Gulavani, B., Tumanov, A.,
  Ramjee, R.** "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve."
  *OSDI 2024*. arXiv:2403.02310. Chunked prefills and stall-free scheduling.
  <https://www.usenix.org/conference/osdi24/presentation/agrawal>

- **Sun, B. et al.** "Llumnix: Dynamic Scheduling for Large Language Model Serving." *OSDI 2024*.
  Live migration of requests and their KV cache across instances.
  <https://www.usenix.org/conference/osdi24/presentation/sun-biao>
  *Full author list still to be taken from the USENIX proceedings.*

- **Patke, A., Reddy, D., Jha, S., Qiu, H., Pinto, C., Narayanaswami, C., Kalbarczyk, Z.,
  Iyer, R.** "Queue Management for SLO-Oriented Large Language Model Serving." *SoCC 2024*
  (ACM Symposium on Cloud Computing), Redmond WA, Nov 20–22 2024. DOI 10.1145/3698038.3698523.
  arXiv:2407.00047. <https://dl.acm.org/doi/10.1145/3698038.3698523>

**Speculative decoding**

- **Leviathan, Y., Kalman, M., Matias, Y.** "Fast Inference from Transformers via Speculative
  Decoding." *ICML 2023*, PMLR 202:19274–19286. arXiv:2211.17192.
  <https://proceedings.mlr.press/v202/leviathan23a.html>

- **Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre, L., Jumper, J.** "Accelerating
  Large Language Model Decoding with Speculative Sampling." arXiv:2302.01318, February 2023.
  Modified rejection sampling preserving the target distribution; 2–2.5× on Chinchilla 70B.
  <https://arxiv.org/abs/2302.01318>
  *Preprint — check for a peer-reviewed version before citing.*

**Queueing theory — limited processor sharing**

- **Zhang, J., Zwart, B.** "Steady State Approximations of Limited Processor Sharing Queues in
  Heavy Traffic." *Queueing Systems* 60:227–246, 2008. <https://ir.cwi.nl/pub/13865>

- **Zhang, J., Dai, J. G., Zwart, B.** "Law of Large Number Limits of Limited Processor-Sharing
  Queues." *Mathematics of Operations Research*, 2009. DOI 10.1287/moor.1090.0412.
  <https://pubsonline.informs.org/doi/10.1287/moor.1090.0412>

- **Zhang, J., Dai, J. G., Zwart, B.** "Diffusion Limits of Limited Processor Sharing Queues."
  *Annals of Applied Probability* 21(2), 2011.
  <https://projecteuclid.org/journals/annals-of-applied-probability/volume-21/issue-2/Diffusion-limits-of-limited-processor-sharing-queues/10.1214/10-AAP709.pdf>

---

## 7. Open items — do not cite until resolved

- **F. Speculative decoding degrades at large batch size.** Needed for §2.2 and to motivate why
  k is the interesting axis. The claim is widely repeated in practitioner writing (one benchmark
  cited a Qwen3-8B speedup falling from 1.93× to 0.99× as batch grew 2 → 48) but I have **not**
  verified a peer-reviewed source. Find one; "Batch Speculative Decoding Done Right"
  (arXiv:2510.22876) is a candidate but is unverified and its relevance is unchecked.

- **G. "vLLM V1 replaced batch expansion *because of* its memory overhead" — could not be
  corroborated, and should not be asserted.** This was the most useful citation available to the
  paper, so it was chased hardest, and it did not stand up:
  - The claim appears in practitioner writing ("deprecated in v1 due to K× memory overhead") but
    no primary source was found for the *rationale*.
  - The V1 alpha announcement (<https://vllm.ai/blog/2025-01-27-v1-alpha-release>) does not
    discuss batch expansion, rejection sampling, or scorer memory at all. Its only mention of
    speculative decoding is to list it among features V1 **did not yet support**: "V1 currently
    lacks support for log probs, prompt log probs sampling parameters, pipeline parallelism,
    structured decoding, speculative decoding, prometheus metrics, and LoRA."
  - Searches of vLLM's docs and GitHub surfaced no RFC or PR stating a memory-overhead rationale.

  So the defensible facts are narrower than the claim: V1 was a ground-up rewrite that initially
  shipped **without** speculative decoding, and when it returned it was built on a flattened
  rejection sampler rather than a padded 3-D expansion. **Whether memory drove that choice is
  unestablished.** The draft's §10.2 is already written correctly — it reports the code
  difference from our own source read and calls V1 persistence an open question — and it must
  stay that way. Do not upgrade it to a claim about intent. If a primary source is wanted, the
  remaining places to look are the git history of the deleted `batch_expansion.py` and the PR
  that introduced `v1/sample/rejection_sampler.py`.

- **H. A statistics reference for Holm correction** in §9, and ideally for Hedges' g and Welch's
  t. Textbook citations; trivial to add, currently absent.

- **I. Thermal throttling as a source of non-stationarity** in performance measurement (§10.4).
  There is a systems-measurement literature on this; none of it has been surveyed here.

- **J. Full author lists** for PagedAttention, Orca and Llumnix, from the proceedings rather than
  from aggregators.

- **K. Nothing has been surveyed on memory fragmentation or allocator behaviour** under
  PyTorch's caching allocator, which §7.4 of the draft invokes to explain why the boundary is
  stochastic rather than deterministic. That explanation is currently asserted from the shape of
  the data. Either find support or label it explicitly as a conjecture.

---

## 8. A note on scope creep

The draft's §13 lists three experiments that would raise the paper's ceiling. None of them are
blocked by this file, and this file should not grow into a survey. §2 of a paper of this size
wants roughly 1200 words and 12–18 references. The verified list above is 9; items F–J would
bring it to a reasonable count without further searching.
