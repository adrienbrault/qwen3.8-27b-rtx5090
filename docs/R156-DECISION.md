# R156 — Promotion decision sheet: keep the daily, or switch to RedHat?

**Status: DECIDED 2026-09-02 — user: "Lets go redhat". Promoted (FINDINGS R156f); rollback `launch-daily-gittensor-0831.sh`. Evidence below is as it stood at the decision.**
Reviewed 2026-09-01 evening — reliability audit in `flan/r156-REVIEW.md`; numbers below carry its labels.

Campaign question (user, 2026-09-01): *"thorough testing/measurement of bf16 vs our current daily …
I want to understand what we're missing"*, later reframed: *"with the dual/tp2 setup, I think it's
worth considering less quantization if it makes a difference in task performance. A bit slower
decode/prefill can be worth it."*

---

## 1. What we were missing

Our daily sits **+4.46% perplexity from bf16 ground truth — last of 13 configurations tested.**
Every gate we had said quality was fine, because every gate we had is too blunt to see it:

| instrument | resolution | saw the gap? |
|---|---|---|
| fidelity probe (certain-bucket flip rate) | **0.003%** | yes, easily |
| GSM8K @ n=250 | ~8.4pp | no — structurally cannot |
| tool-eval | +-2-3 | no |

Decomposition of the 4.46% (all controlled): ~2.5pp quantization **scheme**, ~1.55pp **preservation
count**, ~0.85pp **4-bit lm_head**, 0.13pp fp8 KV, (0.76pp nvfp4 KV if used).

**Caveat on the ruler (see `r156-REVIEW.md`):** the corpus is raw text that is largely in the
model's training data (Python stdlib sources, wikitext-103, GSM8K-train) with no chat template,
tool calls, or thinking. It measures weight fidelity faithfully. **Closed by R156e** (below): the
same gap, at the same ratio, appears on bf16-generated chat-templated agentic turns.

Folklore checked: **lm_head — TRUE** (0.851%, controlled pair). **GDN — MOOT** (every checkpoint
already preserves the recurrence-critical params). **"KV quant is very bad" — only at 4 bits**
(0.76%); fp8 KV is 0.129% and does NOT compound with context out to ~171K.

## 2. The candidate

Three checkpoints tie at the quality floor (~+0.37-0.38%): unsloth, kelnei, **RedHat**. All use
compressed-tensors mixed-precision. RedHat is *nominally* fastest (decode_ss c8 +5.5% over unsloth,
n=3 at a 3-6% run spread — unconfirmed; the spec-OFF kernel instrument never ran for unsloth or
kelnei). The candidate is "one of the three", not RedHat specifically.

## 3. One knob, four consequences

RedHat preserves **303** modules at 8-bit; the daily preserves **148**. That single difference
produces all of the following — they cannot be separated by configuration:

| axis | daily | RedHat | change |
|---|---|---|---|
| **fidelity** (PPL gap vs bf16, raw corpus) | +4.46% | **+0.37%** | **BENEFIT: -4.09pp** |
| **fidelity, DEPLOYED REGIME** (R156e: bf16-generated tool/code/reason/prose turns) | +6.56% PPL, 92.6% top-1 agree, **8.6%** moderate-bucket flips | **+2.41%**, **95.9%**, **3.4%** | **BENEFIT: 2.5x fewer divergences, uniform across kinds** |
| weights | 18.77 GB | 23.42 GB | +4.33 GiB |
| KV pool @262K | 746,849 | 654,491 | **-12.4%** |
| decode, kernel-only c1 (spec OFF) | 129.6 tok/s | 105.5 tok/s | **-18.6%** (upper bound on production cost) |
| decode, kernel-only c8 | ~769 tok/s | ~665 tok/s | **-13.5%** |
| decode, spec ON, c8 (decode_ss, n=3) | 1298.8 tok/s | 1212.1 tok/s | **≈ −6.7% ± ~4** (kernel −13.5% × acceptance +6% — consistent) |
| **decode, spec ON, c1 (interactive, benchy T=0.6)** | 338.7 tok/s | 318.8 tok/s | **≈ −6%** (three instruments agree: −5.5/−5.9/−6.7%) |
| prefill pp2048 c1 (benchy) | 10,190 tok/s | 8,741 tok/s | **−14.2%** (+33 ms TTFT @2K) |

### 3b. Full perf sheet — daily vs RedHat, both on DFlash2 ns9 (syvai drafter), TP2 util 0.92, fp8 KV, no tier

| measurement (instrument) | daily | RedHat | delta |
|---|---|---|---|
| **kernel only, spec OFF** c1 (decode_bench kernel mode, content-independent) | 129.6 tok/s | 105.5 | **−18.6%** |
| kernel only, spec OFF c8 | 759-786 | 658-679 | **−13.5%** |
| **spec ON c1, benchy T=0.6 natural** (pp2048/tg256, n=3) | 338.7 ± 33 | 318.8 ± 8 | **−5.9%** |
| spec ON c1, benchy T=0.6 exact-tg | 351.4 ± 72 | 332.2 ± 29 | −5.5% |
| spec ON c8, benchy T=0.6 natural, mean | 704.3 ± 16 | 656.3 ± 12 | −6.8% (ramp-inclusive) |
| spec ON c8, benchy T=0.6 natural, **peak** | 1,492 | 1,511 | ≈ 0 (steady-state window) |
| spec ON c8 **code**, decode_ss T=0.6 steady-state (n=3) | 1,298.8 [1167-1316] | 1,212.1 [1139-1231] | **−6.7%** |
| spec ON c8 prose, decode_ss | 999.5 [973-1046] | 924.6 [894-956] | **−7.5%** |
| spec ON c1 code, decode_ss | 208.5 [208-322] | 246.1 [176-319] | n/a — c1 decode_ss spread is ±50%, unusable |
| spec ON c1 prose, decode_ss | 224.9 [191-279] | 187.8 [163-204] | n/a — same |
| spec ON c1 prose **@30K context**, decode_ss (n=2) | 166.7 [142-191] | 157.2 [150-165] | −5.7% (wide) |
| draft acceptance per draft token, code c8 / prose c8 | 0.271 / 0.185 | 0.288 / 0.194 | **+6% / +5%** (higher fidelity → drafter agrees more) |
| **prefill pp2048 c1** (benchy) | 10,190 tok/s | 8,741 | **−14.2%** |
| prefill pp2048 c8 (benchy) | 10,787 | 9,284 | −13.9% |
| TTFR @2K c1 / c8 (benchy) | 202 / 1,129 ms | 235 / 1,313 ms | +33 / +184 ms |
| TTFT @30K c1 (decode_ss) | 3.64 s | 4.12 s | +0.48 s (+13%) |
| KV pool @262K max-len | 746,849 | 654,491 | **−12.4%** (2.85x → 2.50x max context) |
| weights on disk | 18.79 GB | 23.44 GB | +4.65 GB (= unsloth 23.46, kelnei 23.44: same recipe, same pool) |

How the rows fit together: kernel −18.6% c1 is the cost of the forward pass alone (303 vs 148
modules at 8-bit); RedHat's higher acceptance (+5-6%) buys back about a third of it under
DFlash, landing spec-ON decode at **≈ −6% single-stream and −7% at c8** on real content. Prefill
gets no acceptance rebate, so it carries the full **−14%**. Retracted instruments (ignore_eos
decode_bench, T=0 benchy) are excluded; c1 decode_ss rows are shown only to document why they are
not used (n=3 with ±50% spread). Numbers from `results/2026-09-01-r156-bf16-ladder/audit.log`
(tags `kern.*`, `benchy.*`, `perf.c`, `perf.h.syvai`).

VRAM accounting verified: pool deltas reduce entirely to the weight delta at 35,089 B/token vs
32,768 theoretical (ratio 1.07) — arms differ only in the intended variable.

**Pool cost in context:** 654,491 is still ~2.5x the 262K max context. This is concurrency headroom
at extreme depth, not a capability loss; neither pool binds everyday short-context traffic.

## 4. The honest gap in the evidence

**No affordable behavioural gate can adjudicate this.** GSM8K at n=250 resolves nothing below
~8.4pp; observed spread was 1.8pp. All arms are statistically indistinguishable (p=0.43-0.84).
Even a full paired GSM (1319 items) lands at ~2-4pp MDD — still likely blind.

So the user's own criterion — *"if it makes a difference in task performance"* — **cannot be
answered with the instruments we have.** What is established:
* the fidelity difference is **real and large** (~1500x the probe's noise floor) **and it is the
  same size in the deployed regime** (R156e: bf16 generates chat-templated tool/code/reason/prose
  turns; on the decision points of those trajectories the daily diverges 1 token in 12, RedHat 1 in
  30 — 2.5x, uniform across all four kinds, including tool calls);
* it has **no demonstrated task-outcome consequence**, because no affordable task gate can show one;
* the costs are **certain and quantified**.

That is a genuine judgement call about how much to weight measured-but-unrealised quality against
measured-and-certain speed. It is not resolvable by more of the same measurement.

## 5. Options

1. **Keep the daily.** Costs nothing. Accepts a 4.46% PPL gap with no known behavioural harm.
2. **Switch to RedHat.** Buys a ~12x reduction in fidelity gap; pays -18.6% single-stream decode,
   -12.4% pool. Rollback is a one-line launch-script change.
3. **Commission a sensitive gate** (a GPU-day: long-horizon agentic/tool tasks, or a much larger
   paired eval) to find out whether the fidelity gap has any behavioural bite before deciding.
   Cheapest form (~2h): bf16 generates responses to held-out agentic prompts; every quant arm is
   teacher-forced on those generations with the existing scorer (see r156-REVIEW.md §8).

Option 3 is the only one that answers the original question rather than working around it.
