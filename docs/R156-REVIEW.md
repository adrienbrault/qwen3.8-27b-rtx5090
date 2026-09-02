# R155/R156 — critical review of 2026-09-01 (measurements, reliability, conclusions)

Written at the user's request after shutdown ("I question a lot some of the measurements
reliability, and so some of the insights/decisions out of these"). Raw results live on the serving host (off at the time),
so this audits the CODE that produced each number, the recorded n/CIs, and the reasoning — it does
not re-run anything. Confidence labels: HIGH / MEDIUM / LOW / RETRACT.

## 0. One-paragraph verdict

The central fidelity result is solid and survives scrutiny: on a raw-text corpus, our daily's
perplexity is ~4.4% above bf16 while three mixed-precision checkpoints are ~0.4% above; the noise
floor (0.015% PPL, 0.003% certain-bucket flips) is two orders of magnitude below that effect, the
tokenization-alignment guard held, and the ranking reproduced on a 544-doc subset. **What is NOT
solid is almost everything downstream of it**: the decode numbers went through three instruments,
two of which I built and mis-described; several sub-claims (drafter "opposite directions",
"unsloth beats fp8", "no KV compounding", scheme-vs-preservation split) are stated more strongly
than their power supports; and two things I told the user today were simply false (`/metrics`
"not exposed"; decode_ss "variable output length"). The promotion decision itself is correctly
left open, but the cost side of the sheet needs relabelling: the production single-stream decode
cost of RedHat is UNMEASURED, bounded between about −7% and −19%.

## 1. The fidelity ladder — HIGH confidence on the headline, with caveats

**Instrument:** `fidelity_ladder.py` dense mode → `/v1/completions {echo:true, max_tokens:0,
logprobs:20, temperature:0}`; PPL from the forced-token logprob (`tok_lp`, full-vocab), flips
bucketed by the REFERENCE arm's p1, Wilson CIs, alignment guard that refuses mismatched dumps.

What holds:
- Noise floor measured FIRST (C1 vs C2, same arm two boots): PPL Δ 0.015%, certain flips 0.003%.
- Daily +4.46% vs unsloth/RedHat/kelnei +0.37/0.38/0.37%: ~300x the PPL floor. **HIGH.**
- Ranking reproduced on arm A's 544-doc subset vs the full 693 (unsloth −3.92% vs −4.02%). Good.
- Alignment guard passed on every pair; the unsloth tokenizer difference was verified decoder-only.
- fp8 KV +0.129% (B vs C, controlled). **HIGH.**
- lm_head 4-bit +0.85% (RadixArk L vs R, controlled) corroborated at +0.72% (N vs C). **HIGH** for
  "lm_head quant costs ~0.7-0.85% on this model"; MEDIUM as an attribution of the daily's own gap
  (extrapolated from another family, with one within-family corroboration).

Caveats that limit what the headline MEANS:
1. **The corpus is largely memorised text.** code = Python dist-packages sources, prose =
   wikitext-103 test, reasoning = GSM8K *train*. All three are near-certainly in Qwen3.8's
   pretraining/post-training data. Only tech-prose (our own notes) is genuinely held-out. PPL on
   memorised text measures how well quantization preserves memorised sequences — a legitimate
   fidelity proxy, but not the same thing as generalisation, and not the deployed regime.
2. **No chat template, no tool calls, no thinking.** The daily's actual workload is agentic coding
   through the chat template. Teacher-forcing raw text measures the base LM distribution. Whether
   quantization error lands the same way inside `<tool_call>` JSON and reasoning traces is
   untested — and that is precisely the question the user cares about.
3. **Arms differ in more than the intended variable at the launcher level:** A ran MAXLEN 3072 /
   util 0.88 / MNBT 2048; i ran 8192 / 0.60; the rest 131072 / 0.50-0.60 / 4096. Pool size cannot
   change logits, and 1.1K-token docs are one prefill chunk at any of these MNBTs. `max_model_len`
   would matter only if vLLM derived a RoPE/YaRN factor from it — I did NOT verify Qwen3.8's
   rope config. Low risk (arm A lands where bf16 should, below every quantized arm), but the
   noise floor (same config twice) cannot catch a config-dependent offset. **Unverified assumption.**
4. **The arms measured the checkpoint, not the deployed stack**: spec OFF, prefix cache OFF, no
   tier. "Spec decode is distribution-preserving" is true in theory; R155 Bug B is the same day's
   proof that the deployed decode path can be silently wrong in practice. The 4.46% is a property
   of the weights. The deployed daily has never been fidelity-scored as deployed.
5. `logprobs=20` truncated-KL was undefined at 98% of positions for D-vs-C — correctly demoted;
   flips + PPL carry the result. Fine.

Over-claims to soften:
- "All three BEAT uniform fp8" — 5.4701 vs 5.4735 is 0.06% at a 0.015% single-observation floor.
  Call it "tie-to-marginally-better". **LOW as a ranking claim.**
- "Scheme ~2.5pp vs preservation ~1.55pp" — Inferact does not "vary ONLY preservation count";
  it is a different vendor's modelopt checkpoint (calibration set, modelopt version, head
  treatment all unknown). The robust decomposition is: KV 0.13, lm_head ~0.75, **remainder ~3.5pp
  = "the checkpoint"**. The finer split is **MEDIUM-LOW**.

## 2. Decode — the shaky part, and where I was wrong twice

Four instruments touched RedHat-vs-daily today:

| instrument | what it measured | verdict on it |
|---|---|---|
| decode_ss (perf arms, drafter 2x2) | spec ON, T=0.6, min_tokens=max_tokens=512, engine-counter tokens over the all-streams-live window | **the repo-standard tool and BETTER than I described it** — see below |
| decode_bench fixed-work | spec ON, ignore_eos, SSE-chunk → later usage-token counting | RETRACTED for ranking: forced-past-EOS degeneracy is checkpoint-dependent (−35%…+133% swing) |
| decode_bench spec-OFF ("kernel") | NOSPEC=1, ignore_eos, usage tokens | **HIGH**: 129.6 vs 105.5 c1, content-independent to 4 s.f. |
| llama-benchy | spec ON, exact-tg and natural | daily only (326.9 ± 33 c1); RedHat never ran (shutdown) |

**What I got wrong about decode_ss** (`decode_bench.py` docstring lists four flaws; two are false):
- "Variable output length" — FALSE. decode_ss sends `min_tokens = max_tokens`, i.e. fixed length.
- "Counts chunks" — never true of decode_ss; it reads `vllm:generation_tokens_total` from
  `/metrics` (exact) and `spec_decode_num_accepted/draft_tokens_total` for acceptance (exact).
- True flaws: c1 steady-state window is 0.5-1.5s (c1 cell unreliable — the FINDINGS said so), and
  T=0.6 random-topic prompts add content variance (3-6% run spread at c8, per the banked memory).
- Note decode_ss ALSO forces length past natural EOS (`min_tokens`), so the degeneracy confound I
  found in decode_bench applies to it in principle — much less in practice, because its prompts
  ask for long outputs and 512 tokens is usually within natural length.
So I replaced a validated tool with a worse one and mis-stated the old one's defects. The user's
"are you not using llama benchy" challenge was pointing at exactly this pattern.

**FALSE statement I made to the user:** "`/metrics` returns 0 lines while the server is actively
serving — the counters genuinely aren't exposed on this build." decode_ss has been reading those
counters all campaign (acceptance 0.271/0.288/0.194 all come from them). Both of my probes hit
the port while no engine was serving (mid-boot; post-teardown, `status=000` = connection refused).
Consequence: exact acceptance is available for free; the composition I was building could have
used it. **RETRACT.**

**What the decode evidence actually supports for RedHat vs daily (syvai drafter):**
- Spec OFF: −18.6% c1, −13.5% c8. **HIGH** (content-independent, rel_iqr ≤0.6%).
- Spec ON, c8, decode_ss: −6.7% with acceptance 0.288 vs 0.271. Triangulates: −13.5% kernel ×
  ~+7% tokens/step ≈ −7%. The three numbers are mutually consistent. **MEDIUM** (n=3 per cell,
  ~3-6% spread ⇒ −6.7% ± ~4).
- Spec ON, c1 (the interactive number that actually matters): **NOT MEASURED.** decode_ss c1 is
  noise-limited, decode_bench c1 is confounded, benchy-h never ran. Bounded between ~−7% (if
  acceptance compensates as at c8) and −18.6% (kernel). The decision sheet's "decode, kernel-only
  c1 −18.6%" row is labelled correctly; my message to the user ("−18.6% single-stream decode") was
  not. **Relabel.**

**"Canon" is circular.** decode_ss c8 "reproduced canon 1298.8 vs 1289" — canon came from decode_ss
(R142). That is a reproducibility check, not validation. Worse, the c1 canon (~299) is from the
c1 cell of the same instrument, which I then declared unreliable. It was still a valid anchor to
catch a 3.7x chunk-counting bug, but "benchy +9% vs canon, decode_bench +56%" treats 299 as truth
and it is not. benchy's 326.9 ± 33 may be the better c1 number.

**Drafter 2x2 "opposite directions" — over-claimed at the THROUGHPUT level only.** The acceptance
interaction is real and exact (engine counters over ~37K draft tokens per run: unsloth 0.271→0.290,
daily 0.271→0.247, RedHat 0.288→0.271 — genuinely opposite signs). What does not hold is "helps
unsloth": its +2.4% throughput is inside the 3-6% run spread because the larger drafter's per-step
cost offsets its acceptance gain. daily −8.3% and RedHat −6.5% are likely real.
Honest statement: the bf16 drafter is net-negative for daily and RedHat and roughly neutral for
unsloth. The "rule falsified" conclusion (affinity must be measured per pair) still stands —
the original rule was fit to two points. **MEDIUM** for the retraction, **LOW** for "opposite".

**QUASAR −26% code decode via acceptance collapse:** acceptance 0.194 vs 0.271 is exact; the
throughput is decode_ss n=3. **MEDIUM-HIGH.** The mechanism claim ("drafter co-adapted to the
incumbent") is a plausible reading, not a demonstrated one.

**"RedHat is the fastest of the three" — NOT established.** It rests on decode_ss c8 1212 vs 1148
(+5.5% over unsloth) at n=3 with the 3-6% run spread; kelnei was measured once. The spec-OFF kernel
instrument — the only HIGH-confidence decode tool today — never ran for unsloth or kelnei. Correct
form: tied on quality; RedHat *nominally* faster, unconfirmed; the candidate is "one of the three".

**Temperature regime.** Production serves at T=0.6. decode_ss samples at T=0.6, where acceptance is
probabilistic; decode_bench (and llama-benchy by default) run T=0, where a draft is accepted only if
it equals the argmax. Spec-ON throughput at T=0 is therefore not the production number. Prefer
decode_ss for spec-ON, or benchy with `--extra-body '{"temperature":0.6}'`.

## 3. Pool — HIGH, with one unverified constant

746,849 vs 654,491 (bench2/benchy, syvai, tier off) and 720,809 vs 628,798 (perf arms): −12.4% /
−12.8%, consistent across boots. The VRAM-accounting check (Δweights / Δpool = 35,089 B/token vs
32,768 theoretical, ratio 1.07) is suggestive but the "theoretical" used **kv_heads=8, head_dim=128,
16 full-attn layers, which I asserted rather than read from config.json**. The 48-GDN count is
verified (preserved-module lists); the head geometry is not. An earlier FINDINGS entry derived
"28 KB/tok" for fp8 KV by a cruder method. Treat 1.07 as "plausible", not "proven". **Unverified.**
The pool comparison itself does not depend on this.

## 4. Task gates — HIGH that they resolve nothing; my own reading of them was wrong twice

- tool-eval x4: 90.2 ± 1.0 vs 90.0 ± 1.4 — flat.
- GSM8K: daily 0.876 (n=250), RedHat 0.8583 (**n=120**, not 250 — my significance table assumed
  250; the RedHat CI is ±6pp, conclusion unchanged), unsloth 0.852 (n=250). MDD ~8-10pp. Nothing
  distinguishable.
- I read these as "supports the switch" (RedHat "passes every gate") in the afternoon and as
  "does NOT support the switch, lean against" in the evening. **Both were reading noise.** The only
  correct statement: the gates are structurally blind at this effect size. Retracted in FINDINGS
  (R156b CORRECTION); the memory file still carries the "LEAN AGAINST" — fixed in this commit.
- No `--log_samples` in any R156 lm_eval run, so no paired test is possible after the fact.

## 5. Depth / "KV quant does not compound" — MEDIUM, worded too strongly

Powered design: 5 bases × 50 continuations per depth, prefix-cached. Flip rate flat at ~22-24%
(dip to 11% at 28K in BOTH arms = content). Two limits the FINDINGS understates:
1. **n=250 overstates independence** — there are 5 long contexts per depth, not 250. A
   context-specific compounding effect is sampled at n=5.
2. Wilson CI at 23%/250 is ±5pp, so "flat" means "no trend larger than ~5pp in flip rate". The
   depth records carry `tok_lp: None` — no per-depth PPL/KL, only argmax flips, which is the
   coarsest possible detector of drift.
Honest form: "no dramatic compounding to 171K; small compounding not excluded." The 40-sample
"trend" was correctly retired.

## 6. R155 (Bug B) — decision fine, two labels too strong

- "Root cause PINNED to a captured-graph NVFP4 scale-base pointer" — the compute-sanitizer run
  that would confirm it never ran (deferred after the Xid31 incident). Strong hypothesis
  (M*=4929 pre-registered and hit; allocation-presence trigger; XQA-off green) — call it that.
- The "−37% single-stream code decode (c1 189 vs ~299)" that fed the park-it decision is from
  decode_ss's unreliable c1 cell on both sides. Direction is supported by exact acceptance
  counters (0.29 vs 0.39); the magnitude should not be quoted. The user's decision also rested on
  quality/c16 being unmeasured, so it stands.

## 7. Process — self-inflicted error count for one day

daily crash (port isolation + no memory math) · KVD `auto`→fp8 duplicate arm (caught by pool
identity) · arm N mislabeled from a README · three queue collisions · a zero-byte script that
passed `bash -n` · a decode probe that counted chunks · a second decode probe design that was
confounded by construction · a probe overwritten mid-chain · a false "no /metrics" claim · a
false critique of decode_ss · noise read as signal (twice, opposite directions) · an unverified
KV-geometry constant · a Python-heredoc edit script that died on a stray line while the surrounding
commit succeeded (caught by the content check this section prescribes). What worked: noise-floor-first, the alignment guard, the pool-identity rule,
prompt written retractions, falsified predictions logged. The dominant failure mode is the same
each time: **building or asserting instead of checking what already exists** (benchy, decode_ss,
/metrics, config.json, the safetensors index).

## 8. What survives, and what to do

Survives (HIGH): the fidelity gap and its size on raw text; fp8-KV ≈ free; lm_head ≈ 0.75%;
three mixed-precision checkpoints tied at the floor; RedHat kernel −18.6%/−13.5%; pool −12.4%;
task gates blind.

Undetermined: whether the fidelity gap has ANY behavioural consequence in the deployed regime;
RedHat's single-stream spec-on decode cost.

Cheapest high-value next step (≈2h GPU, same scorer): **bf16 generates the reference.** Take
held-out agentic PROMPTS (chat-templated, with tools and thinking), let the bf16 arm generate the
responses (~50 tok/s, no spec — that is the 2h), then teacher-force every quantized arm on bf16's
generations. That measures "how well does each quant reproduce the true model in the deployed
regime". NOT "re-score our own transcripts": if their assistant turns came from the daily, the daily
wins by construction; if from another model, you measure that model's likeness, not bf16 fidelity. Second: one benchy-h run (natural
mode, c1) to close the single-stream number. Third: read `/metrics` acceptance directly instead
of inferring it.
