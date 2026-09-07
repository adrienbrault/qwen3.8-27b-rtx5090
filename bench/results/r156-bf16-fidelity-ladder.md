# Fidelity ladder against a bf16 reference: the daily checkpoint ranks last (2026-09-01, `results/2026-09-01-r156-bf16-ladder`)

[← all results](../RESULTS.md)

The task gates (tool-eval ±2–3, GSM8K n=250 with an approximately 8 pp minimum detectable difference) had reported every NVFP4 checkpoint as acceptable. This experiment tested whether those gates were blind, and they were.

Method (`scripts/fidelity_ladder.py`, `fidelity_compare.py`, `build_fidelity_corpus.py`): a 693-document pinned corpus (Python dist-packages, wikitext-103, GSM8K-train; about 1K tokens each), scored teacher-forced via `prompt_logprobs` at concurrency 1 with speculation off, on every arm, for 724,781 positions per arm. The reference is `Qwen/Qwen3.8-27B` bf16 at TP=2 with bf16 KV. Metrics: corpus perplexity delta, and top-1 flip rate bucketed by the reference's confidence. The noise floor, from the same arm across two boots, is 0.003% flips in the certain bucket, roughly 100x finer than tool-eval.

| checkpoint | PPL delta vs bf16 (5.4501) |
|---|---|
| unsloth Dynamic V3.0 NVFP4 | **+0.37%** |
| kelnei NVFP4 | +0.37% |
| **RedHatAI NVFP4** | **+0.38%** |
| fp8 weights (reference quant) | +0.43% |
| RadixArk NVFP4, bf16 lm_head | +1.91% |
| QUASAR QAT NVFP4 | +2.12% |
| RadixArk NVFP4 | +2.77% |
| gittensor + fp8 lm_head | +3.71% |
| gittensor + bf16 KV | +4.38% |
| **gittensor (the daily until 09-02)** | **+4.46%** |

Controlled decompositions: 4-bit `lm_head` is about 0.85 pp (RadixArk pair; an independent family gives 0.72), fp8 KV 0.13 pp, and nvfp4 KV 0.76 pp. Neither KV scheme compounds with context out to about 171K (five depths). The remaining 3.6 pp is the quantizer itself: unsloth, kelnei and RedHat share one llm-compressor recipe family (303 modules preserved at 8-bit vs gittensor's 148) and reach +0.37% at the same 4-bit width.

Caveat, then closed: that corpus is raw, largely memorised text with no chat template, tools or thinking, so it is a weight-fidelity ruler rather than the deployed regime. The second experiment (`scripts/agentic_ref.py`, `build_agentic_prompts.py`, `agentic_by_kind.py`) therefore let bf16 generate greedy responses to 72 held-out chat-templated prompts (32 tool-call tasks with a 5-tool schema, 16 code, 16 reasoning, 8 prose; 57,972 positions). Each candidate was then teacher-forced on bf16's exact token ids, so neither arm authored the text.

| arm | top-1 agreement with bf16 | PPL delta | flips where bf16 was moderately sure (0.5 < p ≤ 0.9) | near-tie flips |
|---|---|---|---|---|
| gittensor | 92.60% | +6.56% | 8.57% | 35.9% |
| **RedHatAI, fp8 KV** | **95.95%** | **+2.41%** | **3.36%** | 22.1% |
| RedHatAI, nvfp4 KV | 95.57% | +2.65% | 3.81% | 24.0% |

Per kind the ratio is uniform (moderate-bucket flips gittensor → RedHat: tool 8.3 → 3.2%, code 8.8 → 3.7%, reason 9.0 → 3.2%, prose 8.0 → 2.8%). Both arms agree with bf16 almost always where bf16 was confident. On the decision points of a greedy trajectory the old daily left bf16's path 1 token in 12, and RedHat 1 in 30. Under teacher forcing each flip is contained, while under real generation each flip is a divergence, so this figure is a floor on behavioural divergence. (Reference PPL here is 1.29, bf16 scoring its own argmax path, so these relative deltas are not comparable to the raw-corpus column: compare arms to each other.)

Cost, measured on the exact daily shape (TP=2, util 0.92, fp8 KV, DFlash2 ns9, syv-ai drafter): the forward pass alone (spec off, content-independent) measured −18.6% c1 / −13.5% c8. RedHat's higher draft acceptance (+5–6%) buys a third of that back, so spec-ON decode measures **−6% c1** (llama-benchy, T=0.6, three instruments agree) and −7% c8. Prefill measures −14%, with no acceptance rebate. Pool measures −12.4% (654,491, still 2.5x the 262K max context), and TTFT @2K +33 ms. A task-outcome difference is not established, because no affordable task gate can see one either way. That is the trade the switch makes: a measured fidelity gain against quantified speed costs.

Drafter 2x2: quantized syv-ai W4A16 vs original bf16 z-lab drafter, on both targets, code c8. gittensor measured 1,299 (syv-ai) vs 1,191 (z-lab, −8.3%), RedHat 1,212 vs 1,134 (−6.5%), and unsloth was flat within run spread. Acceptance measures drafter-target agreement, not either party's quality. It is a property of the pair and is not predictable from target fidelity: RedHat and unsloth tie on fidelity yet respond differently. A mid-fidelity QAT checkpoint (QUASAR) lost 26% code decode this way. Re-run the drafter A/B on every checkpoint switch.

Measurement lessons (`docs/R156-REVIEW.md` records the audit): `ignore_eos` and `min_tokens` force generation past EOS into degenerate text whose draft acceptance is checkpoint-dependent, a −35%…+133% swing on the same arms. The fix is to decompose into a spec-off kernel rate plus a separately measured acceptance, or to use `llama-benchy --extra-body '{"temperature":0.6}'`, since production samples at 0.6 and acceptance at T=0 is argmax-only. Single-stream `decode_ss` at n=3 has ±50% spread and was excluded. A GSM8K difference of 1.8 pp at n=250 is noise in either direction, and it had been read as signal twice.
