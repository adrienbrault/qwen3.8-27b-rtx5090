# Four NVFP4 checkpoints compared on one ruler: gittensor kept as the daily (2026-08-23, `results/2026-08-23-nvfp4-quant-sweep`, `results/2026-08-23-fidelity`)

[← all results](../RESULTS.md)

All four Qwen3.8-27B NVFP4 checkpoints on disk were booted back to back on the identical daily engine (tiers, NVFP4 KV, V2 runner, 262K, util 0.93, MTP ns=4) and measured with the same probes. unsloth (22 GB, GDN in bf16) refuses 262K on this config (kelnei-class) and was taken at 180K.

**Serving** (`scripts/decode_ss.py` steady state, 3 runs; llama-benchy prefill):

| checkpoint | recipe | pool @262K | prose c1 / c4 / c8 | code c1 / c4 | pp8K / pp30K |
|---|---|---|---|---|---|
| [gittensor](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090) (daily) | NVFP4 everywhere incl. GDN | **388,449** | **132 / 540 / 900** | 169 / **675** | 12.7K / 9.4K |
| [Mantrah](https://huggingface.co/Mantrah/Qwen3.8-27B-NVFP4-GDN) | NVFP4 incl. GDN, FP8 lm_head | 347,936 | 127 / 487 / 857 | **182** / 590 | 12.3K / 9.2K |
| [saka](https://huggingface.co/sakamakismile/Qwen3.8-27B-MTP-NVFP4) | NVFP4 attn+MLP, GDN bf16 | 312,189 | 100 / 425 / 742 | 150 / 593 | 12.2K / 9.2K |
| [unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) | NVFP4 attn+MLP, GDN bf16, 22 GB | 182,222 @180K | 105 (@180K) | 162 (@180K) | — |

MTP acceptance is identical across the four (0.43/draft), so the decode spread comes only from the bytes the GDN projections read per step. Prefill is flat because attention and MLP GEMMs are NVFP4 everywhere.

**Fidelity to the unquantized model** (`scripts/fidelity.py`): a fixed 491K-token corpus (80 × 2048-token chunks each of code, wikitext-103, and SWE-bench agent trajectories) scored with `prompt_logprobs` on every checkpoint and on [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) as the near-lossless reference. The probe is deterministic, takes ~3 min per checkpoint, and resolves 0.1%. The constraints: only one request may be in flight (each holds ~1.2 GB of full-vocab logprobs, and four OOM the daily config), and the FP8 reference needs 512-token chunked prefill on the 5090 or a 2048-token `prompt_logprobs` pass materialises 1.45 GiB at once.

| vs FP8 reference | ΔNLL code / prose / agent | top-1 agreement | KL(ref‖m) |
|---|---|---|---|
| unsloth | +1.2 / +1.1 / +4.1 % | **0.924** | **0.106** |
| Mantrah | +1.8 / +3.1 / −1.0 % | 0.900 | 0.156 |
| gittensor | +2.6 / +3.4 / −0.8 % | 0.889 | 0.164 |
| saka | +3.4 / +4.7 / **−18.3 %** | 0.870 | 0.294 |

Quantizing the GDN projections costs ~1 point of top-1 agreement (gittensor and Mantrah vs unsloth), and gittensor and Mantrah match to 0.3% in every column. saka is the least faithful and is shifted rather than better: 18% lower NLL than the unquantized model on agent-trajectory text is impossible for a faithful quant. Its calibration sharpened it toward tool-output and agentic text, which is where its tool-eval (tool-calling benchmark, tool-eval-bench) edge (92 vs 89–90) comes from, at +3–5% NLL on plain code and prose.

**Confidence-bucketed flip profile** (bucket = reference's top-1 probability; flip = argmax disagreement; the confident bucket is mostly literal copies and syntax, where flips matter most):

| vs FP8 ref | p≥0.9 (255K toks) | 0.6–0.9 | 0.3–0.6 | p<0.3 |
|---|---|---|---|---|
| unsloth | **0.82%** | 5.1% | 15.7% | 28.2% |
| Mantrah | 1.13% | 7.4% | 20.8% | 36.2% |
| gittensor | 1.15% | 8.0% | 23.3% | 40.1% |
| saka | 1.73% | 11.4% | 26.5% | 43.8% |

Every quant flips mostly at uncertain positions. saka also overrides the reference's confident predictions at 2.1× the best quant's rate, so its calibration shift is not confined to positions where any answer would do.

**cyankiwi AWQ-INT4** ([cyankiwi/Qwen3.8-27B-AWQ-INT4](https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-INT4), G32 W4A16, no MTP head, the L1T guide's fidelity pick) was audited on the same engine (spec decode off, LMCache chunk 2784 to match the no-spec hybrid attention block). It measured the best fidelity of every checkpoint tested: top-1 agreement 0.934, KL 0.089, 0.54% confident-position flips, and the biggest pool (417,873 @262K, 15.5 GB weights). But W4A16 Marlin without MTP serves at half the decode (70 t/s c1, code = prose) and a third of the prefill (4.1K @8K, 3.7K @30K) against the NVFP4+MTP daily. Its tool-eval (69 tasks ×2) is 87 ± 4.2, the lowest measured (saka 92, the GDN-NVFP4 pair 89–90), so fidelity does not predict tool-calling: the most faithful checkpoint scores worst-in-band on tools while the deliberately shifted saka scores best. It is the fidelity pick for latency-tolerant batch and judging roles, not a daily candidate. Combined with the FP8-attention class result, the pattern is that on a 32 GB FP4-native card only GDN-NVFP4 checkpoints deliver pool, decode, and prefill together.

One instrument caveat: the screen is only meaningful against the checkpoint's own base. An abliterated checkpoint that turned out to be Qwen3.6-based ([llmfan46 heretic-v2](https://huggingface.co/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved), NVFP4 export) read ΔNLL +29% and 6.5% confident-position flips against the Qwen3.8 reference. That is base-model identity plus abliteration plus quant, not attributable damage. Cross-generation disagreement dwarfs quant effects, so finetunes must be vetted against their exact base.

**Task level, with resolution** (lm-eval 0.4.12 over `/v1/chat/completions`, c16, T=0.6, effort medium; GSM8K rescored with `scripts/gsm8k_rescore.py` because lm-eval's flexible-extract misses `**18**`, `70,000` and trailing-context numbers, reading 85–88 raw for every checkpoint):

| checkpoint | GSM8K (n=1319) | IFEval prompt-loose / strict (n=541) | tool-eval 69×N | SWE-Bench (50-slice) |
|---|---|---|---|---|
| unsloth | **97.3 ± 0.4** | 70.8 / 69.1 ± 2.0 | — | — |
| gittensor | 96.8 ± 0.5 | 70.1 / 68.0 ± 2.0 | 89.8 ± 1.3 | 22/30 |
| Mantrah | 96.4 ± 0.5 | 69.7 / 68.2 ± 2.0 | 89.5 ± 1.3 | 25/34 |
| saka | 94.5 ± 0.6 | **73.8 / 71.2** ± 1.9 | **92 ± 1.4** | 37/50; 331/500 full |

GSM8K tracks fidelity exactly, while IFEval inverts it (saka +3.7, ~1.9σ) in the same direction as its tool-eval edge. IFEval absolute levels are low for Qwen3.8 (thinking-mode formatting under effort medium; the setup is identical for all four, so only relative values apply). gittensor stays the daily: best pool and decode, fidelity and math within noise of the best, IFEval within 1σ of the cluster.
