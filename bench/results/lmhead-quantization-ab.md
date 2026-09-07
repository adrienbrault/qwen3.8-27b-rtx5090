# lm_head quantization, a controlled A/B: the fidelity gain sits only in low-confidence tokens (2026-08-25, `results/2026-08-25-lmhead-ab`)

[← all results](../RESULTS.md)

RadixArk published a BF16-lm_head variant of their Qwen3.8-27B NVFP4 checkpoint, byte-identical otherwise, advertising a significant accuracy improvement. That gives a controlled single-variable experiment for the rule that lm_head should never be quantized. Both checkpoints booted on the identical engine config (65K, NVFP4 KV plus tiers plus V2) and were screened on the fidelity ruler against the FP8 reference:

| | NVFP4 lm_head | BF16 lm_head |
|---|---|---|
| top-1 agreement | 0.9013 | 0.9147 |
| confident flips (ref p>=0.9, 255K toks) | 0.93% | 0.92% |
| flips at p 0.3-0.6 | 20.5% | 17.5% |
| KV pool @65K | 167,836 | 115,087 |
| decode c1 prose / code (t/s) | 121 / 164 | 98 / 121 |
| prefill pp8K (t/s) | 8.9K | 8.6K |

The fidelity gain is real (+1.3 pts agreement) but lives entirely in low-confidence tokens: where the reference is confident, the quantized head **flips nothing the bf16 head does not also flip** (0.93% vs 0.92%). The price is 20-26% of decode, since the bf16 head is about 1.5 GB read per step and it reduces MTP acceptance on code, plus about 53K tokens of KV pool. Prefill is untouched, because the head does not run there. For temperature-sampled agentic serving, the rule against quantizing lm_head is falsified on this stack: an NVFP4 lm_head is close to free where it matters. Greedy-decode evals are maximally sensitive to the low-confidence tie-breaks that do move, which is the likely source of the significant-improvement claims.
