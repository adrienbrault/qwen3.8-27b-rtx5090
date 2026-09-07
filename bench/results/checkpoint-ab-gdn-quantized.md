# Checkpoint A/B on the daily engine: the GDN-quantized checkpoints gain decode and pool (2026-08-21/22, results/2026-08-21-radixark-ab, results/2026-08-22-sweep-ab)

[← all results](../RESULTS.md)

Same engine and flags as the daily (tiers, nvfp4 KV, V2 runner, 262K, util 0.93, LMCache chunk 2864), with only `MODEL_DIR` changing. The daily checkpoint keeps the 48 GDN layers' projections in bf16, about 11 GB read every decode step, while the two newer checkpoints quantize them.

| | saka (daily until 2026-08-22) | RadixArk (FP8 attention + GDN) | gittensor `-RTX5090` (GDN NVFP4) | Mantrah `-GDN` (GDN NVFP4) |
|---|---|---|---|---|
| KV pool @262K | 312,189 | cannot boot at 262K; 231,818 @200K | **397,982** | 364,618 |
| decode c1 / c4 / deep-30K c1 (t/s) | 143–150 / 339–363 / 142 | 171 / 332 / 156 (@200K) | **178 / 405 / 175** | 177 / 407 / 172 |
| prefill 8K / 30K (t/s) | 12.8K / 9.3K | 9.6K / 8.1K | 12.3–12.8K / 9.45K | 12.4K / 9.2K |
| needles cold + warm to 261.7K | all hit | 3/3 (@200K) | 4/4 | 4/4 |
| killer / vision / structured output | 8/8 / 8/8 / 4/4 | 8/8 / 8/8 / 4/4 | 8/8 / 8/8 / 4/4 | 8/8 / 8/8 / 4/4 |
| tool-eval 69×2 (Context & State) | 92 ± 1.4 (17–18/20); 69×4 90.0 ± 2.0 | 90.5 ± 0.7 (14/20) | 89.5 ± 2.1 (14/20) with the Qwen3.8 XML template | 89.5 ± 2.1 (16/20) |
| tool-eval 69×4 (decisive, 2026-08-22) | 90.0 ± 2.0 | — | **89.8 ± 1.3** | 89.5 ± 1.3 |
| SWE-Bench Verified, first 50 tasks, same harness (R2E-solved; pull-limit drops shrink the denominators) | 37/50 (74%) | — | 22/30 (73%) | 25/34 (74%) |
| KV pool on the daily config, decisive boot | ~310K | — | **388,449** | 347,936 |

kelnei/Qwen3.8-27B-NVFP4 (2026-08-23, `results/2026-08-23-kelnei-ab`): GPTQ-NVFP4 MLPs, FP8 attention and GDN projections, FP8 `lm_head`, 21.8 GB, same engine config. It refuses 262K, needing 5.75 GiB for one sequence against 4.26 free, and at 180K the pool is 182,222. Steady-state decode measured prose c1 108 / c4 440 and code c1 157 (gittensor 124 / 511 / 183) despite slightly higher MTP acceptance; prefill 9.2K @8K / 7.3K @30K (−30% / −22%); needles 3/3 to 111K; "killer" (a needle check) 8/8, vision 8/8, structured output 3/4; tool-eval 69×2 89 ± 1.4. Rejected, the same outcome as RadixArk: on this card FP8 GEMMs on the attention and GDN path cost pool and prefill and return no quality. HivenetQuant/Qwen3.8-27B-NVFP4 (same recipe class, 23.1 GB) was not run for that reason.

gittensor as the daily, measured on the serving port after promotion (2026-08-22 17:54 UTC, `results/2026-08-22-r90-gittensor-daily/`): decode c1 170 / c2 272 / c4 451 / c8 487 t/s aggregate (142 and 97 per stream at c4/c8); 187 t/s at 30K depth and 193 at 100K; prefill 13.2K / 9.4K / 4.9K t/s at 8K / 30K / 100K (TTFT 0.57 / 2.9 / 18.4 s); MTP acceptance 0.56 per draft token; depth needles 40K/100K/200K ×2 all hit cold and warm (cold 223K-token prompt: 91 s); tool-eval 69×2 89 ± 1.4.

Notes: gittensor ships a chat template whose tool-call format the `qwen3_xml` parser does not read, so tool calls come back with empty arguments and tool-eval 0; serve it with the stock Qwen3.8 template. Its DSpark NVFP4 drafter does not load under vLLM's `dspark` path (head-dim mismatch, and the card's numbers are SGLang). The GDN-NVFP4 checkpoints buy +20–25% decode and +17–27% pool at prefill parity, and the quality cost shows as a small Context & State dip at n=2. The daily stays on saka until the same-tasks SWE-Bench comparison (first 50 tasks) is in.
