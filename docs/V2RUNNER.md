# The V2 model runner + the 2026-08-21 nightly: +16–20% decode on the tier daily (measured 2026-08-21)

**Status: promotion candidate, measured, not yet the daily.** This came out of the DFlash2 audition ([docs/DFLASH2.md](DFLASH2.md)): the drafter's apparent +34% was the runner and the newer nightly. Re-measured with the daily's own stack — fp8 KV + LMCache tiers + MTP `ns=4` — on nightly `ba07e4a48` with `VLLM_USE_V2_MODEL_RUNNER=1`.

| tier stack (mnbt 3231, util 0.95, 200K, tiers ON) | current daily (V1, `ac7509e2b`) | V2 runner, `ba07e4a48` |
|---|---|---|
| KV pool | 201,408 | **209,859** |
| decode c1, pp8192 | 128.3 t/s | **152.2** (+19%) |
| decode c4 | 309.2 | **359.9** (+16%) |
| decode c2 / c8 | — | 248.8 / 367.1 |
| decode c1, pp30000 | 118.9 | **143.0** (+20%) |
| prefill c1 8K / 32K / 100K | — | 9.7K / 8.5K / 5.0K |
| needles 9K→100K, cold + warm (L2) | 10/10 + 10/10 | **10/10 + 10/10**, warm 0.5–1.5 s |
| restart-proof L2 (40K / 60K revisit after `docker restart`) | 4–7 s | **3.3 / 4.7 s**, correct |
| killer 8×24K | 8/8 | **8/8** |
| tool-eval 69×2 | 92.5 ± 0.7 (08-15) | **91 ± 0.0** |

Both columns measured within the same hour on the same card. The vLLM-side tier patches (`0005`, `0010`) are scheduler patches and apply unchanged; LMCache 0.5.4rc4's MP connector attaches to the V2 runner as-is. The two nightlies are 276 commits apart (Mamba state-copy race fix #50729, GDN spec-count #53077, FlashInfer 0.6.17, deterministic prefix-cache hashing #51875, `prefix_cache_retention_interval` default 0 #52216, …); the decomposition on the plain profile was nightly ≈ +7%, V2 runner ≈ +20% on top.

Two things to know before flipping: the nightly's `_C_stable_libtorch` extension raised a spawn-child `ImportError` (undefined cutlass symbol) once in six boots — a retry boots clean, so a health-loop launcher covers it; and at util 0.98 on the plain profile the V2 runner's first request OOMed inside the FlashInfer `fp4_gemm` autotuner, so keep util at 0.95 (the tier profile already is).

Recipe: `patches/lmcache/Dockerfile` with `--build-arg VLLM_BASE=vllm/vllm-openai@sha256:4e9299fb10c93ba020fbbe3237f7b5998d96cfe9fae962319babc9d7796ea66e`, then `serve.sh` with `VLLM_USE_V2_MODEL_RUNNER=1` in the container environment and a **fresh L2 directory** (a new stack generation never shares a namespace with the old one).
