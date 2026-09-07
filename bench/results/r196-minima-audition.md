# R196: the minima-ai all-NVFP4 checkpoint auditioned against the served RedHat checkpoint and rejected on fidelity (2026-09-05 14:22 to 14:42 UTC, results `2026-09-05-r196-minima-audition`, `scripts/r196-minima-audition.sh`)

[← all results](../RESULTS.md)

Two arms on the served route (port 8029, seven draft tokens, batch-sharded sampling, `pcie_ipc` all-reduce, pool 1,052,277 at the 13.98 GB pin), control first: H is [RedHatAI/Qwen3.8-27B-NVFP4](https://huggingface.co/RedHatAI/Qwen3.8-27B-NVFP4) (the served weights, 303 modules kept at 8 bits), M is [minima-ai/mnma_qwen3.8_27b_nvfp4](https://huggingface.co/minima-ai/mnma_qwen3.8_27b_nvfp4) (every linear in NVFP4 including the gated-delta-net projections, 18.8 GB against 22 GB). The rulers are the R156 bf16 dumps (dense corpus n=555,549 positions, bf16-generated agentic turns n=57,972, and the 20-chunk greedy decode ruler at ctx 0 and 30K). The run was stopped after the last decode probe of M; tool-eval did not run.

| ruler vs bf16 | H (RedHat, served) | M (minima) |
|---|---|---|
| dense corpus PPL delta | +0.745% | +4.361% |
| dense top-1 agreement (flips) | 92.774% (40,146) | 90.014% (55,475) |
| dense truncated KL mean / median | 0.0141 / 0.0062 | 0.0250 / 0.0114 |
| agentic corpus PPL delta | +2.699% | +5.930% |
| agentic top-1 agreement (flips) | 95.534% (2,589) | 93.083% (4,010) |
| decode ruler, fully agreeing chunks ctx 0 / 30K | 2/20 / 2/20 | 0/20 / 1/20 |

| decode (tokens/s, accept per draft) | H | M |
|---|---|---|
| code, 1 stream (3 runs) | 274 (0.409) | 305 (0.370) |
| prose, 1 stream | 172 (0.202) | 198 (0.192) |
| code, 8 streams | 1,591 (0.372) | 1,518 (0.329) |
| code, 16 streams | 2,429 (0.408) | 2,311 (0.358) |
| minimum free VRAM after boot | 2,073 MiB | 4,241 MiB |

Readings. The minima checkpoint sits six times farther from bf16 than the served checkpoint on the dense corpus and 2.2 times farther on the agentic turns, at the distance the gittensor checkpoint had before it was replaced (+4.46% on the same corpus, R156). It decodes faster at one stream (+11% code, +15% prose) with lower acceptance, so the target step itself is faster (about 85 against 71 steps/s), which is what quantizing the gated-delta-net projections buys; at 8 and 16 streams it is 5% slower. The 3.2 GB of smaller weights show up as free VRAM, not pool, because the pool is pinned in bytes. Rejected on the fidelity gate; the numbers stay here as the reference for an all-NVFP4 layout on this model. The H column is also the first fidelity reading of the served checkpoint on the seven-draft-token route (the dense delta was +0.67% on the nine-token artifact at R188; compile artifacts differ between the two).
