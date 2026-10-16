# r27: vLLM v0.31.0 with FlashInfer 0.7 and the split-KV override promoted to the served configuration (2026-10-15 22:20 UTC to 2026-10-16, results `2026-10-15-r27-audition-screen-222033-601088`, `2026-10-16-r27-audition-profile-035416-3692711`, `2026-10-16-r27-audition-profile-063752-861443`, `2026-10-16-r27-audition-screen-065122-1708414`, `2026-10-16-r27-audition-qualify-085528-807233`, `2026-10-16-r27-audition-qualify-110820-405361`, `2026-10-16-r27-audition-qualify-123522-1448554`, `2026-10-16-r27-e-h2-20261016T174241Z-3271761`, `2026-10-16-r27-e-h2-20261016T175724Z-4183679`, [scripts/serve-v031-splitkv-daily.sh](../../scripts/serve-v031-splitkv-daily.sh))

[← all results](../RESULTS.md)

The served engine moved from vLLM v0.29.0rc2 with FlashInfer 0.6.16.post3 ([scripts/serve-r231-nvidia-daily.sh](../../scripts/serve-r231-nvidia-daily.sh)) to vLLM v0.31.0 with FlashInfer 0.7.0.post1, the version v0.31.0 pins. The patch chain was rebased onto v0.31.0 as [patches-v0310/](../../patches-v0310/) ([PORT-STATUS.md](../../patches-v0310/PORT-STATUS.md) says what each v0.29 patch became). The checkpoint, the 14.86 GB KV pin and its 1,391,795-token pool, 16 sequences, the MTP head at 3 draft tokens, the NVFP4 KV cache, the bf16 linear-attention state, the `pcie_ipc` all-reduce, batch-sharded sampling and the KV tiers did not change.

The launch differs from the v0.29 launcher in four items besides the image:

| item | value | why |
|---|---|---|
| `VLLM_SM12X_NVFP4_PREFILL_SPLIT_KV` | `1` | patch 0136 turns back on the split-KV path that FlashInfer 0.7 turns off for an NVFP4 KV cache on SM12x (below) |
| `VLLM_USE_V2_MODEL_RUNNER` | `1` | the model runner every v0.31 arm of this round ran |
| `--mamba-backend` | `triton` | the Mamba state backend every v0.31 arm ran |
| `--gdn-prefill-backend` | `flashinfer` | the GDN prefill backend every v0.31 arm ran; vLLM v0.31.0 admits FlashInfer's GDN prefill on SM120 |

Image `vllm-qwen38:v0310-r27c-A_S-splitkv` (sha256:2a081946bc9c0b6694cb246391a91ba19995d7dd55245afabdb282f97cf1808e, built 2026-10-16 04:27 UTC) is the v0.31 chain image ([patches-v0310/Dockerfile.v0310](../../patches-v0310/Dockerfile.v0310)) plus one Python layer ([patches-v0310/Dockerfile.A_S-splitkv](../../patches-v0310/Dockerfile.A_S-splitkv)). The screen, profile and qualify runs used a harness that is not in this repository. The launcher's argv was checked offline against the harness's resolved launches: it differs from the v0.29 launcher by exactly the image and the four items above, which is what the candidate's launch differed by from the v0.29 control's. The harness added diagnostic settings to both arms (core dumps, FlashInfer debug logging, per-arm caches, an explicit compilation config equal to the v0.31 defaults on this card: full-and-piecewise CUDA graphs, every fusion pass off).

## Plain v0.31 decoded slower, and one kernel accounted for it

The first screen ran two adjacent pairs, the v0.29 launcher (A0) and the v0.31 image without the override (A), on 2026-10-15 22:20 to 23:13 UTC (results `2026-10-15-r27-audition-screen-222033-601088`). Decode aggregates at 8 streams, `decode_ss.py`, greedy, 1,024 forced tokens per stream, three runs per cell:

| cell | A0, two boots | A, two boots | change per pair |
|---|---:|---:|---:|
| code, 8 streams | 1,563 / 1,570 t/s | 1,403 / 1,463 | −10.2 %, −6.8 % |
| prose, 8 streams | 1,301 / 1,277 | 1,219 / 1,218 | −6.3 %, −4.6 % |

A torch-profiler capture of 80 decode steps at 8 streams on one boot of each (2026-10-16 03:54 to 04:13 UTC, results `2026-10-16-r27-audition-profile-035416-3692711`) put the unprofiled step at 14.816 ms on A0 and 15.702 ms on A (+0.887 ms, +5.98 %). By kernel family on rank 0, attention and KV cache went from 480 to 1,422 µs per step (+942; rank 1 +909), the NVFP4 GEMMs from 4,696 to 4,651 (−44), FP8 GEMMs +8, GDN −26, all-reduce −4. The attention difference is one kernel, the MTP verify attention on the NVFP4 cache (`BatchPrefillWithPagedKVCacheKernel`, causal mask, bf16 queries, FP4 E2M1 KV): 17 calls per step in both arms, 380 µs per step on A0 and 1,354 on A.

FlashInfer 0.7's `BatchPrefillWithPagedKVCacheWrapper.plan()` sets `disable_split_kv=True` whenever the KV dtype is `uint8` (NVFP4) and the device is SM 12.0 or 12.1 (`_nvfp4_kv_requires_disabled_split_kv`, an upstream workaround for corruption reported on an asymmetric 512/256 head-dimension NVFP4 path; this model's is 256/256). Without split-KV each call runs one block per request and KV head over the whole KV range. The profile shows it: the split-KV merge kernel runs 19 times per step on A0 and 2 times on A. The same guard, present since the FlashInfer 0.6.17/0.6.18 releases, is why the v0.29 image pins FlashInfer 0.6.16.post3 ([R168](r168-029-program.md)). Patch 0136 replaces the guard with a function returning false when `VLLM_SM12X_NVFP4_PREFILL_SPLIT_KV=1`; its replacement took one argument, and FlashInfer 0.7 calls the guard with two (`kv_data_type, device`), so with the knob on the first `plan()` would raise `TypeError`. The A_S layer changes that replacement to accept any arguments.

## The override restores the v0.29 step

Profile of A against A_S (A plus the layer and the knob), 2026-10-16 06:37 to 06:51 UTC (results `2026-10-16-r27-audition-profile-063752-861443`):

| | A | A_S |
|---|---:|---:|
| split-KV merge calls per step, rank 0 / rank 1 | 2 / 2 | 19 / 19 |
| verify attention, µs per step, rank 0 / rank 1 | 1,377 / 1,361 | 380 / 392 |
| decode step at 8 streams, unprofiled | 15.861 ms | 14.606 ms (−7.92 %) |

A0's verify attention in the earlier profile was 380 / 392 µs per step and its step 14.816 ms, on another boot.

## Screen against the v0.29 launcher: no loss

Boots A0, A_S, A0, A_S on 2026-10-16 06:51 to 07:44 UTC (results `2026-10-16-r27-audition-screen-065122-1708414`), same instrument as above. Median of the two paired changes, A_S against A0:

| | 1 stream | 4 streams | 8 streams |
|---|---:|---:|---:|
| code | −1.25 % (noise) | −0.72 % (noise; pairs +1.7 %, −3.1 %) | +2.44 % (tie) |
| prose | +5.0 % (noise) | +0.57 % (tie) | +0.14 % (noise; pairs within ±1 %) |

No cell reads as a loss; the screen's verdict is noise. The qualify ladders below are the fidelity reading.

## Qualify: every gate passed, twice

A_S on 2026-10-16 08:55 to 09:52 UTC (results `2026-10-16-r27-audition-qualify-085528-807233`). Every gate passed; the unit then stopped on a harness timeout while formatting its evidence (exit 124), after the gates, and was re-run.

| gate | result |
|---|---|
| boot | pool 1,391,795 tokens, attention block 1,472 tokens, CPU tier 595 chunks, 5,015 MiB free on the fuller card before any request, split-KV override logged, `pcie_ipc` all-reduce with the MTP head admitted, batch-sharded sampling |
| tool-eval 69 × 4 | 90.5 ± 1.3 (95 % interval 89.5 to 91.5; trial points 123, 127, 124, 126) |
| needles at 131K and 220K prompt tokens | 4 of 4 cold; 4 of 4 again after a flood of 20 unrelated 90K prompts, all 4 served from the disk tier |
| warm resend | 3 of 3 completions identical to their cold answers, 4,416 prompt tokens from the prefix cache each |
| indentation probe | 60 of 60 answers indented, none empty |
| dense ruler vs bf16, 555,549 positions, four ladders | identical on all four: top-1 90.71 %, perplexity 5.5537 against 5.4501 (+1.90 %), truncated KL 0.0226 |
| boundary reasks (prefix-cache resends around block boundaries) | passed |

The same gates passed again on 2026-10-16 11:08 to 12:31 UTC and 12:35 to 13:58 UTC (results `2026-10-16-r27-audition-qualify-110820-405361` and `2026-10-16-r27-audition-qualify-123522-1448554`), dense perplexity +1.90 % in both. The v0.29 launcher's dense ladders read +1.825 % and +1.831 % in the first screen; A_S is 0.07 % above their mean, inside the 0.10 to 0.15 % spread between two boots of one configuration.

## The v0.29 control crashed: FlashInfer 0.6.16 frees cuDNN FP8 plans that CUDA graphs replay

In both later qualify runs the v0.29 control died on its second dense ladder at the same document: a burst of Xid 13 on both GPUs (12:13:41 and 13:40:41 UTC), 384 lines per GPU, with a different exception type in each run. Document 340 has 111 tokens and is the only document of the ladder short enough to replay through a captured CUDA graph; it had passed on the first ladder. cuda-gdb on the 13:40 core names a cuDNN-generated `matMul_pointwise_pointwise` kernel, FlashInfer's cuDNN FP8 GEMM, which vLLM reaches through `FlashInferFP8ScaledMMLinearKernel` → `bmm_fp8` → `CudnnFp8GemmRunner` for the checkpoint's FP8 layers.

FlashInfer 0.6.16 builds one cuDNN FP8 plan per (M, N, K, tactic) behind an `lru_cache(maxsize=2048)`. Plans built during CUDA-graph capture are baked into the captured graphs. Each eager prefill of a new length adds about three keys per rank; once the cache is full it evicts the oldest plan, and the next replay of a graph that captured it runs freed code. The reproduction (2026-10-16 17:42 and 17:57 UTC, results `2026-10-16-r27-e-h2-20261016T174241Z-3271761` and `2026-10-16-r27-e-h2-20261016T175724Z-4183679`, 900 prefills of distinct lengths with short probes in between, three arms):

| arm | result |
|---|---|
| v0.29 image, unchanged | the first capture-time plan (M=128, K=5120, N=8192, a linear-attention input projection) was evicted after distinct length 677; 23 prefills later the next short probe faulted with Xid 13 (request 754) |
| v0.29 image, capture-time plans pinned in the cache | eviction pressure reached the same point at length 720; 900 of 900 prefills and all 80 probes clean |
| v0.29 image, FlashInfer FP8 kernel off (CUTLASS FP8) | 900 of 900 and all 80 probes clean, no cuDNN plan built |

The v0.29 launcher is therefore exposed after about 680 distinct eager prefill lengths since boot. No Xid fell inside a serving window of the 27B daily in the host's records. FlashInfer 0.7 removes cuDNN from the `bmm_fp8` auto candidates on SM12x while cuDNN's override-shape support is missing (`gemm_base.py`, the per-shape graph build is "unbounded at serving time"); every v0.31 boot logs `Skipping cuDNN in bmm_fp8 auto candidates on SM12x: override_shape unavailable (cuDNN backend < 9.23.1)`, so the served image never builds these plans. The pinning fix is not served; it would matter only for an image that stays on FlashInfer 0.6.16.

## Not yet measured on the served launcher

The decode and prefill curves ([R675](r675-27b-curves.md)), the standard benchmark ([R793](r793-27b-std-bench.md)), SWE-Bench Verified ([R231](r231-promote-nvidia.md)), the bf16 decode ruler and the long-prompt residency stress were measured on the v0.29 launcher and have not been repeated. The README names them with that configuration.
