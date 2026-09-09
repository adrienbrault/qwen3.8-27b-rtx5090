# R231 and R234: the NVIDIA checkpoint promoted to the served configuration, then the KV pin raised (2026-09-09 10:11 to 13:38 UTC, results `2026-09-09-r231b-promote-nvidia`, `2026-09-09-r232-nvidia-pool-ladder`, `2026-09-09-r233-nvidia-pin-clean`, `2026-09-09-r234-promote-pin`, [scripts/serve-r231-nvidia-daily.sh](../../scripts/serve-r231-nvidia-daily.sh))

[← all results](../RESULTS.md)

The served checkpoint changed from [RedHatAI/Qwen3.8-27B-NVFP4](https://huggingface.co/RedHatAI/Qwen3.8-27B-NVFP4) to [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4), and the KV pin from 13.98 to 14.86 GB per GPU. Nothing else moved: same image, same NVFP4 KV dtype, same MTP head at 3 draft tokens, TP2, 16 sequences, same tiers and kernels.

The two checkpoints differ in three ways. NVIDIA puts NVFP4 group-16 on all 64 MLP layers where RedHat leaves 56 to 63 in FP8; its lm_head is NVFP4 where RedHat's is FP8; and it carries an input scale on all 401 quantized layers, so activations are quantized too (W4A4). It is a ModelOpt 0.47.0.dev80 checkpoint with `kv_cache_quant_algo: null`, so the KV dtype is the launcher's, not the checkpoint's. It served on `FlashInferCutlassNvFp4LinearKernel`, not Marlin.

## What it costs: fidelity

| ruler | RedHat (previous daily) | NVIDIA (promoted) |
|---|---:|---:|
| dense text, top-1 agreement vs bf16 | 92.79% | **90.67%** |
| dense text, perplexity vs bf16 | +0.83% | **+1.83%** |
| dense text, truncated KL | 0.0143 | **0.0226** |

Two boots of one configuration differ by 0.10 to 0.15% on the perplexity ruler, so the gap is real and about seven times the noise floor. The checkpoint was promoted anyway, deliberately, with that number in hand.

The agentic ruler has not been re-measured on this checkpoint; the RedHat figure in the README's fidelity row is not a NVIDIA number and is marked as such.

The bf16 *decode* ruler disagrees with the dense one, which is worth recording: median absolute delta-logprob 0.00046 at no context and **0.00516 at 30K**, inside the 0.0051 to 0.0062 band of the previous route and slightly closer than its 0.00592. Prefill-shaped and decode-shaped rulers are not measuring the same thing.

## What it does not cost: everything else measured

Paired against RedHat on the experiment port in the same hour, same image, route and sequence limit, only the checkpoint differing:

| row | RedHat | NVIDIA | change |
|---|---:|---:|---:|
| decode, code, 1 stream | 205.5 t/s | 202.1 | −1.7% |
| decode, code, 8 streams | 1,531.7 | 1,468.8 | −4.1% |
| decode, prose, 1 stream | 168.6 | 167.1 | −0.9% |
| decode, prose, 8 streams | 1,230.6 | 1,229.4 | −0.1% |
| KV pool | 1,309,368 | 1,309,368 | identical |
| free VRAM after pre-warm | 3,533 MiB | 4,413 MiB | +880 |

The code deficit is draft acceptance, not kernel speed: 0.635 and 0.650 accepted per draft against RedHat's 0.664 and 0.697. Each checkpoint ships its own MTP head, so that is the head differing, and it is the one thing the swap genuinely changes about decode.

That paired run also re-measured RedHat on the instrument that produced the published rows, and reproduced them: 205.5 against 209 published, 1,531.7 against 1,539, 1,230.6 against 1,231. Without that check the NVIDIA rows would not be comparable to the ones they replace.

SWE-bench Verified reads 387 of 500 (77.4%) on this checkpoint, against 386 to 388 for three other checkpoints spanning the whole fidelity range above. The benchmark does not adjudicate quantization on this model; the bf16 rulers do.

Tool-eval is unsettled. Two readings on this checkpoint, same instrument, 69 scenarios x 4 trials: 88.5 +- 0.6 and 90.5 +- 3.7, against 91.2 +- 0.5 for RedHat. The second interval contains both the first reading and RedHat's, and the per-trial scores were [123, 122, 123, 122] on one run and [127, 131, 122, 120] on the other. A single run of this benchmark is not enough to establish a gap of this size on this stack.

## Gates on the serving port

| gate | reading |
|---|---|
| tier | wiped on the checkpoint stamp, as designed: the previous checkpoint's KV blocks cannot be served to different weights |
| KV pool | 1,309,368, 4,479 MiB free |
| 120K prompt, 5 concurrent | all five resident at 62.5% pool, no preemptions |
| needles at 131K and 220K | 4 of 4 cold, then 4 of 4 from the tier after a flood of 16 unrelated 90K prompts |
| indentation probe, 60 answers | 59 with indentation, 0% single-space-dominant |
| engine errors, preemptions | none |

## Raising the KV pin: the limit is boot reliability, not memory

The pool follows `--kv-cache-memory-bytes`, not the weights. So the NVIDIA checkpoint's smaller weights produced 880 MiB more free VRAM and exactly the same pool, until the pin moved. `--gpu-memory-utilization 0.88` is not a second ceiling under a pin: RedHat served at 32,607 − 3,533 = 29,074 MiB used, above 0.88 × 32,607 = 28,694.

A ladder over four pins found no memory ceiling. 14.86 GB gave pool 1,391,795 with 3,579 MiB still free through a stress of five concurrent 120K prompts and one 250K prompt. 15.90 and 17.90 GB failed warmup with `CUDA error: invalid argument` while the larger 16.90 GB booted twice — non-monotonic, so not a memory limit.

Booting each pin three times, with the tier wiped before every boot and a fresh seed per boot, gives the rate:

| pin | clean boots | worst free VRAM during stress | pool |
|---|---|---:|---:|
| 14.86 GB | 1 of 1 | 3,579 MiB | 1,391,795 |
| 15.90 GB | 2 of 3 | 2,485 MiB | ~1,489,000 |
| 16.90 GB | 1 of 3 | 1,371 MiB | 1,583,674 |

Every pin held far above the 1,000 MiB bar, so VRAM never decided this. The warmup flake rate rises with the pin, and a 1-in-3 flake makes every restart a coin toss. 14.86 GB is the largest pin with a clean boot record, and it leaves the daily at the headroom the previous checkpoint already served in production.

The first version of that ladder was wrong and is worth recording. The capacity probe took a fixed seed on every arm while the experiment tier persists across boots, so the first arm populated the tier and later arms were served from it: 208,435 tokens in 3.3 seconds, 63,162 tokens per second against a measured cold prefill of about 4,200 at that depth. Their headroom readings were taken under a stress four times shorter and were not comparable. The re-measurement wipes the tier before every boot, carries a per-boot seed, and fails any arm whose 250K prefill exceeds 10,000 tokens per second.

## Gates at the new pin

Pool 1,391,795 exactly, first-attempt boot, 3,419 MiB free. The tier survived the change — block size and checkpoint stamps both unchanged, 152G to 156G — so no warm context was lost. Five concurrent 120K prompts at 58.8% pool with no preemptions; a 250K prompt at 23.1%; cold prefill verified at 3,826 tokens per second. Needles 4 of 4 cold and 4 of 4 after eviction, with the eviction flood raised to 19 unrelated 90K prompts: 12 of them total 1.08M tokens and cannot evict a 1.39M pool, so the re-asks would have been served from the pool itself and misread as tier hits.

Decode is unchanged by the larger pool:

| row | pin 13.98 | pin 14.86 |
|---|---:|---:|
| code, 1 stream | 227.2 t/s | 216.0 |
| prose, 1 stream | 170.0 | 164.5 |
| prose, 1 stream at 30K | 156.6 | 157.6 |
| code, 8 streams | 1,482.2 | 1,475.9 |
| prose, 8 streams | 1,249.9 | 1,267.2 |
| code, 16 streams | 2,594 | 2,595.7 |
| prose, 16 streams | 2,174 | 2,183.0 |

Every delta is inside the run-to-run spread of a single arm — the 1-stream code row alone ranged 176.7 to 224.1 within one measurement. 82,427 extra tokens of pool cost nothing measurable.

All numbers on this page were measured at stock power limits (600 W and 575 W). The daily runs capped at 400 W; the promotion gates lift the cap for the measurement and restore it afterwards.
