# R201: Qwen3.8-27B-Kearuga audition, rejected (2026-09-05 22:41 to 2026-09-06 00:23 UTC, `results/2026-09-06-r201-kearuga`, [scripts/r201-kearuga-audition.sh](../../scripts/r201-kearuga-audition.sh), [scripts/merge_quantized_layers.py](../../scripts/merge_quantized_layers.py))

[← all results](../RESULTS.md)

0xWhiteMage/Qwen3.8-27B-Kearuga (revision 1a7f4231, 24.85 GB) is a ModelOpt mixed-precision export: 15 of 16 attention layers and 45 of 48 GDN layers in static FP8, MLP layers 2 to 61 in W4A16 NVFP4 (GPTQ requant), the four boundary MLPs in FP8, embeddings, lm_head, norms, vision and MTP head in bf16. It was measured on the daily route (same image, DFlash draft length 7, nvfp4 KV, 16 sequences, pool 1,052,277 tokens) against the R156 bf16 dumps, with the RedHat checkpoint from R196 as the control and the R197 draft-length-7 rows for decode.

Boot problems, in order:

- The checkpoint's `config.json` lists 375 quantized layers; `hf_quant_config.json` and the tensors list 385. The 10 missing entries are the FP8 boundary MLPs, so vLLM built them unquantized and the weight loader stopped on their `input_scale` tensors. `merge_quantized_layers.py` writes a hard-linked sibling directory whose `config.json` carries the union.
- Two runs lost the engine 4 minutes into the dense ruler's prefill, on different documents and different GPUs: an Xid 13 with a Triton "illegal instruction" inside the GDN core, then a CUDA "unknown error" in a piecewise-graph replay. Both runs served the static-FP8 layers with vLLM's FlashInfer FP8 scaled-MM kernel, which vLLM admits at compute capability 100 and above, so an RTX 5090 (sm120) gets the sm100 kernel. `--linear-backend cutlass` cannot boot this checkpoint (the cutlass set contains a W4A8 kernel, so the W4A16 filter does not fall back). `--linear-backend torch` selects the per-tensor torch FP8 scaled-MM kernel for the FP8 layers, leaves Marlin for the W4A16 MLPs, and ran the full battery with 0 engine errors and no Xid. This does not yet isolate the kernel: the two crashing runs shared one compile artifact and the torch-backend run compiled a fresh one (the flag changes the artifact hash), and this repo has recorded discrete numerics classes per fresh artifact before (R190c, R193). The discriminating test, still to run, is the FlashInfer kernel on a fresh artifact through the dense ruler. Until then FP8-attention candidates on this box run with `--linear-backend torch` as a precaution. Upstream issue vllm#52540 (open, no maintainer reply as of 2026-09-06) describes the same shape: the FlashInfer FP8 scaled-MM path grows its cuDNN workspace with `resize_()` during piecewise CUDA-graph capture, and later replays use freed memory; fix PR #52553 is open. This is consistent with the crashes but not separated by us from the compile-artifact effect. `VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel` is the narrower workaround available in v0.29.0rc2.
- 1,273 orphan `/dev/shm/psm_*` segments (18 GB, left by every crashed or removed engine over four days) blocked the crashed engine's restart. The base launcher now removes orphan segments before every boot.

Results, run 5 (torch FP8 kernel), with the RedHat control in parentheses:

| ruler | Kearuga | RedHat (R196) |
|---|---|---|
| dense PPL delta vs bf16 | +1.348 % | +0.745 % |
| dense top-1 agreement | 94.19 % | 92.77 % |
| dense truncated KL mean | 0.0082 | 0.0141 |
| agentic PPL delta vs bf16 | +1.424 % | +2.699 % |
| agentic top-1 agreement | 96.49 % | 95.53 % |
| decode ruler, fully agreeing chunks (ctx 0 / 30K) | 2/20 / 4/20 | 2/20 / 2/20 |

| decode row (tokens/s @ acceptance, steps/s) | Kearuga | daily, draft length 7 (R197) |
|---|---|---|
| code, 1 stream | 268.8 @ 0.413, 69 | 273, 71 |
| prose, 1 stream | 171.8 @ 0.215, 69 | 171 |
| code, 8 streams | 1,360 @ 0.380, 372 | 1,633, 442 |
| prose, 8 streams | 943 @ 0.223, 368 | 1,134, 442 |
| code, 16 streams | 1,789 @ 0.387, 482 | 2,455, 630 |

Cold prefill inside the needle gate: 131K tokens in 32.9 s and 220K in 69.0 s. No matched daily number exists for this instrument (the daily's gate serves those contexts from the tier); the like-for-like W4A16 prefill cost is R192's +10 to +31 % on llama-benchy prefill. tool-eval 69×4: 90 ± 1.4 (daily 91.2). Needle: 4/4 cold, 4/4 tier re-asks. The partial run on the FlashInfer kernel agreed with these where it got (dense +1.378 % on 618 documents, agentic +1.439 %).

Reading: the mixed recipe is closer to bf16 than RedHat on the agentic turns and on top-1 agreement, and farther on dense perplexity, at the W4A16 batch cost (8 streams −17 %, 16 streams −27 %, prefill slower) and only on a kernel the checkpoint was not shipped for. Not promoted.
