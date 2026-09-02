# Design notes

Why the weights, why the model, where the VRAM goes, and why the second card helps the way it does. Numbers are measured on this hardware; protocols in [../bench/RESULTS.md](../bench/RESULTS.md).

## Why W4A4 weights

Prefill is a compute-bound GEMM, so its speed is set by which tensor-core path the quantization format dispatches. That is decided by the activation format, not the weight format:

| format | tensor-core path | relative GEMM rate | prefill measured here at 8K |
|---|---|---|---|
| W4A4 (NVFP4, served) | native FP4 (Blackwell) | ~4× bf16 | 11.9K to 13.5K t/s |
| W8A8 (fp8) | fp8 | ~2× bf16 | attention layers of NVIDIA's export only |
| W4A16 (AutoRound, GPTQ, AWQ) | bf16 with inline dequant | ~1× bf16 | ~4.0K t/s |

Weight-only quantization saves bandwidth and decodes fast, but prefills no faster than bf16. W4A4 quantizes activations on the fly and runs the GEMM on the FP4 units. The activation-quantization quality cost was bounded at about one tool-eval point by a checkpoint A/B in 2026-07 (W4A4 MLPs with fp8 attention versus all-W4A4), and the bf16-anchored fidelity ladder of 2026-09 put the best W4A4 checkpoints within +0.37% perplexity of bf16.

## Why this model

Qwen3.8-27B fits long-context agents on a 32 GB card because of its layout:

1. A hybrid of 48 GDN linear-attention layers and 16 full-attention layers. Linear layers pay a fixed per-sequence state instead of per-token KV, so only a quarter of the layers grow with context.
2. Calibrated W4A4 exports exist from several quantizers; the recipe matters more than the bit width (see the checkpoint section of the [README](../README.md#the-checkpoint)).
3. The attention stack tolerates low-precision KV: fp8 KV costs +0.13 percentage points of perplexity and passes every concurrency gate; NVFP4 KV costs +0.76 and works on one card with the store overlay.
4. An MTP draft head ships in the checkpoint (0.8 GiB), and DFlash2 drafters exist for it.
5. A compact vision tower, and a clean compressed-tensors export that vLLM detects without a `--quantization` flag.

## Where the VRAM goes

One card, measured from the boot log (`gpu_worker` prints the breakdown) on the 2026-07 W4A4 export; the Qwen3.8 exports have the same tensor layout and size.

| slice | GiB | notes |
|---|---:|---|
| weights | 19.5 | language model 12.8 (64 layers, NVFP4 W4A4 plus block scales), embeddings 2.4 and lm_head 2.4 (bf16 in that export), vision tower 0.9, MTP drafter 0.8 |
| KV pool, nvfp4 KV | ≈ 8.1 | 309,090 tokens at util 0.93 in 2026-08; 381,300 at 0.955 on the v0.28 stack |
| activation reserve, CUDA graphs, autotune workspace | remainder | sized by profiling at the prefill chunk; the FlashInfer autotuner allocates more at serve time on the first new batch shape |

Why NVFP4 KV gives +48% pool over fp8 and not ×1.8: only the 16 attention layers' KV shrinks. GDN state and activations do not. With MTP off the same engine reads 415K to 444K tokens; the drafter's lookahead costs 120K to 150K tokens of pool.

## Why the second card helps the way it does

Tensor parallelism across two RTX 5090s splits the weights and the KV pool. Two consequences follow, and they explain the two-card table in the [README](../README.md#the-three-configurations):

- Speculative decoding with low acceptance (DFlash2, 0.12 to 0.30 accepted per draft token depending on content) is bound by weight bandwidth per step. The second card doubles that, so single-stream decode rises 37% to 71% and stays flat from the surface to 100K context.
- Speculative decoding with high acceptance (MTP, about 0.37) amortizes the weight reads over more tokens per step. The second card then buys KV space and admission headroom instead: a 1.5M-token pool and the best aggregate throughput at 16 streams.

Decode allreduces go through vLLM's custom allreduce over PCIe peer-to-peer. NCCL transport settings measured neutral for decode. The memory-clock offset delivers its full 15% bandwidth gain in silicon on both cards, but TP=2 decode gains only about 4% from it, because at TP=2 decode is co-bound by allreduce latency and drafter compute.

## What a cache hit is worth

60K context on the 2026-08 one-card stack: in-pool revisit about 1 s, DRAM tier 1 to 2 s, NVMe tier after a restart 3.3 s, full re-prefill 11.6 s. On the two-card stack a 32K warm revisit from the disk tier takes 0.45 s against 7.5 s cold. An agent step resends its whole transcript, so nearly every agent request is a long prefix revisit.
