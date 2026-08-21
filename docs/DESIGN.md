# Design notes

Why the weights, where the VRAM goes, what the host contributes. Numbers are measured on this box; protocols in [../bench/RESULTS.md](../bench/RESULTS.md).

## Why W4A4 weights

Prefill is a compute-bound GEMM, so its speed is set by which tensor-core path the quant format dispatches, and that is decided by the activation format:

| format | tensor-core path | relative GEMM rate | prefill measured here |
|---|---|---|---|
| W4A4 (NVFP4, the daily) | native FP4 (Blackwell) | ~4× bf16 | 12.8–13.5K t/s at 8K |
| W8A8 (fp8) | fp8 | ~2× bf16 | (attention layers of NVIDIA's export) |
| W4A16 (AutoRound, GPTQ, AWQ) | bf16 + inline dequant | ~1× bf16 | ~4.0K t/s at 8K (2026-07 daily) |

Weight-only quants save bandwidth and decode fast, but prefill no faster than bf16. W4A4 quantizes activations on the fly and runs the GEMM on the FP4 units. The activation-quant quality cost was bounded at about 1 tool-eval point in 2026-07 by a chimera checkpoint A/B (W4A4 MLPs + fp8 attention); the calibrated export covers it.

The model shape that fits a 32 GB Blackwell card for long-context agents, which this checkpoint is:

1. A hybrid layout (48 GDN linear-attention + 16 full-attention layers): linear layers pay a fixed per-sequence state instead of per-token KV.
2. W4A4 on the MLPs with real calibration.
3. An attention stack and KV cache that tolerate low-precision KV: fp8 passed every concurrency gate; nvfp4 passes them on the V2 runner with the FA2 path and the store overlay.
4. An MTP draft head in the checkpoint (0.8 GiB, about 2× single-stream decode) and a compact vision tower.
5. A clean compressed-tensors export that vLLM auto-detects (no `--quantization` flag) with an unquantized `lm_head`.

## Where the VRAM goes

Measured from the boot log (`gpu_worker` prints the breakdown) on the 2026-07-19 W4A4 export; the 3.8 export has the same tensor layout and size, and the KV rows are from the 2026-08-21 daily.

| slice | GiB | notes |
|---|---:|---|
| weights | 19.5 | language model 12.8 (64 hybrid layers, NVFP4 W4A4 + block scales), embeddings 2.4 and lm_head 2.4 (bf16), vision tower 0.9, MTP drafter 0.8 |
| KV pool, nvfp4 KV | ≈ 8.1 | 309,090 tokens at util 0.93 (fp8 KV at 0.95: 208,450 tokens in ≈ 7.9) |
| LMCache sidecar | 0.78 | 796 MiB of CUDA context, invisible to `--gpu-memory-utilization` |
| activation reserve, CUDA graphs, autotune workspace | remainder | sized by profiling at `mnbt` 5727; the FlashInfer autotuner allocates at serve time on the first new shape (gotcha 4) |

Why nvfp4 KV gives +48% and not ×1.8: only the 16 attention layers' KV shrinks; GDN state and activations do not. With MTP off the same engine reads 415–444K: the drafter's lookahead costs 120–150K tokens of pool.

What a cache hit is worth (60K context, 2026-08-21 on the nvfp4 tier daily): in-pool revisit ≈ 1 s; DRAM tier ≈ 1–2 s; NVMe tier after a restart 3.3 s; full re-prefill 11.6 s. Tier capacities in tokens have not been re-measured for nvfp4 pages; a serialized page is 1152 bytes per token for the attention layers (8 heads × 144 bytes) plus the GDN state page.

Effective deep concurrency is pool-bound: four 50K-token agent sessions fill the hot pool; the fifth demotes to a tier and comes back in seconds.

## Host

RTX 5090 32 GB with a memory-only overclock (+4500 MHz, 16,051 MHz effective vs 14,001 stock) at the 600 W power limit, core stock, persisted by a systemd unit; Ryzen 9 5900X; 64 GB RAM (24 GiB pinned by the L1 tier); NVMe with 200 GiB reserved for L2; Ubuntu 24.04; swap off. Commands in [CONFIG.md](CONFIG.md#host).
