# Design notes — why the config is built this way

Mechanism-level reference for the choices the [README](../README.md) states as conclusions: why W4A4 weights, where every GiB of VRAM goes, and what the host contributes. Numbers here are measured on this box (see [../bench/RESULTS.md](../bench/RESULTS.md) for protocols).

## Why these weights — and what actually governs prefill speed

Prefill is a compute-bound GEMM problem: thousands of tokens multiplied through the weights at once. So prefill speed is set by **which tensor-core path the quant format lets vLLM dispatch** — and that is decided by the *activation* format, not the weight bits:

| format | tensor-core path | relative GEMM rate | prefill measured here |
|---|---|---|---|
| **W4A4** (NVFP4, this daily) | native FP4 (Blackwell) | ~4× bf16 | **13.5K t/s** @8K |
| W8A8 (fp8) | fp8 | ~2× bf16 | (attention layers of the NVIDIA export) |
| **W4A16** (AutoRound, GPTQ, AWQ…) | bf16 + inline Marlin dequant | 1× bf16 − overhead | ~4.0K t/s @8K |

W4A16 keeps activations in 16-bit, so the tensor cores run plain bf16 GEMM; the 4-bit weights only save memory *bandwidth* — which is why weight-only quants decode fast but prefill no faster than bf16. W4A4 quantizes activations to FP4 on the fly and runs the whole GEMM on Blackwell's FP4 units. Decode barely differs between formats (decode is bandwidth-bound); **prefill is where the format war is won**, and for agents prefill is the latency you feel.

The quality side: W4A4's extra activation-quant error measured **≈ 1 point** of tool-eval on this checkpoint — we bounded it directly by building a [chimera checkpoint](HISTORY.md) (this model's W4A4 MLPs + NVIDIA's fp8 attention) and scoring all three variants on the full 69×2 suite. natfii's calibration eats that point: it scores at parity with the best W4A16 daily we ever ran. That killed the last reason not to switch.

**The ideal model shape for this box** (32 GB Blackwell + long-context agents) — natfii is essentially it:

1. **Hybrid attention layout** (48 GDN linear-attention + 16 full-attention layers here) — linear layers pay a fixed per-sequence state instead of per-token KV, which is what makes a 200K context affordable at all on 32 GB.
2. **W4A4 NVFP4 on the MLPs at minimum** (~70% of prefill FLOPs), **with real calibration** — quantized *activations*, or the FP4 units idle.
3. **fp8-tolerant attention + fp8 KV cache** — the only KV format that survived concurrency on this hybrid; every custom 4-bit-KV kernel corrupted or crashed.
4. **MTP draft head included** (0.79 GiB → ~2× deep single-stream decode) and a compact vision tower (0.86 GiB).
5. **Clean ModelOpt/compressed-tensors export** that vLLM auto-detects — no `--quantization` flag games, no baked tokenizer quirks (see [GOTCHAS.md](GOTCHAS.md) #9; this checkpoint shipped one).

## Where the 31.35 GiB goes — memory budget

All numbers below are measured — the boot log (`gpu_worker` prints the breakdown at every launch) plus the checkpoint's safetensors headers for the weight split. Weights are identical in both profiles; the KV/util/sidecar rows are the daily's (tiers on, util 0.95):

| slice | GiB | notes |
|---|---:|---|
| **Weights, total** | **19.53** | split ↓ (19.15 on disk + loader overhead) |
| · language model | 12.76 | 64 hybrid layers (48 GDN + 16 full-attention), NVFP4 W4A4 + per-block scales |
| · embeddings | 2.37 | bf16 — not quantized |
| · lm_head | 2.37 | bf16 — embed+head = 4.7 GiB, a **24% big-vocab tax** on the weight budget |
| · vision tower | 0.86 | bf16, always loaded (even text-only requests) |
| · MTP drafter | 0.79 | shipped less-quantized than the previous daily's (0.28) — the price of the ns=4 head that ~2×'s deep decode |
| **KV pool** | **7.98** | **= 214,084 tokens** @ ~39 KiB/tok (fp8 attention KV + GDN state pages, one unified pool) |
| **LMCache sidecar** | **0.78** | 796 MiB of CUDA context + kernel modules, **invisible to `--gpu-memory-utilization`** — this is what caps the tier profile at util 0.95. Was 1,412 MiB before the staging-batch diet |
| **Peak-activation reserve** | 1.89 | sized by profiling at `mnbt` 3231 |
| **CUDA graphs + non-torch** | 0.40 | |
| **util 0.95 budget** | **29.78** | of 31.35 usable; killer-shape floor **531 MiB** free — healthier than the plain profile's 130–190 MiB at 0.98 |

Honest trade vs the previous daily: natfii's weights are **1.8 GiB heavier** in VRAM (fatter MTP head + FP4 scale tensors), which is why the hot pool is 214K where the old W4A16 daily reached 270K. We took it: prefill 3.4×, deep-concurrent throughput 2.2×, equal quality — and the tiers put ~2.4M tokens back underneath. Capacity you reload in 2–7 s is worth more than capacity you wait on.

What each token costs and what a cache hit is worth (60K-token context, measured):

| tier | capacity | revisit cost | survives restart |
|---|---|---|---|
| GPU pool (vLLM prefix cache) | 214,084 tok | **~1–2 s** (≈ decode time only) | no |
| host DRAM (LMCache L1, 24 GiB pinned) | ~245K tok @ ~98 KiB/tok serialized | **~2 s** | no |
| NVMe (LMCache L2, 200 GiB) | ~2.13M tok | **~4.4–7.5 s** | **yes** |
| miss → full re-prefill | — | **~11–13 s** (was ~23 s on the W4A16 daily — the FP4-GEMM dividend) | — |

**≈2.59M reusable tokens** total, of which ~2.13M persist across restarts.

Notes that save you from wrong conclusions:

- **util is the pool lever that matters** (+~8.4K tok per 0.01, measured across 0.92 → 0.98). `max-num-seqs` is *nearly* free but not exactly: `align` packs GDN state into the unified pool and consumes it per **active** request, so seqs 4 vs 8 measured identical — but seqs 8 → 16 costs **2,817 tokens (−1.3%)**, presumably per-slot bookkeeping crossing block granularity. Don't treat it as a capacity knob in either direction; do re-read the pool from the boot log if you change it, because a sweep run at a different `max-num-seqs` isn't strictly comparable to one that wasn't.
- The in-pool ~39 KiB/tok is an *effective average*: full-attention layers pay per-token fp8 KV; GDN layers pay a fixed per-sequence state that amortizes with depth. Serialized tiers pay ~98 KiB/tok because LMCache ships whole 1616-token unified blocks (attention KV + state page + metadata, padded to full block). **A tier token is ~2.5× the bytes of a pool token** — that ratio, not the tier's raw size, is what you budget against.
- Effective deep concurrency is **pool-bound, not `max-num-seqs`-bound**: 4 × ~50K-token agent sessions fill the hot pool; the fifth doesn't queue any more, though — it demotes to a tier and comes back in seconds. That's the whole point of the tiers.
- **The tiers required six local patches, none upstream.** Two of them are on **vLLM itself** and only fire with a connector + MTP + a hybrid model: a connector-path prefix-hit reduction that mixed an EAGLE-adjusted attention hit with an unadjusted Mamba hit (leaving one allocated-but-never-filled attention block at *every* local hit — ~10 eval points), and a store-boundary fix that stopped LMCache exporting vLLM's *null Mamba block* under a valid hash. Two more make LMCache's fp8 page regrouping correct at all; two are operational (sidecar VRAM, L2 cap enforcement). Table and failure modes: [../patches/lmcache/README.md](../patches/lmcache/README.md); the investigation: [LMCACHE.md](LMCACHE.md). Whatever you run: **needle-test across a restart before trusting any external KV tier on a hybrid model** — hit counters and coherent output do not prove fidelity. They looked perfect through four rounds of being wrong.

## Host notes

**GPU overclock (affects the benchmark numbers).** This box runs a **memory-only overclock: +4500 MHz VRAM offset** (16,051 MHz effective, vs ~14,001 stock) at the 600 W power limit, persisted across reboots via a systemd unit. Core clock is left at stock (0 offset) — deliberately, because decode throughput on this model is **memory-bandwidth-bound**, so the memory OC is the only knob that moves the needle and a core OC would add heat for nothing. **All throughput numbers in [../bench/RESULTS.md](../bench/RESULTS.md) are with this OC.** A stock 5090 will decode somewhat slower; prefill (compute-bound) is barely affected. Reproduce with:

```bash
sudo nvidia-smi -pm 1 && sudo nvidia-smi -pl 600 && sudo nvidia-smi -rgc
sudo python3 -c "import pynvml as N; N.nvmlInit(); h=N.nvmlDeviceGetHandleByIndex(0); \
  N.nvmlDeviceSetGpcClkVfOffset(h,0); N.nvmlDeviceSetMemClkVfOffset(h,4500)"
```

Validate the offset actually applied (a "running" service isn't proof): `nvidia-smi -q -d CLOCK`, or read `nvmlDeviceGetMemClkVfOffset`. +4500 is stable on this sample; your mileage varies — step up and watch for memory-ECC errors or artifacts.

**Disable swap.** Over-committing RAM (e.g. running a benchmark container next to vLLM) swap-thrashes the box into a hard hang instead of failing fast. `swapoff -a` turns a wedge into a clean OOM kill.
