# NVFP4 KV cache on the 5090 — measured (2026-08-21)

**Status: experimental candidate, not the daily.** It boots, retrieval is clean, decode is at or above the fp8 daily, and the pool is +37% — but one MTP-specific throughput cliff is open, tool-eval is ~1.5 pts under fp8, and it runs without LMCache (the tier patches are fp8-page-specific). Numbers below are from one afternoon on one box; treat them as a first measurement, not a record.

The trigger was [drowzeys' DGX-Spark repo](https://github.com/drowzeys/keys-vLLm.0.27-Qwen3.8-27B-ADay777Ablit-NVFP4-A4Q-NVFP4-KV-4M-KV-token-pool-MTP3-Single-DGX-Spark): "back-porting the upstream FA2 NVFP4-KV path to GB10 takes the KV pool to >4M tokens". 4M is a 121 GB box; on 32 GB the same path gives the numbers below.

## What the stack is

Three pieces, all in [`patches-nvfp4kv/`](../patches-nvfp4kv/) (README there has the mechanism table, build and gauntlet):

1. **FlashInfer's FA2 NVFP4-KV attention for sm120/121 (head_dim 256)** — merged, already inside the nightly's FlashInfer (0.6.16.post3 here; 0.6.17 on current main). Nothing to build.
2. **vLLM routing, [PR #49891](https://github.com/vllm-project/vllm/pull/49891) (ch2lab, open)** — on sm120 send `--kv-cache-dtype nvfp4` to the FA2 wrappers instead of trtllm-gen, HND layout, bf16 q/o, cudagraph prefill wrappers. Rebased by hand onto the daily digest (`0001`, plus `0001b` for a signature the merge dropped).
3. **A linear-V-scale store overlay (`0002`/`0002b`)** — vLLM's in-tree NVFP4 store kernel always writes V block scales in the SM100 trtllm-gen 4-token swizzle; the FA2 reader addresses them linearly. The overlay is the in-tree kernel with `swizzle_v_sf = major < 12`, AOT-built as a stable-ABI op and routed to from the FlashInfer backend. Credit: drowzeys' writer patch.

Serving: [`scripts/serve-nvfp4kv.sh`](../scripts/serve-nvfp4kv.sh) — the plain profile (no LMCache) with `--kv-cache-dtype nvfp4 --attention-backend FLASHINFER`, MTP `ns=4`, `--no-async-scheduling`, `--mamba-cache-mode align`, util 0.95, max-len 200K, on an experiment port. Gauntlet driver: [`scripts/nvfp4kv-gauntlet.sh`](../scripts/nvfp4kv-gauntlet.sh).

## Numbers (RTX 5090, Qwen3.8-27B saka W4A4, nightly `ac7509e2b`)

| | nvfp4 KV (this) | fp8 KV (plain nightly, same profile) |
|---|---|---|
| KV pool @util 0.95, max-len 200K | **293,181** (1.47× of 200K) | 214,084 (tier @0.95) / 207,042 (plain @0.98) |
| pool with MTP off | 415–444K | — |
| decode c1, pp8192 tg512 | **133.8 t/s** | 123.1 |
| decode c2 / c4 | 200.8 / 311 | 207 / 315 |
| deep pp30K c8 decode | **136.7** | 122.9 |
| prefill pp8192 / 32K / 100K | 12.7K / 9.16K / 4.93K | 13.0K / 9.69K / 5.05K |
| **c8 × pp8192 decode** | **159–162** ← the cliff | 321.7 |
| tool-eval 69×2 @T0.6 medium | **89.5 ± 0.7** | 91 (plain) / 92.5 ± 0.7 (tier) |

**Why only +37% and not ×1.7:** on this 3:1 GDN hybrid only the 16 attention layers' KV shrinks; the GDN state and activations don't. And the MTP drafter costs 120–150K tokens of nvfp4 pool on its own (415–444K with `ns=0`).

**Correctness — all pass, MTP on and off:** depth needles 9K→100K cold+warm 10/10 + 10/10; the concurrent "sean gate" (4 × 20K loaders, 32/48/64K × 5, cold+warm) 15/15 + 15/15; 8×24K killer burst 8/8; 8×4×2048² vision burst 8/8; json_schema structured output 4/4. The July deep-decode penalty of the pre-merge FA2 path (−8..−23% at depth) is gone.

**The open cliff:** with MTP on and ≳50–57K prompt tokens in flight across ≥4 requests (c8@8192, c7@8192, c8@8000, c4@16384), aggregate decode halves and TTFT serializes (6–15 s ± 8 s); ≤49K in flight is fine (c8@6144 = 296 t/s, c6@8192 = 288, c4@12288 = 208), and **with MTP off the same shape is fine (c8@8192 = 344.6)**. fp8 KV has no such cliff. Not a 2^16 boundary. If your workload is many concurrent long prefills with MTP, this stack is slower today; single-stream and ≤4-way deep work is faster than fp8.

## The swizzle question — measured, not argued

The negative control (`VLLM_SM12X_NVFP4_LINEAR_VSF=0`, in-tree swizzled writer) **passed every behavioural probe** — needles, sean gate, and tool-eval 89.5 ± 2.1. Behavioural probes are blind to this bug. [`overlay/diag_vsf_layout.py`](../patches-nvfp4kv/overlay/diag_vsf_layout.py) pushes the same random K/V through both writers and runs the FA2 paged prefill against an fp32 reference: V-scale bytes differ in 30,091/32,768 positions (K scales and V data identical), and the FA2 output error is **2.7× worse with the in-tree writer on flat-magnitude V, 10× worse with group-structured V** (rel-L2 0.335–1.30 vs 0.124 for the overlay, identical at block 16 and 64). The in-tree writer is wrong for the FA2 reader on sm120; the model merely tolerates a 0.3–0.5 relative error on a quarter of its layers. Keep the overlay; the fix belongs upstream alongside #49891.

## Not done / next

- The c8-long-prefill cliff (engine-side metrics, then the drafter's prefill path or the spec-capped `max_num_scheduled_tokens=4096`).
- LMCache on nvfp4 pages (the tier patches `0001`/`0002` regroup fp8 pages; nvfp4 needs its own).
- `WITH_0003=1` (PR #49891's MTP drafter full-cudagraph) — built as an option, unmeasured.
- A4Q (fp4 Q × fp4 K for QKᵀ, prefill-only, +8–10% TTFT on drowzeys' box) — not prepared.
- Rebase on nightly `ba07e4a48` (Mamba state-copy race fix #50729, GDN spec-count #53077, FlashInfer 0.6.17): image built, all patches apply, unmeasured.
