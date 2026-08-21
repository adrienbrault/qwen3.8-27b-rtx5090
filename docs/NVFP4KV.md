# NVFP4 KV cache on the 5090 — measured (2026-08-21)

**Status: the daily since 2026-08-21 evening, with the LMCache tiers (below).** It boots, retrieval is clean, decode is at or above the fp8 daily, and the pool is +37% — but one MTP-specific throughput cliff is open, tool-eval is ~1.5 pts under fp8, and it runs without LMCache (the tier patches are fp8-page-specific). Numbers below are from one afternoon on one box; treat them as a first measurement, not a record.

The starting point was [drowzeys' DGX-Spark repo](https://github.com/drowzeys/keys-vLLm.0.27-Qwen3.8-27B-ADay777Ablit-NVFP4-A4Q-NVFP4-KV-4M-KV-token-pool-MTP3-Single-DGX-Spark): "back-porting the upstream FA2 NVFP4-KV path to GB10 takes the KV pool to >4M tokens". 4M is a 121 GB box; on 32 GB the same path gives the numbers below.

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

**Mechanism (engine `/metrics` around one c8×8K run): 23 preemptions for 8 requests, 35 s of summed queue time, KV usage near 0% at the end** — the scheduler evicts and re-admits at well under a quarter of the pool, so it's allocator/block-granularity pressure, not compute. The hybrid's `align` mode unifies the block across attention and GDN groups: with nvfp4 attention bytes at 0.56× fp8, the shared block inflates from 1616 to ≈2870 tokens (drowzeys saw 2848 on GB10), per-block GDN state snapshots get scarcer per request, and MTP's lookahead slots tip it over. First suspects: nightly `ba07e4a48`'s [#52216](https://github.com/vllm-project/vllm/pull/52216) (`prefix_cache_retention_interval` default 0 for SSM/hybrid models — exactly this pressure; image built, unmeasured), then PR #49891's own `0003` MTP-drafter cudagraph piece (excluded from this build).

## The swizzle question — measured, not argued

The negative control (`VLLM_SM12X_NVFP4_LINEAR_VSF=0`, in-tree swizzled writer) **passed every behavioural probe** — needles, sean gate, and tool-eval 89.5 ± 2.1. Behavioural probes are blind to this bug. [`overlay/diag_vsf_layout.py`](../patches-nvfp4kv/overlay/diag_vsf_layout.py) pushes the same random K/V through both writers and runs the FA2 paged prefill against an fp32 reference: V-scale bytes differ in 30,091/32,768 positions (K scales and V data identical), and the FA2 output error is **2.7× worse with the in-tree writer on flat-magnitude V, 10× worse with group-structured V** (rel-L2 0.335–1.30 vs 0.124 for the overlay, identical at block 16 and 64). The in-tree writer is wrong for the FA2 reader on sm120; the model merely tolerates a 0.3–0.5 relative error on a quarter of its layers. Keep the overlay; the fix belongs upstream alongside #49891.

## Update 2026-08-21 (later): on the V2 runner the cliff is gone

Re-run on the promoted daily's runner and nightly (`ba07e4a48`, `VLLM_USE_V2_MODEL_RUNNER=1`, image rebuilt with the same patches): **c8 × pp8192 = 355 t/s** (was 159), c8 × pp4096 431–447, the whole c4/c6/c8 bracket healthy; needles 10/10 + 10/10 to 100K for both `nvfp4` and `nvfp4_4over6`; pool **338,636 @0.95** (the FlashInfer autotuner OOM-falls-back at that utilization on V2 — use 0.93 → 300,000); tool-eval **89 ± 1.4** (`nvfp4_4over6`: 88.5 ± 0.7) vs the fp8 tier daily's 90.8 ± 0.5. So nvfp4 KV is now a clean plain-profile option — +44–63% pool for ~2 tool-eval points and no LMCache tiers. The remaining item to make it the daily is LMCache page regrouping for nvfp4 pages.

## Update 2026-08-21 (evening): LMCache tiers on nvfp4 pages — it works, and needed no page code

The expected blocker ("the rc4 page patches are fp8-specific") was not one: LMCache's rank-4 group edit classifies pages by exact byte accounting and moves logical pages byte-opaquely, so the nvfp4 page `[blocks, 2·heads, block, 144]` is just another fused page. The one real requirement is the hybrid's **unified block**: vLLM aligns the attention block to the Mamba page — 1616 tokens with fp8 KV, **2864 with nvfp4** — and the LMCache chunk must equal it (the launcher now derives `BLK` from the KV dtype; `mnbt = 2·BLK − 1 = 5727`). Image: the nvfp4kv Dockerfile built `FROM` the tier image.

Measured (V2 runner, util 0.93, 200K, tiers ON, fresh L2): **pool 309,090** (fp8 tier daily: 208,450, **+48%**); needles 9K→100K cold+warm 10/10 + 10/10 with real retrieves (100K: 27 s → 1.2–2 s); **restart-proof** revisit after `docker restart` 40K 6.3 → 2.5 s, 60K 11.6 → 3.3 s, correct; sean gate under 4 loaders 15/15 + 15/15; decode c1 141 / c4 339 / c8 353 / deep-30K 142 t/s (fp8 tier: 152 / 360 / 367 / 143); prefill 12.8K @8K (the larger mnbt helps); **tool-eval 69×2 92 ± 1.4**. Promoted the same evening; 69×4, vision burst, SO probe and a soak on the promoted engine are pending.

## Not done / next

- ~~The c8-long-prefill cliff~~ — gone on `ba07e4a48` + V2 (above).
- ~~LMCache on nvfp4 pages~~ — works as-is with chunk = 2864 (above).
- `WITH_0003=1` (PR #49891's MTP drafter full-cudagraph) — built as an option, unmeasured.
- A4Q (fp4 Q × fp4 K for QKᵀ, prefill-only, +8–10% TTFT on drowzeys' box) — not prepared.
- Rebase on nightly `ba07e4a48` (Mamba state-copy race fix #50729, GDN spec-count #53077, FlashInfer 0.6.17): image built, all patches apply, unmeasured.
