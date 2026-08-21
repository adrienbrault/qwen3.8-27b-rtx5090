# LMCache KV tiers — the profile, and what removing it changes

The daily runs LMCache 0.5.4rc4's MP connector with the [rc4 patch stack](../patches/rc4/README.md) on top of an NVFP4 KV cache. Launcher: [`../scripts/serve-tier-rc4.sh`](../scripts/serve-tier-rc4.sh).

## Current profile (2026-08-21)

| | |
|---|---|
| image | `vllm-qwen38:tiers-nvfp4kv` = `patches/rc4/Dockerfile.rc4` on nightly `ba07e4a48` + `patches-nvfp4kv/Dockerfile.nvfp4kv` on top |
| engine | W4A4 NVFP4 weights + `nvfp4` KV + FlashInfer FA2 + MTP `ns=4` + vision, V2 model runner |
| pool | util 0.93, max-len 200K → **309,090 tokens** |
| chunk / batched | **2864** (= the unified block with nvfp4 KV) / **5727** (= 2·chunk − 1) |
| L1 | 24 GiB pinned host RAM |
| L2 | 200 GiB `fs_native` NVMe, watermark LRU eviction (patches 0008/0009), survives restarts |
| sidecar VRAM | 796 MiB (`LMCACHE_MP_GPU_STAGING_BATCH_SIZE=1` + `CUDA_MODULE_LOADING=LAZY`) |
| correctness gates (R81) | needles 9K→100K cold + warm 10/10 + 10/10, warm 0.5–2 s; restart-proof 40K 6.3 → 2.5 s, 60K 11.6 → 3.3 s; 4 loaders × 32/48/64K × 5: 15/15 + 15/15 |
| quality | 69×2 tool-eval 92 ± 1.4 |

What changed for nvfp4 pages: nothing in LMCache. The rc4 group edits classify vLLM's fused rank-4 pages by exact byte accounting and move logical pages byte-opaquely, and vLLM reports `uint8` as the KV dtype for both fp8 and nvfp4, so the `[blocks, 2·heads, block, 144]` nvfp4 page is just another fused page (the pool-sizing metadata over-allocates 2048 vs 1152 bytes per token, harmless). The one requirement is the chunk: vLLM's unified hybrid block inflates from 1616 to 2864 tokens with 4-bit KV, and the LMCache chunk must equal it.

Tier capacities in tokens have not been re-measured for nvfp4 pages; the 2026-07 figures (245K DRAM, 2.13M NVMe at 98 KiB per serialized token) were for fp8 pages of 1616 tokens.

## What removing LMCache changes

The same engine without the connector is the plain profile ([`../scripts/serve-nvfp4kv.sh`](../scripts/serve-nvfp4kv.sh) with `EXTRA_ENV="-e VLLM_USE_V2_MODEL_RUNNER=1"`). Measured 2026-08-21 on the V2 runner:

| | tiers on (daily) | plain |
|---|---|---|
| GPU pool | 309,090 (util 0.93, mnbt 5727) | 300,000 (util 0.93, mnbt 4096); 338,636 at 0.95 with autotuner OOM-fallback |
| DRAM / NVMe tiers | 24 GiB / 200 GiB, restart-proof | none |
| after a restart | stored contexts come back in 2.5–3.3 s (40–60K) | everything cold |
| decode c1 / c4 / c8 (pp8192) | 141 / 339 / 353 | c8 355–363 (R80; c1/c4 not re-measured on this image) |
| prefill c1 at 8K | 12.8K | 12.7–12.8K |
| tool-eval 69×2 | 92 ± 1.4 | 89 ± 1.4 |
| patches | rc4 + nvfp4kv | nvfp4kv |
| host | 24 GiB pinned RAM, 200 GiB NVMe, a sidecar process | none |

Drop the tiers when prompts are mostly fresh, when you cannot carry the rc4 patches, or when the host lacks the pinned RAM. Keep them when agents share large prefixes or sessions are revisited across hours and restarts.

The agentic A/B that motivated the tiers (16 SWE-Bench-Verified tasks at 4 concurrent agents: 3.4× wall-clock with tiers on, external prefix hit 88.8%) and the concurrency sweep (c4 is the knee; past it streams evict each other's prefixes out of L1) were measured in 2026-07 on the previous model and are in [../bench/RESULTS.md](../bench/RESULTS.md#archive--qwen36-era-2026-07). The mechanism has not changed: an agent step resends its whole transcript, so nearly every request is a long prefix revisit.

## Why a tier that looks fine can be wrong

Four rounds in 2026-07 ended in a confident, wrong "validated":

1. Stores failed `Unsupported EngineKVFormat: 10` in the sidecar while serving continued; every measured "tier revisit" was vLLM's own prefix cache.
2. A hand-written transfer kernel made stores run; a 60K needle vanished after reload and tool-eval fell 88 → 47. The kernel was patched under wrong metadata.
3. Transfers correct, quality 9–12 points below control: a vLLM scheduler bug that only exists with connector + MTP + hybrid left one unfilled attention block at every local hit.
4. Parity reached, and a deterministic 3-turn repro still produced an empty turn: LMCache exported vLLM's null Mamba block under a valid hash at the prefill boundary.

Coherent output and rising hit counters are compatible with a completely broken cache. The discriminating test is a needle planted in a long context and retrieved after a restart, and it is the first gate in every audition since. The same lesson applied to the nvfp4 store overlay in 2026-08: needles and tool-eval passed with the wrong writer; only a numeric diagnostic caught it.

The investigation as it unfolded is archived in [HISTORY.md](HISTORY.md#archive--the-2026-07-lmcache-investigation-vllm-024--lmcache-051--the-023-base).
