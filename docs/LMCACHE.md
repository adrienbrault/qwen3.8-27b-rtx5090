# MTP + LMCache tiered KV caching — every flag, and why

The launch command lives in [`../scripts/serve-lmcache.sh`](../scripts/serve-lmcache.sh). This explains it. For the *why-at-all* and the benchmark numbers, see the [README section](../README.md#update--2026-07-14-mtp--lmcache-tiered-kv-caching-recommended-for-multi-agent-coding).

**When to run this instead of the [k8v4 daily](CONFIG.md#recommended-clean-tq-image--turboquant_k8v4-our-daily):** multi-agent coding, where several agents share large near-identical prefixes and cache retention beats single-stream latency. The profile runs **no vision** (`image:0`) — that's the one capability it gives up vs the daily. It uses **fp8 KV**, not the daily's k8v4: LMCache's persistence tier only round-trips faithfully with fp8 ([why](#lmcache--k8v4-composes-but-the-persisted-tier-is-lossy--not-shipped)).

Every ingredient below was earned by a failure. Removing any one of them reintroduces a specific, documented break.

## Architecture

vLLM keeps its KV cache on the GPU and evicts on pool pressure; a revisit re-prefills from scratch (**5.8s** on a 40K-token session). [LMCache](https://github.com/LMCache/LMCache) adds tiers below the GPU:

```
on-GPU prefix cache   ≈ instant     (vLLM's own, evicts under pressure)
  └─ L1: 24 GB pinned host RAM   ~0.5s hit
       └─ L2: 150 GB SSD         ~2.7s hit   (~2M tokens; survives restarts)
            └─ cold re-prefill    5.8s
```

Qwen3.6 is a **hybrid** (GDN/linear-attention + full-attention), so its mamba state must be stored as an opaque page. Only LMCache's **`LMCacheMPConnector`** does that ([LMCache PR #3613](https://github.com/LMCache/LMCache/pull/3613), in 0.5.1), and it needs an out-of-process `lmcache server` sidecar (ZMQ :5555) that owns the L1/L2 tiers.

## The image

```bash
IMAGE=lmcache-vllm:fixed
```

`lmcache/vllm-openai:latest-cu129` (vLLM **0.24** + lmcache **0.5.1**) with one fix: it's CUDA-13-linked but ships no `libcudart.so.13`, so `csrc` load fails. Multi-stage `COPY` `libcudart.so.13*` from `nvidia/cuda:13.0.1-runtime-ubuntu24.04`, then `ldconfig`. (`pip install nvidia-cuda-runtime-cu13` does not work — PEP 668, then no wheel.)

**Do not use the nightly pairing** (`latest-nightly-cu129`, lmcache 0.5.2-dev): it rejects this model's KV layout with `Unsupported EngineKVFormat: 10` on every store — *unless* you rebuild lmcache with [`../patches/lmcache-0.5.1-format10-NL_X_NB_NH_BS_TWO_HS.patch`](../patches/lmcache-0.5.1-format10-NL_X_NB_NH_BS_TWO_HS.patch), which recovers the nightly path (composed decode c1 122 / c8 458, slightly ahead of 0.24's 118/450). See [the patch section](../README.md#the-format-10-kernel-patch-lmcache-cant-store-this-model).

## Container flags

```bash
--entrypoint bash        # the image entrypoint is `vllm serve` — it would swallow our `bash -c`
--ipc=host               # CUDA-IPC across processes; without it, hangs at "Creating transfer context"
--memory 52g             # cgroup cap. The 24 GB L1 is PINNED host RAM — drop_caches before launch
-e MAX_JOBS=4 -e FLASHINFER_NUM_COMPILE_JOBS=4      # cap the sm120 fp4-GEMM JIT (see gotcha 1)
-v .../cache/flashinfer:/root/.cache/flashinfer     # MANDATORY persistent cache for that JIT build
```

**Never** add `-e PYTORCH_ALLOC_CONF=expandable_segments:True`. cuMem/VMM memory is not CUDA-IPC-exportable ([pytorch #165685](https://github.com/pytorch/pytorch/issues/165685), [vllm #29544](https://github.com/vllm-project/vllm/issues/29544)); the sidecar crashes importing the KV handles and never acks, and vLLM's `register_kv_caches` silently times out at 300s. This is the single most expensive gotcha here — it read as "version skew" for days.

## The `lmcache server` sidecar

```bash
lmcache server --host 0.0.0.0 --port 5555 --chunk-size 1600 \
  --l1-size-gb 24 --l1-init-size-gb 2 --eviction-policy LRU \
  --worker-reap-timeout-seconds 0 \
  --l2-adapter '{"type":"fs_native","base_path":"/l2","max_capacity_gb":150,"num_workers":4}'
```

- **`--chunk-size 1600`** — must equal vLLM's **unified block size**: **1600** with MTP `ns=3`, **1568** without (discovered, not documented — it is not 16). Mismatch → "chunk size must be a multiple of vLLM block size".
- **`--l1-size-gb 24`** — the pinned-RAM tier. **It must exceed the hot working set / 0.8**, or LMCache's lookup breaks at the first missing chunk and LRU evicts the oldest session's *head* chunks first → head-miss → full re-store → evicts the next head → permanent thrash, **0% hits**. Partial caching does not degrade gracefully; undersize this and the cache is inert, not merely smaller.
- **`--worker-reap-timeout-seconds 0`** — disables the worker-registration reaper. Default 120s + a lazily-started heartbeat means a long idle/blocked span gets reaped, after which the cache **cannot recover** (permanent zombie: `found_count=0`, stores silently dropped at the worker's health gate).
- **`--l2-adapter fs_native … 150 GB`** — the SSD tier on a host dir mounted at `/l2`. ~2M tokens; survives container restarts. A 10×40K working set (29 GB) spills here with zero thrash.

## vLLM flags

```bash
--kv-cache-dtype fp8_e4m3
```
fp8 KV. (4-bit TurboQuant KV is not composed here — this profile prioritises retention + concurrency, where fp8 already wins.)

```bash
--kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}'
```
Routes KV through the MP connector (store + load). The **only** connector that handles this hybrid's mamba state. *Belt-and-braces:* vLLM 0.24 also accepts `kv_load_failure_policy: "recompute"` here — the default `"fail"` turns any load failure into a request 500.

```bash
--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
```
MTP, composed with the cache. The composition the upstream trackers call unsupported — it works here; the "MTP+LMCache crashes" reports were a VRAM burst-OOM (gotcha below), not the scheduler wall. Keep **default cudagraph mode**: forcing `FULL_DECODE_ONLY` fails to cover MTP's verify-step shapes on 0.24 and collapses decode to **c1 46 / c8 179** (vs 118/450).

```bash
--gpu-memory-utilization 0.93
```
**Not 0.94.** The `lmcache server` process holds ~1.4 GB of VRAM (CUDA context + IPC mappings) that `--gpu-memory-utilization` does not account for. At 0.94, a burst of concurrent cold prefills OOMs (74 MB free at crash) and kills the engine — this was the entire "MTP crashes with LMCache" myth.

```bash
--max-model-len 120000
--max-num-batched-tokens 3199        # = 2*chunk - 1
```
`120000` = the opencode client's context. `--max-num-batched-tokens` **must** be `2·chunk−1` (3199 with chunk 1600; 3135 with 1568) — LMCache's MP connector requires batched-tokens ∈ [chunk, 2·chunk). This batched-token ceiling is why prefill runs ≈−10% vs the daily's 8192 path at depth.

```bash
--limit-mm-per-prompt '{"image":0,"video":0}'
```
**Vision off.** The one capability gap vs the daily. A vision-on variant is untested; `image:0` also buys a thriftier KV pool (163K tokens no-MTP, 124K composed).

```bash
--max-num-seqs 8 --mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill
--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml
--override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20}'
```
Identical to the daily — see [CONFIG.md](CONFIG.md) for each. `--mamba-cache-mode align` is what lets prefix caching work on the Mamba layers; `qwen3_xml` is the correct tool parser (`hermes` drops the calls).

## Known rough edges

- **Two stochastic flakes** shared by every composed run: an instant-EOS at depth (`comp=1` empty — the documented temp-0.6 quirk) and a ~1-in-3 flake at pool capacity. Noisier than the clean no-MTP LMCache run; a flake-rate probe is pending.
- **`engine_driven` transfer mode** serves but scores 0 hits on this hybrid — stick with the default `lmcache_driven`.
- The **no-MTP** LMCache config (chunk 1568, no `--speculative-config`, util 0.90+) is the conservative fallback if MTP composition ever regresses: c8 517 (beats the daily's 488), c1 69 (the 0.24 image tax + no MTP).

## LMCache + k8v4: composes, but the persisted tier is lossy — not shipped

The daily's `turboquant_k8v4` KV composes with LMCache in the lab, but its **persisted (L2 SSD) tier is not bit-faithful** — so this profile stays **fp8-only**.

- **It builds and runs.** The clean TQ image already ships lmcache 0.5.1; graft the format-10 `c_ops.so` from the [fmt10 build](../patches/lmcache-0.5.1-format10-NL_X_NB_NH_BS_TWO_HS.patch) (identical ABI, single file, no recompile). It launches, composes with MTP, stores land (0 format-10 errors), and the L2 SSD tier fills.
- **But the L2 reload corrupts long-context retrieval.** After a container restart LMCache reloads 16–21K tokens in ~26 ms and the output stays fully coherent — yet planted long-context needles **vanish** (measured **7/7 miss** across two needles; a fresh prefill retrieves every time; the sidecar log confirms LMCache served the reload). Root cause: the format-10 transfer kernel copies bytes for the standard `[NB, NH, BS, 2·HS=512]` fp8/bf16 layout, but k8v4 packs `[…, 262]` (8-bit K + 4-bit V + scales) — the stride mismatch corrupts the L2 serialization round-trip. Coherent-but-lossy is exactly the failure that kills long-context coding.
- **`engine_driven` doesn't rescue it.** The `engine_driven` transfer mode ([LMCache PR #4073](https://github.com/LMCache/LMCache/pull/4073)) reclaims the ~900 MB sidecar VRAM (1,370 → 498 MiB), but its SHM-registration handshake is unstable grafted onto the older-nightly TQ base (300 s `register_kv_caches` timeout), and even the prior working run was parked as unstable (≥30K-prefill OOM) with fidelity unverified.
- **Conclusion.** LMCache's persistence tier only round-trips faithfully with **fp8 KV**. k8v4 keeps vLLM's in-pool `--enable-prefix-caching` (fast in-pool reuse) but **no tiered persistence**. A faithful k8v4 tier would need a new lmcache `KVFormatSpec` + transfer kernel for the 262-wide packed layout (or forcing opaque BINARY blocks) — not worth it single-user.
