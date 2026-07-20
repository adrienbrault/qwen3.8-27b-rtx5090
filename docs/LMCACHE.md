# MTP + LMCache tiered KV caching — the profile, and the four rounds of being wrong

**This is the daily.** The launch command is [`../scripts/serve.sh`](../scripts/serve.sh); the patches are [`../patches/lmcache/`](../patches/lmcache/README.md); the capacity/latency trade against running without it is in the [README](../README.md#what-removing-lmcache-changes).

## Current profile (2026-07-20)

| | |
|---|---|
| image | `vllm-qwen36:tiers` — base daily image + LMCache `main` @`e38ee415` + patches 0001/0002/0003/0005/0007/0008 |
| engine | natfii NVFP4 W4A4 + fp8 KV + FlashInfer + MTP `ns=4` + **vision on** (`image:4`) |
| pool | util **0.95**, `--max-model-len 200000` → **214,084 tokens** |
| chunk / batched | **1616** (= unified block size at `ns=4`) / **3231** (= 2·chunk−1) |
| L1 | 24 GiB pinned host RAM ≈ 245K tokens, ~2 s revisit |
| L2 | 200 GiB `fs_native` NVMe ≈ 2.13M tokens, 4.4–7.5 s revisit, **survives restarts** |
| sidecar VRAM | 796 MiB (`LMCACHE_MP_GPU_STAGING_BATCH_SIZE=1` + `CUDA_MODULE_LOADING=LAZY`) |
| quality | full 69×2 tool-eval **89** — parity with the ~89.8 no-connector baseline |
| soak | **858 cycles** (needle + `pp8192×c8` killer + vision per cycle): all green, free VRAM flat at 701 MiB, L2 oscillating 39–47 GB under its cap |

> **Caveat on the 200 GiB L2.** Cap enforcement was soaked at **60 GiB** — 858 cycles holding steady. 200 GiB is the same admission-control path with a bigger number, and the host has 834 GB free, but it has *not* been soaked at that size. Watch `du -sh` on the L2 directory for the first day. The failure mode this guards against filled a root filesystem once already.

Vision is **on** in this profile — the `image:0` restriction in the historical text below belonged to the old 0.24/0.5.1 build and no longer applies.

Every ingredient was earned by a failure. Removing any one reintroduces a specific, documented break — and most of those breaks are *silent*.

## Why this took four rounds

Worth reading before you trust your own tier setup, because each round ended in a confident, wrong "validated":

1. **"LMCache works!"** — it stored **nothing**. Every store failed `Unsupported EngineKVFormat: 10`, logged by the sidecar while serving continued. Every "tier revisit" measured was vLLM's own in-GPU prefix cache.
2. **"Fixed with a format-10 kernel!"** — stores ran, tier filled, output stayed fluent, and a 60K needle **vanished** after reload; tool-eval fell 88 → 47. We had patched the kernel under wrong *metadata*.
3. **"Fixed with the stride-aware regroup!"** — transfers now correct, but quality sat 9–12 points below control. Root cause was not in LMCache at all: a **vLLM scheduler** bug that only exists with connector + MTP + hybrid, leaving one unfilled attention block at every local hit.
4. **"Parity reached!"** — 88, and a deterministic 3-turn repro still produced an empty turn. LMCache was exporting vLLM's **null Mamba block** under a valid hash at the EAGLE-adjusted prefill boundary, poisoning later retrieves.

The pattern: **coherent output and rising hit counters are compatible with a completely broken cache.** The only test that discriminated was a needle planted in a long context and retrieved after a restart. Everything else we tried — hit rates, throughput, spot-checking answers, even token-exact single-turn comparisons — passed while the cache was wrong.

---

*The rest of this document is the investigation as it unfolded, against the earlier vLLM 0.24 / LMCache 0.5.1 profile. Historically accurate; superseded by the table above wherever the two conflict (notably: vision, util, chunk size, `ns`, and the image).*

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

**Do not use the nightly pairing** (`latest-nightly-cu129`, lmcache 0.5.2-dev releases through at least 2026-07-06): it rejects this model's KV layout with `Unsupported EngineKVFormat: 10` on **every store** — and because the error is logged by the sidecar while serving continues, the profile silently degrades to vLLM's in-GPU prefix cache only; every "tiered revisit" you measure is a mirage. Our first fix attempt — a hand-written format-10 transfer kernel ([patch, now **withdrawn**](../patches/README.md#lmcache-format-10-kernel-patch-separate-project)) — made stores *run* but restored **corrupted context** (needle vanishes after reload; tool-eval 88 → 47). Root cause (established later, and it is *not* what we first guessed): vLLM's fp8 attention backend registers each hybrid-aligned page as ~100 contiguous **16-token kernel pages** in a fused rank-4 tensor; LMCache's kernel-page→logical-page regrouping only matched the rank-5 split-K/V layout, so it misread the 16-vs-1616 slots/tokens ratio as *compression* and transferred **one 16-token page per logical block** — wrongly addressed, zero errors logged. The GDN state pages were stored and restored fine all along (which is why output stayed fluent while distant facts vanished). LMCache **`main`** (≥ `0.5.2.dev66`, PR #4128) has the native format kernels, but as of `e38ee415` the regrouping gap is still there for the fp8 fused layout — bf16 hybrid passes a cross-restart needle (its raw page is already scheduler-sized); fp8 hybrid does not. A stride-aware fix is in progress. Whatever you run: **needle-test across a restart before trusting any external KV tier on a hybrid model** — hit counters and coherent output do not prove fidelity.

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

`turboquant_k8v4` KV (the prior daily; the current daily's `turboquant_4bit_nc` packs tighter still) composes with LMCache in the lab, but its **persisted (L2 SSD) tier is not bit-faithful** — so this profile stays **fp8-only**.

- **It builds and runs.** The clean TQ image already ships lmcache 0.5.1; graft the format-10 `c_ops.so` from the [fmt10 build](../patches/lmcache-0.5.1-format10-NL_X_NB_NH_BS_TWO_HS.patch) (identical ABI, single file, no recompile). It launches, composes with MTP, stores land (0 format-10 errors), and the L2 SSD tier fills.
- **But the L2 reload corrupts long-context retrieval.** After a container restart LMCache reloads 16–21K tokens in ~26 ms and the output stays fully coherent — yet planted long-context needles **vanish** (measured **7/7 miss** across two needles; a fresh prefill retrieves every time; the sidecar log confirms LMCache served the reload). Root cause: the format-10 transfer kernel copies bytes for the standard `[NB, NH, BS, 2·HS=512]` fp8/bf16 layout, but k8v4 packs `[…, 262]` (8-bit K + 4-bit V + scales) — the stride mismatch corrupts the L2 serialization round-trip. Coherent-but-lossy is exactly the failure that kills long-context coding.
- **`engine_driven` doesn't rescue it.** The `engine_driven` transfer mode ([LMCache PR #4073](https://github.com/LMCache/LMCache/pull/4073)) reclaims the ~900 MB sidecar VRAM (1,370 → 498 MiB), but its SHM-registration handshake is unstable grafted onto the older-nightly TQ base (300 s `register_kv_caches` timeout), and even the prior working run was parked as unstable (≥30K-prefill OOM) with fidelity unverified.
- **Conclusion.** LMCache's persistence tier only round-trips faithfully with **fp8 KV**. k8v4 keeps vLLM's in-pool `--enable-prefix-caching` (fast in-pool reuse) but **no tiered persistence**. A faithful k8v4 tier would need a new lmcache `KVFormatSpec` + transfer kernel for the 262-wide packed layout (or forcing opaque BINARY blocks) — not worth it single-user.
