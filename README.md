# Qwen3.6-27B on a single RTX 5090 — 287K KV pool, fp8 cache, MTP K=4 speculative decoding

Serving **Qwen3.6-27B with a 287K-token KV pool (200K usable context), MTP speculative decoding at `ns=4`, and vision** on one 32 GB consumer GPU (RTX 5090, Blackwell `sm_120`).

The daily is the **[Lorbus INT4-AutoRound](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound)** weights + **`fp8_e4m3` KV cache** + **FlashInfer** attention + **MTP `ns=4`**, on a patched vLLM image. The one patch that makes `ns=4` possible on Blackwell is [vLLM PR #42603](https://github.com/vllm-project/vllm/pull/42603) — without it, MTP + fp8 KV illegal-memory-access-crashes under any real concurrency.

## What you get

- **Qwen3.6-27B** (Lorbus INT4-AutoRound weights, `--quantization auto-round`) over an OpenAI-compatible endpoint.
- **287K-token KV pool → 200K usable context** on `fp8_e4m3` KV — the fp8 attention path is flat with depth (no decode crater) where custom 4-bit-KV kernels crater.
- **MTP speculative decoding at `ns=4`** — draft head inside the weights, ~0 VRAM — crash-free under concurrency thanks to [PR #42603](https://github.com/vllm-project/vllm/pull/42603).
- **Vision** — the model's image tower, on.
- All of it on **one 32 GB RTX 5090** (`sm_120`), memory-OC'd, 600 W.

## Benchmarks

Hardware: RTX 5090 32 GB (`sm_120`, +4500 MHz mem OC, 600 W) + Ryzen 9 5900X + 64 GB RAM, Ubuntu 24.04.
Model: Qwen3.6-27B Lorbus INT4-AutoRound + `fp8_e4m3` KV + FlashInfer 0.6.15 + MTP `ns=4` + vision, `--no-async-scheduling`, util 0.98, pool **287,323 tok**.
Tool: [llama-benchy](https://github.com/eugr/llama-benchy) 0.3.8. Full detail in [bench/RESULTS.md](bench/RESULTS.md).

**Stability — the point of the patch.** Every axis that reliably IMA-crashed on stock (with `ns=4`) now runs zero-crash: full concurrent c4/c8 (pp512+pp4096, ×3), a repeat c8 ×5 stress, deep pp30000 × c4, **deep pp90000 × c4** (worst case — deep + concurrent + `ns=4`), and the full **69×2 tool-eval** under load.

**Decode — `t/s (total)` (aggregate), `--pp 512 4096 --tg 128 --concurrency 1 2 4 8 --runs 3`:**

| decode t/s (total) | c1 | c2 | c4 | c8 |
|---|---|---|---|---|
| pp512 | 114 | 212 | 355 | 496 |
| pp4096 | 129 | 164 | 198 | 157 |

**Long context (c1) — prefill / e2e-TTFT / decode.** Decode is **flat ~128–133 t/s from 30K → 180K** — the fp8 + FlashInfer attention kernel has no deep-context crater, now with `ns=4` spec on top:

| context | prefill t/s | e2e TTFT | decode t/s |
|---|---|---|---|
| 30K | 3,653 | 7.4 s | 128 |
| 90K | 2,924 | 27.9 s | 132 |
| 180K | 2,249 | 72.4 s | 133 |

Deep context holds *under* concurrency too — pp90000 × c4 stays alive at ~102 t/s/req.

**Quality — [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench): 90** / 100 (full 69-scenario suite × 2 trials; mean 88 ± 2.8, pass@k 89.9). The run doubles as the heaviest concurrent-load stability stress on the fix.

## Where the 31.35 GiB goes — memory budget

All numbers below are measured — the boot log (`gpu_worker` prints the breakdown at every launch, 2026-07-18 boot) plus the checkpoint's safetensors headers for the weight split:

| slice | GiB | notes |
|---|---:|---|
| **Weights, total** | **17.73** | split ↓ (17.69 on disk + 0.04 loader) |
| · language model | 11.82 | 48 hybrid layers (GDN + full-attention), INT4-AutoRound |
| · embeddings | 2.37 | bf16 — not quantized |
| · lm_head | 2.37 | bf16 — embed+head = 4.7 GiB, a **27% big-vocab tax** on the weight budget |
| · vision tower | 0.86 | bf16, always loaded (even text-only requests) |
| · MTP drafter | 0.28 | the whole speculative-decoding head costs less than 1% of VRAM |
| **KV pool** | **10.72** | **= 287,323 tokens** @ ~39 KiB/tok (fp8 attention KV + GDN state pages, one unified pool) |
| **Peak-activation reserve** | 1.89 | sized by profiling at `mnbt` 4096 |
| **CUDA graphs + non-torch** | 0.37 | |
| **util 0.98 budget** | **30.72** | of 31.35 usable; the last ~0.6 GiB of margin is burst-tested (text + vision transients), not guessed |

What each token costs and what a cache hit is worth (60K-token context, measured):

| tier | capacity | revisit cost | status |
|---|---|---|---|
| GPU pool (vLLM prefix cache) | 287,323 tok | **~1–2 s** (≈ decode time only) | production |
| host RAM (LMCache L1, 24 GiB pinned) | ~245K tok @ ~98 KiB/tok serialized | ~2 s | **experimental — failed fidelity validation, see below** |
| NVMe (LMCache L2) | ~640K tok per 60 GiB | **3.4 s** (survives restarts) | **experimental — failed fidelity validation, see below** |
| miss → full re-prefill | — | **~23 s** | the thing caches exist to avoid |

Notes that save you from wrong conclusions:

- **util is the only pool lever** (+~8.4K tok per 0.01); `max-num-seqs` provably isn't on this hybrid (`align` packs GDN state *into* the unified pool, consumed per **active** request — seqs 4 vs 8 gave the identical pool).
- The in-pool ~39 KiB/tok is an *effective average*: full-attention layers pay per-token fp8 KV; GDN layers pay a fixed per-sequence state that amortizes with depth. Serialized tiers pay ~98 KiB/tok because LMCache ships whole 1616-token unified blocks (attention KV + state page + metadata, padded to full block).
- Effective deep concurrency is **pool-bound, not `max-num-seqs`-bound**: 4 × ~70K-token agent sessions fill the pool; request 5 queues.
- The RAM/NVMe tiers require LMCache ≥ the [PR #4128](https://github.com/LMCache/LMCache/pull/4128) kernels for this model's fused hybrid layout — release pairings up to 0.5.1 silently store nothing, and our own [withdrawn kernel patch](patches/README.md#lmcache-format-10-kernel-patch-separate-project) stored corrupted GDN state. Until a pairing passes a **cross-restart needle test**, treat every external tier as unvalidated ([details](docs/LMCACHE.md)).

## Why it needs a patch: MTP × fp8-KV × Blackwell crashes on stock vLLM

> ⚠️ **`ns≥2` MTP with `fp8_e4m3` KV on `sm_120` is a 100%-reproducible illegal-memory-access under concurrency** — a known, still-open upstream bug ([vllm#40756](https://github.com/vllm-project/vllm/issues/40756), same Qwen3.6-27B model; [vllm#35288](https://github.com/vllm-project/vllm/issues/35288) "MTP corrupted output at concurrency ≥ 4"). It crashes at `rejection_sampler.py:267 parse_output → cudaErrorIllegalAddress`. Single-stream and `ns=1` are both clean; `CUDA_LAUNCH_BLOCKING=1` masks it (→ a timing race).

The root cause ([PR #42603](https://github.com/vllm-project/vllm/pull/42603)): the MTP draft loop in `vllm/v1/spec_decode/llm_base_proposer.py` writes shared cudagraph input buffers (`input_ids`, `hidden_states`), then immediately launches the draft-model forward that reads them — **without a stream sync**. CUDA is async, so under concurrency the draft FlashInfer kernels observe stale / partially-written buffers → illegal access. The fix is one line:

```python
self.input_ids[:batch_size] = input_ids
self.hidden_states[:batch_size] = hidden_states
torch.accelerator.current_stream().synchronize()   # PR #42603
```

That's it. It restores `ns=4` at full pool, is **perf-neutral**, and is validated crash-free across the concurrent + deep-context + tool-eval load that reliably crashed before (see [Benchmarks](#benchmarks)). [`patches/install_pr42603_sync.py`](patches/install_pr42603_sync.py) is an idempotent, `ast`-verified graft of it onto the installed vLLM tree.

Plus one quality graft: **[PR #44993](https://github.com/vllm-project/vllm/pull/44993)** — structured output (`response_format` json_schema) silently returned empty `content` under a reasoning model (the schema JSON leaked into `reasoning_content`). Fixed with the `--structured-outputs-config` flag below.

## Quick start

```bash
# build the patched image (~1 min of pure-Python patches on top of the vLLM base;
# the FlashInfer 0.6.15 step JIT-compiles its kernels on first run, so mount its cache)
cd patches && docker build -t vllm-qwen36:patched .

# serve
./scripts/serve.sh
```

Then `http://localhost:8020/v1` speaks OpenAI. Every flag is explained inline in [`scripts/serve.sh`](scripts/serve.sh) and in [Config essentials](#config-essentials) below.

## What's in the patch stack

The daily uses these three; the image also carries an (unused-by-this-config) TurboQuant 4-bit-KV stack documented separately in [docs/HISTORY.md](docs/HISTORY.md).

| patch | what it does |
|---|---|
| [`install_pr42603_sync.py`](patches/install_pr42603_sync.py) — [PR #42603](https://github.com/vllm-project/vllm/pull/42603) | **The fix that makes MTP `ns=4` usable on Blackwell.** One `torch.accelerator.current_stream().synchronize()` in the MTP draft loop, after the cudagraph-buffer writes and before the draft forward — closes the stale-buffer race that IMA-crashes `ns≥2` + fp8 KV under concurrency ([#40756](https://github.com/vllm-project/vllm/issues/40756), [#35288](https://github.com/vllm-project/vllm/issues/35288)). Perf-neutral. [Validated numbers](bench/RESULTS.md#mtp-k4-restored-on-blackwell--the-fp8-kv-spec-decode-crash-pr-42603). |
| FlashInfer 0.6.15 (Dockerfile pip step) | Latest FlashInfer, carrying the `sm_120` GDN/TMA fixes. cu130 AOT cubin/jit-cache isn't published for .15, so the image drops the mismatched 0.6.13 caches and lets 0.6.15 JIT-compile at runtime — **mount `/root/.cache/flashinfer`** (one build, warm forever). |
| [PR #44993 graft](https://github.com/vllm-project/vllm/pull/44993) — `v1/structured_output/__init__.py`, `v1/core/sched/scheduler.py` | **Structured output that survives thinking.** With a reasoning model, `response_format` json_schema + thinking-on returned EMPTY `content` (the schema JSON leaked into `reasoning_content`) — `should_advance`'s delta window skips `</think>` when MTP rejects drafts, so the grammar never re-engages. Needs `--structured-outputs-config '{"backend":"xgrammar","reasoning_parser":"qwen3","enable_in_reasoning":false}'`. Two pure-Python files. |

## Config essentials

`./scripts/serve.sh` runs the daily (flags also annotated inline there). The load-bearing ones:

- `--quantization auto-round` — Lorbus INT4-AutoRound weights (compressed-tensors AutoRound; native in the vLLM base, no extra patch).
- `--kv-cache-dtype fp8_e4m3` + `-e VLLM_ATTENTION_BACKEND=FLASHINFER` — the flat-with-depth attention path; `e5m2` is **not** usable on this checkpoint (vLLM: "fp8_e5m2 not supported with fp8 checkpoints").
- `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":4}'` — MTP `ns=4`. **Requires the [PR #42603](https://github.com/vllm-project/vllm/pull/42603) graft** or it IMA-crashes under concurrency.
- **`--no-async-scheduling` — keep it.** MTP emits a multi-token verify batch every step; vLLM's async scheduler desyncs its request-ID→batch-row mapping under spec decode ([vllm#42655](https://github.com/vllm-project/vllm/issues/42655)) and corrupts KV. Async-off is the documented fix.
- `--mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill` — hybrid-model prefix caching; **`align` is load-bearing for the 287K pool** (it packs the GDN/Mamba state into the unified KV pool). Mode `all` is same speed / same pool and does **not** avoid the crash — don't bother.
- `--gpu-memory-utilization 0.98 --max-model-len 200000 --max-num-batched-tokens 4096` — let vLLM profile the pool; don't hand-set `--kv-cache-memory` (its "fully utilize" hint ignores warmup transients and OOMs). util is the only pool lever here (~+8.4K tok/0.01 → 287,323 at 0.98); `mnbt` doesn't change the pool, and at high util `mnbt 4096` avoids a ~9% deep-prefill slowdown that `mnbt 8192` incurs. The ceiling was probed above 0.98 with near-full text bursts **and** concurrent max-res vision bursts — see the [util sweep + ceiling probe](bench/RESULTS.md#pool-vs-util--the-util-ceiling). 0.96 / 0.94 are the fallback floors.
- `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml` — `qwen3_xml` is correct; `hermes` silently drops tool calls.
- `--override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20}'` + `--default-chat-template-kwargs '{"preserve_thinking":true}'` — keep historical `<think>` blocks across turns. **Caveat:** the *client* must resend prior reasoning in the **`reasoning`** field (not the deprecated `reasoning_content`, which vLLM ignores on input).
- `--structured-outputs-config '{"backend":"xgrammar","reasoning_parser":"qwen3","enable_in_reasoning":false}'` — enables the reasoning gate the [#44993](https://github.com/vllm-project/vllm/pull/44993) graft needs. Give it an adequate `max_tokens` budget (reasoning + JSON) or it truncates mid-think and looks empty.
- `--limit-mm-per-prompt '{"image":4,"video":0}'` — vision on.

**Verify container identity after launch.** Confirm the startup log reports a **~287K-token KV pool** (at util 0.98). A wrong pool size means a preset/dtype/align mismatch reading as a perfectly healthy server at the *wrong* config.

## Gotchas that bite during setup

1. **MTP `ns≥2` needs [PR #42603](https://github.com/vllm-project/vllm/pull/42603) or it IMA-crashes under concurrency.** Single-stream and `ns=1` pass every test and hide it; `CUDA_LAUNCH_BLOCKING=1` masks it. **Load-test with 3+ parallel streams on day one** — that's the only thing that reproduces it.
2. **`--no-async-scheduling` is mandatory with MTP.** Async scheduling desyncs the request-ID→batch-row mapping under spec decode ([#42655](https://github.com/vllm-project/vllm/issues/42655)) and corrupts KV.
3. **Verify the ~287K KV pool in the launch log.** A silent config fallback (wrong preset, `align` off, wrong image) looks like a healthy server at the wrong pool.
4. **flashinfer JIT eats all host RAM (non-nightly images).** Any non-nightly vLLM image on `sm_120` JIT-compiles CUTLASS kernels on the first forward with unbounded `nvcc` parallelism — multi-GB per job, reads as a mystery "hang" or whole-host livelock. Cap it (`MAX_JOBS=4` + `FLASHINFER_NUM_COMPILE_JOBS=4`) and **mount a persistent `/root/.cache/flashinfer`** (one build, warm forever).
5. **vLLM's prefix-cache metric lies on this model.** `vllm:prefix_cache_hits_total` / "Prefix cache hit rate: 0.0%" report **0% while the cache works**. Don't debug the counter — time a repeated prompt ([`bench/prefix_probe.py`](bench/prefix_probe.py)).
6. **Validate coherence via raw `/v1/completions`.** The chat endpoint's reasoning parser swallows degenerate output as *empty content*, so a broken model looks "fine but quiet." Tells: constant-token output, or **flat 100% MTP acceptance** (draft and verify locked in step).
7. **Re-verify after every vLLM bump.** These grafts are version-sensitive; a nightly that moves `llm_base_proposer.py` will fail the `ast` check at build (loud) or, worse, shift the anchor.
8. **Burst-test the util ceiling — a ramping benchmark won't.** The failure mode at high util is a *cold-start* transient: many simultaneous fresh prompts hitting prefill (or the vision encoder) at once, which a benchmark that ramps concurrency never reproduces. A [util × mnbt sweep + ceiling probe](bench/RESULTS.md#pool-vs-util--the-util-ceiling) burst-tested `{0.94 … 0.98}` with near-pool-full **distinct-prompt** text bursts, 8× concurrent 4-image (2048²) vision bursts, and mixed vision+deep-text — all survive, so the daily runs **0.98** (287K pool, ~600 MB VRAM margin). Two traps when you reproduce this: (a) identical prompts are silently collapsed by prefix caching and never fill the pool — use distinct prompts; (b) text-only bursts miss the vision-encoder transient. If you change model/context/hardware, re-run the bursts before trusting high util; 0.96/0.94 are the fallback floors. (An earlier note here claimed 0.95 crashes — that was pre-sweep and is superseded.)

## Daily lineage — what each daily was, and why the next took over

Newest first. Every switch is documented with numbers in [docs/HISTORY.md](docs/HISTORY.md) and [bench/RESULTS.md](bench/RESULTS.md).

| daily | weights · KV | pool | why it took over |
|---|---|---|---|
| **Lorbus INT4-AutoRound · fp8_e4m3** + FlashInfer + MTP `ns=4` **(current, 2026-07-18)** | INT4-AutoRound · fp8 | **~287K** | Flat deep decode (fp8+FlashInfer has **no** decode crater at depth, where the custom TurboQuant kernel drops); biggest pool yet; **MTP `ns=4`** restored by [PR #42603](https://github.com/vllm-project/vllm/pull/42603); tool-eval 90; and it drops the experimental TurboQuant KV kernel for the **battle-tested fp8** path. Smaller INT4-AutoRound weights free more VRAM for KV than NVFP4, so fp8 KV reaches 253K at util 0.94 (vs ~136K under NVFP4 weights), and **287K at util 0.98** after a burst-tested [util sweep + ceiling probe](bench/RESULTS.md#pool-vs-util--the-util-ceiling) (text near-full + concurrent vision bursts all pass). |
| turboquant_4bit_nc (NVFP4) + MTP `ns=3` (2026-07-15) | NVFP4 · TQ 4-bit K/V | ~235K | +42% pool over k8v4, once the "4bit_nc destroys retrieval" **0/8** was traced to the async×spec KV confound and fixed with `--no-async-scheduling`. Decode still craters at deep single-stream (the custom-kernel cost). |
| turboquant_k8v4 (NVFP4) | NVFP4 · TQ 8-bit K/4-bit V | ~165K | +21% pool over fp8 at fp8-equal retrieval quality (8-bit keys). |
| fp8_e4m3 (stock nightly) | NVFP4 · fp8 | ~136K | The original battle-tested baseline — flat deep decode, no patches, smallest pool. |

The current daily is, in effect, the fp8 baseline's **flat-decode virtue** brought back — on stronger INT4-AutoRound weights, with a bigger pool and working `ns=4` — now that PR #42603 makes MTP + fp8 KV crash-free on Blackwell.

## How we got here / what didn't work

- **[docs/HISTORY.md](docs/HISTORY.md)** — the full path to this config, including the multi-round bisection that localized the `ns=4` crash: barriers placed *around* the proposer in `gpu_model_runner` all fired and still crashed (the race is *inside* the proposer loop); `--mamba-cache-mode all` still crashed (so it isn't the fused Mamba postprocess); a draft-token sanitizer never fired (the fault was never a bad token) — until the upstream trail led to [PR #42603](https://github.com/vllm-project/vllm/pull/42603). Also documents the earlier TurboQuant 4-bit-KV and NVFP4 work that this repo previously shipped as the daily.
- **[docs/REJECTED.md](docs/REJECTED.md)** — everything tried and rejected, with the number that killed it. Read it before "improving" the config.

## License

MIT (see [LICENSE](LICENSE)) for the original work here — docs, benchmarks, scripts, and the grafts. The vLLM files the patches modify stay under **Apache-2.0**; see [THIRD_PARTY.md](THIRD_PARTY.md).

## Credits

- [vLLM](https://github.com/vllm-project/vllm) and [PR #42603](https://github.com/vllm-project/vllm/pull/42603) — the draft-loop sync that makes MTP `ns=4` usable on Blackwell.
- [Lorbus](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) for the Qwen3.6-27B INT4-AutoRound quant.
- [llama-benchy](https://github.com/eugr/llama-benchy), [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) for the measurements.
