# How this config was arrived at — the dead ends and the one real bug

## Daily lineage — what each daily was, and why the next took over

Newest first. Every switch is documented with numbers in the Status sections below and [../bench/RESULTS.md](../bench/RESULTS.md).

| daily | weights · KV | pool | why it took over |
|---|---|---|---|
| **+ LMCache DRAM/NVMe tiers (current, 2026-07-20)** | *(same engine)* | **214K** @0.95 **+ ~2.4M tiered** | Six local patches made tiered KV *faithful* on this fp8 hybrid (cross-restart needle + 69×2 = **89** vs a ~89.8 baseline). Trades 25K hot tokens and `mnbt` 4096 for ~2.4M tokens of second-chance capacity and a **warm start after restarts** — a 60K revisit costs 2 s (DRAM) or 4–7 s (NVMe) instead of an 11–13 s re-prefill. Validated by an 858-cycle soak (flat VRAM, L2 stable under its cap). [`../scripts/serve-plain.sh`](../scripts/serve-plain.sh) keeps the row below available. |
| natfii NVFP4 W4A4 · fp8_e4m3 + FlashInfer + MTP `ns=4` (2026-07-19) | NVFP4 W4A4 · fp8 | ~239K @0.98 | **Prefill 3.4×** (13.5K vs 4.0K t/s @8K — native Blackwell FP4 GEMM vs Marlin dequant), deep-concurrent sustained **2.2×** (148 vs 67 t/s at pp30K×c8 tg512), cold 60K context 10 s vs 23 s — at **equal 69×2 quality** (~90, 4 trials each side; the W4A4 activation cost was bounded at ≈1 pt via a chimera A/B and natfii's calibration covers it). Survived the full promotion gauntlet incl. a 106-cycle soak and a 0.98 combined-wave battery. Pool is 11% smaller than AR's 270K (heavier MTP head + FP4 scales) — traded for re-prefilling 3× faster. |
| Lorbus INT4-AutoRound · fp8_e4m3 + FlashInfer + MTP `ns=4` (2026-07-18) | INT4-AutoRound · fp8 | ~270K @0.96 | Flat deep decode (fp8+FlashInfer has **no** decode crater at depth, where the custom TurboQuant kernel drops); biggest pool ever; **MTP `ns=4`** restored by [PR #42603](https://github.com/vllm-project/vllm/pull/42603); tool-eval 90; dropped the experimental TurboQuant KV kernel for the **battle-tested fp8** path. (A one-day 0.98/287K promotion was reverted the same night: serve-time autotune OOM — [GOTCHAS.md](GOTCHAS.md) #8.) |
| turboquant_4bit_nc (NVFP4) + MTP `ns=3` (2026-07-15) | NVFP4 · TQ 4-bit K/V | ~235K | +42% pool over k8v4, once the "4bit_nc destroys retrieval" **0/8** was traced to the async×spec KV confound and fixed with `--no-async-scheduling`. Decode still craters at deep single-stream (the custom-kernel cost). |
| turboquant_k8v4 (NVFP4) | NVFP4 · TQ 8-bit K/4-bit V | ~165K | +21% pool over fp8 at fp8-equal retrieval quality (8-bit keys). |
| fp8_e4m3 (stock nightly) | NVFP4 · fp8 | ~136K | The original battle-tested baseline — flat deep decode, no patches, smallest pool. |

The arc, compressed: the fp8 baseline's **flat-decode virtue** survived every generation; PR #42603 added working `ns=4`; AutoRound added quality and pool; NVFP4 W4A4 cashed in the GPU's native FP4 compute — the first daily where *prefill* got a generational jump instead of decode or capacity; and the KV tiers finally broke capacity out of the 32 GB box entirely, which is the one axis a bigger GPU would otherwise have been the only answer to.

## Status: the daily added LMCache DRAM/NVMe KV tiers (2026-07-20)

**Same natfii engine as the section below, plus the tiered KV offload** — util 0.95, pool 214,084, +24 GB pinned DRAM (~245K tok) +200 GB NVMe (~2.13M tok, restart-proof), at quality parity (69×2 = 89 vs ~89.8; later confirmed **89.0 ± 1.4** over a 69×4 re-run — [cross-trial stats](../bench/RESULTS.md#tool-eval-cross-trial-statistics--694-on-the-tier-daily-2026-07-22)) after six local patches. That campaign — four rounds of "validated" tier profiles that were silently wrong — is its own story: [LMCACHE.md](LMCACHE.md) and [../patches/lmcache/README.md](../patches/lmcache/README.md). The section below documents the natfii engine promotion this profile is built on; its util-0.98/239,436 numbers are now the *plain* (no-tiers) profile.

**Same era, the final word on TurboQuant.** With boot unblocked on the modern stack, `turboquant_4bit_nc` got one last audition against the promotion gates: it reached a **563,888-token pool** (2.6× the daily's) — and failed both disqualifying gates, returning a **wrong 60K needle** (the hybrid buffer co-location corruption class, unfixed) and **dying under the `pp8192×c8` concurrency killer**. A pool that size is worth nothing if retrieval lies, so TurboQuant KV is **closed permanently for this hybrid**: revival would need a state-corruption fix, a concurrency fix, a deep-decode-crater fix, *and* new LMCache serde kernels for its packed layout — against a working fp8 tier stack. The 2026-07-15 un-rejection below remains true as far as it went (the 0/8 *was* the async confound); it just wasn't the whole story. [REJECTED.md](REJECTED.md) carries the verdict.

## Status: the daily is natfii NVFP4 W4A4 — prefill's turn (2026-07-19)

**The daily moved to the [natfii W4A4 NVFP4 export](https://huggingface.co/natfii/Qwen3.6-27B-VLM-NVFP4-MTP) at util 0.98, pool 239,436.** The whole case in one line: equal quality (69×2 pooled 89.8 vs 87.8, 4 trials each), **prefill 3.4×** (13.5K vs 4.0K t/s @8K — W4A4 dispatches Blackwell's native FP4 GEMM where W4A16 runs bf16 GEMM plus Marlin dequant), deep-concurrent sustained 2.2× (148 vs 67 t/s at pp30K×c8 tg512), cold-60K TTFT 10 s vs 23 s. Cost: pool −11% (natfii's MTP head is 0.79 GiB vs AR's 0.28, plus FP4 scale tensors).

Three findings from the campaign worth keeping:

- **The W4A4 quality question was settled by construction, not vibes.** We built a *chimera* checkpoint — natfii's W4A4 MLPs + NVIDIA's fp8 attention projections merged tensor-by-tensor into one MIXED_PRECISION export (both kernel classes co-dispatching in one graph) — and scored all three variants on the full 69×2. Chimera 90.0, natfii ~89.8, NVIDIA 91.0: the *entire* activation-quant cost is ≈1 point, natfii's calibration covers it, and no attention/MLP remix beats it without requantizing. The chimera was archived the day it answered the question.
- **The util ceiling is model-specific.** 0.98 serve-time-OOM'd the AR daily (addendum below) but passes on natfii — lighter margin pressure per shape, `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=134217728` capping the autotune workspace, and a boot-time `pp8192×c8` pre-warm so the allocation happens before traffic. Earned via: full battery at 0.98 (needle, both killer shapes, 8× text flood, 8× vision burst), then two *simultaneous* combined waves on a cold engine, then a 106-cycle overnight soak. Steady-state floor ~130–190 MiB, no drift.
- **The checkpoint shipped a poison pill:** `tokenizer.json` with `truncation: {max_length: 8192}` baked in (calibration leftover) — text fine, multimodal >8K tokens hard-400s. `serve.sh` now nulls it at every launch, because a re-download reinstates it.

## Status addendum: util 0.98 lived one day (2026-07-18 → 2026-07-19)

The 0.98/287K promotion below was reverted the same night it shipped. 0.98 boots clean and survives every boot-time burst probe (near-pool-full distinct-prompt text, 8× concurrent 4-image vision, mixed) — and then OOM-dies **in production** the first time the fp4-GEMM/FlashInfer autotuner meets a genuinely new deep batch shape: a fresh `pp8192 × c8` wave allocates ~266 MiB of lazy benchmark workspace at serve time against ~600 MB of margin (2/2 reproducible, `EngineDead`, full log captured). No boot-margin methodology can see this failure — the workspace only materializes when traffic produces the shape. The AR daily then ran **util 0.96, pool 270,422** (~1.2 GiB margin), validated against that exact killer shape plus the full burst battery, and `benchy --pp 8192 --concurrency 8` is a permanent member of the promotion gauntlet. (The *natfii* daily above later re-earned 0.98 — with the workspace-cap env, a boot pre-warm, and this exact battery; the ceiling is model-specific, the methodology is not.) `mnbt 8192` (a +15–20% concurrent-decode candidate) wants ~486 MiB of workspace and only fits at ≲0.94 — measured, parked.

The daily config in this repo (patched TurboQuant image + `turboquant_4bit_nc` KV + `--no-async-scheduling`) did not arrive in a straight line — it took **two** reversals. First we shipped stock fp8 for weeks, treating the TurboQuant image as experimental over what we *thought* was intermittent memory corruption; that turned out to be a noisy detector plus 4-bit-*key* quality loss, so we moved to `turboquant_k8v4` (8-bit keys). Then we found the 4-bit-key "quality loss" was itself largely a *third* confound — vLLM's async scheduler corrupting KV under speculative decode — and that one flag (`--no-async-scheduling`) makes the denser `turboquant_4bit_nc` clean and it becomes the daily. This page is that story, newest chapter first, plus the one genuine bug — a discarded out-param under CUDA-graph capture — that the patch stack really does fix.

## Status: the daily is now Lorbus INT4-AutoRound + fp8 + MTP ns=4 (the PR #42603 sync workaround)

**2026-07-18 — the daily moved off TurboQuant KV to a simpler, battle-tested path: Lorbus INT4-AutoRound weights + `fp8_e4m3` KV + FlashInfer + MTP `ns=4`.** The fp8 attention path is flat with depth (no decode crater), the pool is **~287K** at util 0.98 (bigger than the 235K TurboQuant daily below), and tool-eval is **90**. The one thing that blocked it was a crash — and finding it was a long bisection worth recording, because almost every "obvious" fix was wrong.

**The crash.** MTP `ns≥2` + `fp8_e4m3` KV on Blackwell `sm_120` illegal-memory-accesses at `rejection_sampler.py:267 parse_output` under **any** real concurrency (c≥4). Single-stream is clean; `ns=1` is clean; `CUDA_LAUNCH_BLOCKING=1` masks it → a timing race.

**The dead ends (all empirically killed, each a fresh build + concurrent repro):**
- **Device-wide `torch.accelerator.synchronize()` barriers in `gpu_model_runner`** — placed after the accepted-state postprocess (before the proposer), before the `bookkeep` block, and inside `execute_model` right after the Mamba input staging. All three **fired** (log-proven) and **all three still crashed**. So it is not an ordering race anywhere in the model runner's spec loop — a late sync can't repair an already-wrong value.
- **`--mamba-cache-mode all`** (skips the fused `align` Mamba postprocess kernel entirely) — **still crashed**, same pool. So it is not the fused Mamba postprocess.
- **A draft-token sanitizer** (drop any `id<0 or id≥vocab` before the rejection sampler, the closed upstream [#46574](https://github.com/vllm-project/vllm/pull/46574) approach) — the drop-warning **never fired**. The fault was never an out-of-vocab draft token.

**The working hypothesis and the workaround — [vLLM PR #42603](https://github.com/vllm-project/vllm/pull/42603).** A search of upstream turned up [#40756](https://github.com/vllm-project/vllm/issues/40756) (same Qwen3.6-27B-FP8 model) and [#35288](https://github.com/vllm-project/vllm/issues/35288) ("MTP corrupted output at concurrency ≥ 4") — a known, still-open class: **MTP × fp8 KV × Blackwell**. The PR's hypothesis: the MTP draft loop in `vllm/v1/spec_decode/llm_base_proposer.py` writes shared cudagraph buffers (`input_ids`/`hidden_states`) then launches the draft forward reading them without a stream sync, so under concurrency the draft FlashInfer kernels read stale buffers. That locality — *inside* the proposer loop — is at least consistent with why every `gpu_model_runner` barrier missed it. The workaround is one `torch.accelerator.current_stream().synchronize()` after the writes. **Upstream closed the PR unmerged**: maintainers held these operations should already be stream-ordered, that a forced sync may only perturb timing, and asked for a proven root cause. Our claim is therefore strictly empirical: on this profile the crash is 100%-reproducible without the sync and has never occurred with it — validated across concurrent c4/c8 + deep pp90000×c4 + the full 69×2 tool-eval, perf-neutral. Grafted as [`patches/install_pr42603_sync.py`](../patches/install_pr42603_sync.py); the true upstream root cause remains unresolved.

**Lesson (again):** localize before fixing. `CUDA_LAUNCH_BLOCKING` naming the *class* (timing race), then serializing one candidate edge at a time to prove where it *isn't*, is what pointed at the proposer's own loop — and the upstream issue search is what turned a multi-day bisection into a one-line graft.

---

## Status: turboquant_4bit_nc is the daily (the async×spec reversal)

**2026-07-15 — this reverses the "`turboquant_4bit_nc` destroys retrieval" call recorded in the next section.** We had rejected `turboquant_4bit_nc` (4-bit Keys) because it scored a catastrophic **0/8** on needle-in-haystack, and concluded "4-bit keys destroy long-context retrieval." That conclusion was a **confound**. The real culprit was vLLM's **async scheduling × speculative decode** interaction — a batch-row-mapping desync ([vllm#42655](https://github.com/vllm-project/vllm/issues/42655)).

**Mechanism.** With MTP, every scheduler step emits a multi-token *verify* batch, so even a **single** in-flight request occupies multiple batch "rows." vLLM's async scheduler computes its request-ID→batch-row mapping one step ahead of execution; under those multi-row verify batches the mapping **desyncs**, and KV gets written to the wrong slots. 4-bit keys are far more sensitive to that corruption than 8-bit keys — so it manifested as a **catastrophic 0/8 for `4bit_nc`** but only **~10% intermittent degradation for `k8v4`** (which is why k8v4 seemed "fine" and `4bit_nc` seemed "broken," when both were being corrupted by the same bug).

**The fix is one flag: `--no-async-scheduling`.** With it, and after confirming the engine actually loaded genuine `4bit_nc` (log shows a **~235K pool**, not a silent 165K `k8v4` fallback), `turboquant_4bit_nc` is completely clean:

| test (with `--no-async-scheduling`) | turboquant_4bit_nc |
|---|---|
| single-stream needle-in-haystack @9K / 20K / 40K | **8/8 / 8/8 / 8/8** |
| high-pressure concurrent (3 rounds × 30 needles, 6 background loaders) | **90/90** |
| tool-eval-bench v2.1.0 (matched protocol, hardmode) | **89** (parity with k8v4) |

That 90/90 is *the exact test* that supposedly "kills all 4-bit-KV kernels." It passes.

**Why it's now the daily.** `4bit_nc` buys **~235K pool → 200K usable context** (+42% pool, +25% context vs `k8v4`'s 165K→160K) for a small decode cost — fresh same-session llama-benchy (both async-off): pp512 decode c1 137→133 (−3%), c8 467→435 (−7%); pp4096 c1 145→126 (−13%), c8 parity. The 4-bit-key dequant (Lloyd-Max codebook + per-GQA-head norm-correction; the inverse Hadamard is hoisted to one per-query GEMM, not per key) costs more than k8v4's cheap FP8-cast keys, worst at deep single-stream. A pool-for-modest-decode trade, and interactive coding is exactly the low-concurrency / deep-context regime where the extra 40K of context is worth ~10% decode.

**Broader implication.** The whole "all custom 4-bit-KV kernels corrupt under concurrency" belief — which this repo's [`fix_spec_guard.py`](#the-bug-nobody-caught) patch partly addressed — was very likely **this async×spec scheduler bug, not the kernels.** We have **not** retested whether the existing guard patches are still needed with async scheduling off, so they stay in the stack for now; the [`fix_spec_output.py`](#the-bug-nobody-caught) out-param fix (#40914) is a genuinely separate bug and stands regardless. `--no-async-scheduling` is the essential *companion* flag, not a replacement for the patches.

**`turboquant_k8v4`** — the prior daily, documented in the next section — remains a good **decode-optimal middle ground** (a bit faster than `4bit_nc`, smaller pool). It is no longer rejected as "the fix for 4bit_nc"; both are shipping presets now. fp8 stays the deep-context high-concurrency batch alternative.

## Status: turboquant_k8v4 (the prior daily)

> **Superseded 2026-07-15** by `turboquant_4bit_nc` (above). Kept as the record of the *first* reversal — from "TurboQuant corrupts" to k8v4 — and because k8v4 is still a shipping alternative. The claim below that "`turboquant_4bit_nc` scored 0/8, 4-bit keys destroy retrieval" is the call the section above overturns: that 0/8 was async×spec corruption, not the keys.

**2026-07-15 — this reverses our earlier call.** For weeks we shipped stock fp8 KV and treated the patched TurboQuant image as experimental, because it produced constant `!!!!` at **0% MTP acceptance** under real agent sessions and we called it intermittent memory corruption. **That diagnosis was wrong — there was never a corruption bug.** A ~30-round investigation traced the `!!!!` to two mundane causes: (1) a soak-test **degeneracy detector that false-positived** on coherent replies to random-gibberish prompts, and (2) genuine long-context **quality loss from quantizing the *keys* to 4 bits** (`turboquant_4bit_nc`). Keys are what attention indexes on — 4-bit keys destroy long-context retrieval.

**The fix was one flag.** Use **`turboquant_k8v4`** (8-bit Keys / 4-bit Values) instead of `turboquant_4bit_nc`. 8-bit keys preserve retrieval; 4-bit values keep most of the density win. Proven by fair needle-in-haystack retrieval — plant 5-digit codes in coherent filler, exact-match:

| KV cache | 9K | 20K | 40K |
|---|---|---|---|
| `turboquant_4bit_nc` (4-bit keys) — *async on* | **0/8 across depths** | | |
| fp8_e4m3 | 6/6 | 8/8 | — |
| **turboquant_k8v4** | **8/8** | **8/8** | **6/6** |

*(We now know the `4bit_nc` 0/8 above was measured with async scheduling on — it was the [async×spec desync](#status-turboquant_4bit_nc-is-the-daily-the-asyncspec-reversal), not the 4-bit keys. With `--no-async-scheduling`, `4bit_nc` scores 8/8 at all three depths. The k8v4 and fp8 rows still hold.)*

So the whole TurboQuant + MTP stack and its four patches are **still real and still needed** — the [`!!!!`-from-a-discarded-out-param bug](#the-bug-nobody-caught) was genuine and its fix stands. What's retired is the "demoted to experimental because it corrupts" conclusion. **The daily image is the same clean TQ image, just `VLLM_TQ_PRESET=turboquant_k8v4` + `--kv-cache-dtype turboquant_k8v4`.** It's faster single-stream, **+21% pool** (165K vs 136K tokens, in *less* KV memory), and matches fp8's retrieval and MTP acceptance. fp8 remains the alternative for **deep-context high-concurrency batch serving** (see [Benchmarks](../README.md#benchmarks)). *(At the time we also concluded `turboquant_4bit_nc` was a rejected variant on its 0/8 retrieval — the [section above](#status-turboquant_4bit_nc-is-the-daily-the-asyncspec-reversal) overturns exactly that, once the async×spec bug is removed.)*

## The bug nobody caught

vLLM PR #40914 fixes TurboQuant's spec-decode routing. Its new branch ends:

```python
attn_out = triton_turboquant_decode_attention(...)
return attn_out          # ← the bug
```

But `TurboQuantImpl.forward()` is invoked as a **mutated-out-param custom op** (`unified_attention_with_output`). Under `FULL_AND_PIECEWISE` CUDA-graph capture, **the return value is discarded** — the caller reads the `output` buffer. So attention output stays stale/zeroed and the model decodes a constant token:

```
prompt: "def fib(n):"
output: "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
```

Every other branch of `forward()` writes the buffer. That one didn't. The fix ([`patches/fix_spec_output.py`](../patches/fix_spec_output.py)):

```python
if output.ndim == 3:
    output[:N] = attn_out.to(output.dtype)
else:
    output[:N] = attn_out.reshape(N, -1).to(output.dtype)
return output
```

This is almost certainly why the PR passed on the author's Ampere box (the eager/piecewise path *does* consume the return value) and fails on Blackwell, where full CUDA-graph capture is the default.
