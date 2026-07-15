# How this config was arrived at — the dead ends and the one real bug

The daily config in this repo (patched TurboQuant image + `turboquant_4bit_nc` KV + `--no-async-scheduling`) did not arrive in a straight line — it took **two** reversals. First we shipped stock fp8 for weeks, treating the TurboQuant image as experimental over what we *thought* was intermittent memory corruption; that turned out to be a noisy detector plus 4-bit-*key* quality loss, so we moved to `turboquant_k8v4` (8-bit keys). Then we found the 4-bit-key "quality loss" was itself largely a *third* confound — vLLM's async scheduler corrupting KV under speculative decode — and that one flag (`--no-async-scheduling`) makes the denser `turboquant_4bit_nc` clean and it becomes the daily. This page is that story, newest chapter first, plus the one genuine bug — a discarded out-param under CUDA-graph capture — that the patch stack really does fix.

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
