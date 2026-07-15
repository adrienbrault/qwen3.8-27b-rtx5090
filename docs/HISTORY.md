# How this config was arrived at — the dead ends and the one real bug

The daily config in this repo (patched TurboQuant image + `turboquant_k8v4` KV) did not arrive in a straight line. For weeks we shipped stock fp8 and treated the TurboQuant image as experimental, chasing what we *thought* was intermittent memory corruption. It wasn't. This page is the story: how the "corruption" turned out to be a noisy detector plus a 4-bit-*key* quality loss, and the one genuine bug — a discarded out-param under CUDA-graph capture — that the patch stack really does fix.

## Status: turboquant_k8v4 is the daily

**2026-07-15 — this reverses our earlier call.** For weeks we shipped stock fp8 KV and treated the patched TurboQuant image as experimental, because it produced constant `!!!!` at **0% MTP acceptance** under real agent sessions and we called it intermittent memory corruption. **That diagnosis was wrong — there was never a corruption bug.** A ~30-round investigation traced the `!!!!` to two mundane causes: (1) a soak-test **degeneracy detector that false-positived** on coherent replies to random-gibberish prompts, and (2) genuine long-context **quality loss from quantizing the *keys* to 4 bits** (`turboquant_4bit_nc`). Keys are what attention indexes on — 4-bit keys destroy long-context retrieval.

**The fix was one flag.** Use **`turboquant_k8v4`** (8-bit Keys / 4-bit Values) instead of `turboquant_4bit_nc`. 8-bit keys preserve retrieval; 4-bit values keep most of the density win. Proven by fair needle-in-haystack retrieval — plant 5-digit codes in coherent filler, exact-match:

| KV cache | 9K | 20K | 40K |
|---|---|---|---|
| `turboquant_4bit_nc` (4-bit keys) | **0/8 across depths — destroys retrieval** | | |
| fp8_e4m3 | 6/6 | 8/8 | — |
| **turboquant_k8v4** | **8/8** | **8/8** | **6/6** |

So the whole TurboQuant + MTP stack and its four patches are **still real and still needed** — the [`!!!!`-from-a-discarded-out-param bug](#the-bug-nobody-caught) was genuine and its fix stands. What's retired is the "demoted to experimental because it corrupts" conclusion. **The daily image is the same clean TQ image, just `VLLM_TQ_PRESET=turboquant_k8v4` + `--kv-cache-dtype turboquant_k8v4`.** It's faster single-stream, **+21% pool** (165K vs 136K tokens, in *less* KV memory), and matches fp8's retrieval and MTP acceptance. fp8 remains the alternative for **deep-context high-concurrency batch serving** (see [Benchmarks](../README.md#benchmarks)). `turboquant_4bit_nc` is a **rejected variant** — 0/8 retrieval is why.

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
