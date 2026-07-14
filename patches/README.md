# Patches

Applied on top of `vllm/vllm-openai:nightly`. All pure-Python — no CUDA recompile, ~1 min build.

```bash
docker build -t vllm-turboquant:patched .
```

| file | purpose |
|---|---|
| `vllm-only.diff` | Upstream [PR #40914](https://github.com/vllm-project/vllm/pull/40914) (open) — TurboQuant K+1 spec-verify routing. |
| `fix_spec_output.py` | **Makes #40914 actually work on Blackwell.** It returned the kernel tensor instead of writing the out-param buffer; under full CUDA-graph capture the return value is discarded → constant-token garbage. |
| `tq_auto_fallback.py` | MTP draft runner never inherits `cache_config.cache_dtype` (arrives `"auto"`, crashes). Falls back to `$VLLM_TQ_PRESET`. |
| `tq_splits.py` | Exposes TurboQuant's fixed decode KV-split count as `$VLLM_TQ_KV_SPLITS`. Default 32 is optimal — this exists so you can verify that, not change it. |

**After any vLLM nightly bump**: re-verify. These patch a file (`vllm/v1/attention/backends/turboquant_attn.py`)
that upstream actively moves. A failed `patch` fails the build loudly; a *shifted* apply could
silently produce garbage. Test with raw `/v1/completions` — the chat parser hides degeneration
as empty content.

## LMCache format-10 kernel patch (separate project)

`lmcache-0.5.1-format10-NL_X_NB_NH_BS_TWO_HS.patch` is **not** for vLLM — it patches [LMCache](https://github.com/LMCache/LMCache) `csrc/mp_mem_kernels.cu` (a CUDA recompile, unlike the pure-Python patches above). It's only needed for the [MTP + LMCache](../docs/LMCACHE.md) profile **on the nightly pairing**.

| file | purpose |
|---|---|
| `lmcache-0.5.1-format10-NL_X_NB_NH_BS_TWO_HS.patch` | Adds the missing `EngineKVFormat 10` (`NL_X_NB_NH_BS_TWO_HS`) transfer kernel — the fused rank-4 `[NB, NH, BS, 2·HS]` KV layout vLLM's unified hybrid allocator emits. LMCache **defines and detects** the format but implements it in **no kernel** (not 0.5.1, not 0.5.2-dev), so every store aborts with `Unsupported EngineKVFormat: 10`. 17 lines: the two offset-calculator branches (global + local) plus the dispatch case; the Python `KVFormatSpec` registry already maps the axes. Validated on `vllm/vllm-openai:nightly`: 0 format errors, 12/12 hits, composed decode c1 122 / c8 458. **Being prepared as an upstream PR to LMCache** — every vLLM-nightly hybrid is uncacheable without it. Apply against a lmcache 0.5.1 checkout (`git apply`) and build. The [`lmcache-vllm:fixed`](../docs/LMCACHE.md#the-image) 0.24 image needs it **not** — it only bites the nightly path. |
