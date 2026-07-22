# Patches

Applied on top of `vllm/vllm-openai:nightly`. All pure-Python — no CUDA recompile, ~1 min build.

```bash
docker build -t vllm-qwen36:patched .
```

> **The daily needs a second image on top of this one.** [`lmcache/`](lmcache/README.md) holds the six patches that make DRAM/NVMe KV offload *faithful* on this fp8 hybrid — four on LMCache, two on vLLM. Build it after this one. If you're running [`../scripts/serve-plain.sh`](../scripts/serve-plain.sh) (no tiers), this image is all you need.

**What the current fp8 daily actually uses from this image:** [`install_pr42603_sync.py`](install_pr42603_sync.py), the FlashInfer 0.6.15 pip step, and the [PR #44993](https://github.com/vllm-project/vllm/pull/44993) graft (last row). The four `tq_*`/`#40914` rows are the retired TurboQuant-era stack — inert on the fp8 path, kept so the image still serves those historical configs ([REJECTED.md](../docs/REJECTED.md)).

| file | purpose |
|---|---|
| [`install_pr42603_sync.py`](install_pr42603_sync.py) — from [PR #42603](https://github.com/vllm-project/vllm/pull/42603) (closed unmerged) | **The daily-critical one.** One stream-sync in the MTP draft loop — the locally validated workaround for the MTP × fp8-KV × Blackwell illegal-memory-access crash. [Why, in the README](../README.md#why-it-needs-a-patch-mtp--fp8-kv--blackwell-crashes-on-stock-vllm). |
| `vllm-only.diff` | Upstream [PR #40914](https://github.com/vllm-project/vllm/pull/40914) (open) — TurboQuant K+1 spec-verify routing. |
| `fix_spec_output.py` | **Makes #40914 actually work on Blackwell.** It returned the kernel tensor instead of writing the out-param buffer; under full CUDA-graph capture the return value is discarded → constant-token garbage. |
| `tq_auto_fallback.py` | MTP draft runner never inherits `cache_config.cache_dtype` (arrives `"auto"`, crashes). Falls back to `$VLLM_TQ_PRESET`. |
| `tq_splits.py` | Exposes TurboQuant's fixed decode KV-split count as `$VLLM_TQ_KV_SPLITS`. Default 32 is optimal — this exists so you can verify that, not change it. |
| [PR #44993](https://github.com/vllm-project/vllm/pull/44993) graft (`v1/structured_output/__init__.py`, `v1/core/sched/scheduler.py`) | **Structured output that survives thinking.** With a reasoning model, `response_format` json_schema + thinking-on returned **empty `content`** — the schema JSON leaked into `reasoning_content`. `should_advance`'s delta window (`num_computed_tokens − num_output_placeholders`) skips `</think>` when MTP rejects drafts, so the grammar never re-engages. Open, approved; stacked on merged #44297. Two pure-Python files. Needs the launch flag `--structured-outputs-config '{"backend":"xgrammar","reasoning_parser":"qwen3","enable_in_reasoning":false}'`. Lifted tool-eval **85→89**. |

**After any vLLM nightly bump**: re-verify. These patch a file (`vllm/v1/attention/backends/turboquant_attn.py`)
that upstream actively moves. A failed `patch` fails the build loudly; a *shifted* apply could
silently produce garbage. Test with raw `/v1/completions` — the chat parser hides degeneration
as empty content.

## LMCache format-10 kernel patch (separate project)

> **⚠️ WITHDRAWN (2026-07-18) — this patch restores corrupted content on hybrid models. Do not use it.** It was aimed at the wrong layer: the fault was never the transfer kernel, it was LMCache's kernel-page→logical-page *metadata* regrouping for vLLM's fused fp8 layout. **[`lmcache/0001` + `0002`](lmcache/README.md) are the real fix**, and the profile they enable is the current daily. Kept for historical reference only.

`lmcache-0.5.1-format10-NL_X_NB_NH_BS_TWO_HS.patch` is **not** for vLLM — it patches [LMCache](https://github.com/LMCache/LMCache) `csrc/mp_mem_kernels.cu` (a CUDA recompile, unlike the pure-Python patches above). It targeted the [MTP + LMCache](../docs/LMCACHE.md) profile **on the nightly pairing**.

| file | purpose |
|---|---|
| `lmcache-0.5.1-format10-NL_X_NB_NH_BS_TWO_HS.patch` | **Withdrawn.** Adds a transfer kernel for `EngineKVFormat 10` (`NL_X_NB_NH_BS_TWO_HS`) — the fused rank-4 `[NB, NH, BS, 2·HS]` KV layout vLLM's unified hybrid allocator emits, which lmcache 0.5.1 aborts on (`Unsupported EngineKVFormat: 10`, every store). The kernel *transfers* cleanly (0 format errors, SSD tier fills, post-restart 60K reload in 3.4s vs 23s re-prefill) but restores corrupted context. (Root cause, established after withdrawal: not the kernel at all — LMCache's kernel-page→logical-page regrouping doesn't cover vLLM's fused rank-4 fp8 layout, where each hybrid page registers as ~100 contiguous 16-token kernel pages; LMCache misclassifies the ratio as *compression* and transfers one 16-token page per logical block, wrongly addressed. GDN state was actually fine — a kernel-format patch below wrong metadata never had a chance.) Measured damage: a 60K-context needle vanishes after an L2 reload, and in-session L1 hits dropped a 69-scenario tool-eval from 88 to 47/100 (trial 2, hitting trial 1's cached chunks, collapsed 100→65 of 138 points). Coherent-but-amnesiac output with zero errors logged — the worst failure mode. Our earlier "validated: 12/12 hits, c1 122 / c8 458" note checked hit *mechanics* and throughput, not distant-fact recall — that's what missed it. LMCache `main` implements this format natively (spec + detector + kernels + tests, incl. opaque state pages); build against that instead. Lesson for any external-KV integration on hybrids: **needle-test across an eviction/restart boundary before trusting it** — coherence checks and hit counters will lie to you. |
