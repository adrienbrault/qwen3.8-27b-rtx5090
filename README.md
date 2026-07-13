# Qwen3.6-27B on a single RTX 5090 — 240K context, 4-bit KV, speculative decoding

Serving **Qwen3.6-27B with a 245K-token KV cache, MTP speculative decoding, and vision** on one 32 GB consumer GPU (RTX 5090, Blackwell `sm_120`).

The interesting part: this configuration **does not work on stock vLLM**. TurboQuant 4-bit KV cache and MTP speculative decoding produce garbage output together — a known, unfixed upstream bug ([vllm#40880](https://github.com/vllm-project/vllm/issues/40880), tracked as unsupported in [#40069](https://github.com/vllm-project/vllm/issues/40069)). The open PR that claims to fix it ([#40914](https://github.com/vllm-project/vllm/pull/40914)) **does not work on Blackwell** — it has a bug of its own ([the full story](#the-bug-nobody-caught)).

This repo contains the patches that make it work, and the benchmarks proving it does.

```
        context      decode c1    decode c4    prefill @4K
fp8      172K          129 t/s      492 t/s      9,607 t/s
TQ 4bit  261K  (+52%)  143 t/s      552 t/s     10,222 t/s     ← this setup
```
*Pools measured at `--gpu-memory-utilization 0.95`; the shipped config runs 0.94 → **245K pool** — 0.95 dies under concurrent cold-prompt bursts (see [CONFIG.md](docs/CONFIG.md)).*

---

## Quick start

```bash
# 1a. pull the prebuilt image…
docker pull ghcr.io/adrienbrault/vllm-turboquant:2026-07-13

# 1b. …or build it yourself (~1 min — pure-Python patches, no CUDA recompile)
cd patches && docker build -t vllm-turboquant:patched .

# 2. serve
./scripts/serve.sh
```

The `:2026-07-13` tag is the **exact image behind every benchmark number in this repo** — the
patches target a moving `vllm-openai:nightly`, so the pinned digest is the reproducible path
(`:patched` floats with rebuilds). It's ~28 GB; that's the vLLM base, [not the patches](#whats-in-the-patch-stack).

Then `http://localhost:8020/v1` speaks OpenAI. See [`docs/CONFIG.md`](docs/CONFIG.md) for every flag and why.

## What's in the patch stack

| patch | what it does |
|---|---|
| [`vllm-only.diff`](patches/vllm-only.diff) | Upstream [PR #40914](https://github.com/vllm-project/vllm/pull/40914) (open, unmerged): routes K+1 spec-verify batches through the TurboQuant decode kernel instead of the continuation-prefill path, which was attending only to just-drafted tokens and ignoring cached KV. |
| [`fix_spec_output.py`](patches/fix_spec_output.py) | **The fix that makes #40914 actually work on Blackwell.** Honors the out-param contract ([details below](#the-bug-nobody-caught)). Without this you get `!!!!!!!`. |
| [`tq_auto_fallback.py`](patches/tq_auto_fallback.py) | Second upstream gap: the MTP draft runner never inherits `cache_config.cache_dtype`, so TurboQuant layers on the draft path arrive with `"auto"` and crash. Falls back to `$VLLM_TQ_PRESET`. |
| [`tq_splits.py`](patches/tq_splits.py) | Makes TurboQuant's fixed decode KV-split count runtime-tunable (`$VLLM_TQ_KV_SPLITS`). *Tested: leave it at the default 32 — lowering it hurts both single-stream and batched.* |

## Benchmarks

Hardware: RTX 5090 32 GB (`sm_120`) + Ryzen 9 5900X + 64 GB RAM, Ubuntu 24.04.
Model: [`unsloth/Qwen3.6-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4) (4-bit weights) + `turboquant_4bit_nc` KV + MTP `ns=3` + vision.

### Throughput ([llama-benchy](https://github.com/eugr/llama-benchy), decode t/s)

| KV cache | ctx | c1 | c2 | c4 | c8 | prefill @4K |
|---|---|---|---|---|---|---|
| fp8_e4m3 | 172K | 129 | 253 | 492 | **868** | 9,607 |
| **turboquant_4bit_nc** | **261K** | **143** | 251 | **552** | 540 | **10,222** |

TurboQuant wins to c4 and **plateaus** past it — its Triton kernel does 4-bit dequant as ALU work that flash-attn's fp8 path gets free in hardware. For single-user coding, it's strictly better. Keep fp8 for 8+ concurrent (benchmark runs, multi-client).

### Quality

| eval | score | notes |
|---|---|---|
| **Aider polyglot** (225 exercises, diff) | **72.3%** pass@2 | 97.3% well-formed edits — reliably emits machine-applicable diffs |
| **Terminal-Bench 2.1** (8-task subset ×2) | **7/8** pass@2 | matches the fp8 baseline — 4-bit KV costs no measurable quality |
| **[tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) v2.1.0** (84 scenarios, hardmode, 4 trials) | **89.0 ± 0.0** /100 | Hard Mode 80%; deterministic across trials. On the v2.0.6 protocol: **90.0** vs [published](https://github.com/MiaAI-Lab/Unsloth-Qwen3.6-27B-UD-Q8_K_XL_vs_nvidia-Qwen3.6-27B-NVFP4_tools_eval) nvidia NVFP4 **89** / Unsloth Q8 **83** — details in [bench/RESULTS.md](bench/RESULTS.md) |

MTP acceptance: **75.8%** (per-position 0.945 / 0.764 / 0.564 — a healthy decay curve). Needle-in-haystack at 10K: recalled.

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

Every other branch of `forward()` writes the buffer. That one didn't. The fix ([`patches/fix_spec_output.py`](patches/fix_spec_output.py)):

```python
if output.ndim == 3:
    output[:N] = attn_out.to(output.dtype)
else:
    output[:N] = attn_out.reshape(N, -1).to(output.dtype)
return output
```

This is almost certainly why the PR passed on the author's Ampere box (the eager/piecewise path *does* consume the return value) and fails on Blackwell, where full CUDA-graph capture is the default.

## Things that DON'T work (so you don't repeat them)

| tried | verdict |
|---|---|
| **DFlash** speculative decoding | **Works, but boxed in — measured and rejected.** Requires a full source build of [PR #40898](https://github.com/vllm-project/vllm/pull/40898) (SWA draft support). Real result with the [z-lab draft](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash): **185 t/s single-stream (2.0× its no-spec baseline — a bigger uplift than MTP's ~1.6×)**… at the cost of **21K max context** (3.3 GB draft + bf16-KV-only: fp8 trips the branch's hybrid page-size assert, and the branch predates NVFP4-`lm_head` support so quantized targets are limited), **zero batch scaling** (c4 aggregate ≈ c1), and ~3× slower prefill. MTP's draft head lives *inside* the weights, costs ~0, and keeps 245K context. Revisit if #40898 merges into nightly. |
| `nvidia/Qwen3.6-27B-NVFP4` (official) | ~2.6× slower prefill, 20–25% slower decode, fatter checkpoint (max ~150K ctx), MTP crashes at moderate batch. Community quants win. |
| `--async-scheduling` | c4 552 → 526. No. |
| [froggeric fixed chat templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) | Bundled template already scores **4/4** on a behavioural probe (single + parallel tool calls, chat→tool→chat, chat-after-tools). Zero measured gain. |
| `--max-num-batched-tokens 4096` | **~4× prefill regression** (9.6K → 2.6K t/s) for +28K ctx. Keep 8192. |
| `VLLM_TQ_KV_SPLITS` < 32 | Hurts *both* c1 (139→132) and c8. Not the batching bottleneck. |
| `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass` | Already the nightly default. No-op. |

## ⚠️ Gotchas that will waste your day

1. **vLLM's prefix-cache metric lies on this model.** `vllm:prefix_cache_hits_total` and the "Prefix cache hit rate: 0.0%" log line report **0% while the cache is working**. Verified by timing: repeated 9,827-token prompt → **1.16s cold, 0.23s warm (5×)**. Don't debug the counter — time a repeated prompt ([`bench/prefix_probe.py`](bench/prefix_probe.py)).
2. **Always validate coherence via raw `/v1/completions`.** The chat endpoint's reasoning parser swallows degenerate output as *empty content*, so a broken model looks "fine but quiet". Degeneration tells: constant-token output, or **flat 100% MTP acceptance** (means draft and verify are locked in step).
3. **Re-verify after every vLLM bump.** These patches are version-sensitive; a nightly that moves `turboquant_attn.py` will silently fail to apply or, worse, apply to shifted code.
4. **`--kv-cache-memory` "fully utilize" hint OOMs at 240K+** — it ignores warmup transients. Use `--gpu-memory-utilization` and let vLLM profile.
5. **util 0.95 crashes under concurrent cold starts.** A burst of ~8 simultaneous fresh prompts OOMs the GDN prefill kernel (~96 MiB transient) and **kills the engine** — `expandable_segments` doesn't save it, and a ramping benchmark (llama-benchy) never trips it, so you find out in production. Run **0.94** (245K pool, burst-verified) unless you're strictly single-client.

## License

MIT (see [LICENSE](LICENSE)) for the original work here — docs, benchmarks, scripts, and the fixes.

`patches/vllm-only.diff` is redistributed verbatim from [vllm#40914](https://github.com/vllm-project/vllm/pull/40914)
by @Sandermage and stays under **Apache-2.0**, as do the vLLM files the patches modify. See
[THIRD_PARTY.md](THIRD_PARTY.md).

## Credits

- [vLLM](https://github.com/vllm-project/vllm) and PR [#40914](https://github.com/vllm-project/vllm/pull/40914) by @Sandermage — the foundation this builds on.
- [Unsloth](https://huggingface.co/unsloth) for the NVFP4 quant that beat every alternative tested.
- [llama-benchy](https://github.com/eugr/llama-benchy), [Terminal-Bench / Harbor](https://www.tbench.ai/), [aider](https://github.com/Aider-AI/aider) for the measurements.
