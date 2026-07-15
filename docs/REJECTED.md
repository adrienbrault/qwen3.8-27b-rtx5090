# Things that DON'T work (so you don't repeat them)

Every row below was tried on this exact box and rejected, with the number that killed it. Keep this list handy before you "improve" the config — most of these look like obvious wins and cost a day each to disprove. (Benchmark-only rejects with their throughput deltas also live in [`../bench/RESULTS.md`](../bench/RESULTS.md#rejected-with-numbers-so-nobody-redoes-them).)

| tried | verdict |
|---|---|
| **`turboquant_4bit_nc`** (4-bit Keys) | **Rejected — the old headline config.** 4-bit keys destroy long-context retrieval: **0/8** on fair needle-in-haystack (vs k8v4's 8/8). Its ~47.8K tok/GiB density is real but worthless if the model can't find what it stored. `turboquant_k8v4` (8-bit K / 4-bit V) is the fix. |
| **`nvfp4` native KV cache** | **Won't load.** `head_size=256` has no FlashInfer nvfp4 backend on `sm_120` — needs unmerged [vLLM PR #44389](https://github.com/vllm-project/vllm/pull/44389). Projected density ≈47–48K tok/GiB, **unmeasured**. |
| **DFlash** speculative decoding | **Works, but boxed in — measured and rejected.** Requires a full source build of [PR #40898](https://github.com/vllm-project/vllm/pull/40898) (SWA draft support). Real result with the [z-lab draft](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash): **185 t/s single-stream (2.0× its no-spec baseline — a bigger uplift than MTP's ~1.6×)**… at the cost of **21K max context** (3.3 GB draft + bf16-KV-only: fp8 trips the branch's hybrid page-size assert, and the branch predates NVFP4-`lm_head` support so quantized targets are limited), **zero batch scaling** (c4 aggregate ≈ c1), and ~3× slower prefill. MTP's draft head lives *inside* the weights, costs ~0, and keeps 245K context. Revisit if #40898 merges into nightly. |
| `nvidia/Qwen3.6-27B-NVFP4` (official) | ~2.6× slower prefill, 20–25% slower decode, fatter checkpoint (max ~150K ctx), MTP crashes at moderate batch. Community quants win. |
| `--async-scheduling` | c4 552 → 526. No. |
| [froggeric fixed chat templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) | Bundled template already scores **4/4** on a behavioural probe (single + parallel tool calls, chat→tool→chat, chat-after-tools). Zero measured gain. |
| `--max-num-batched-tokens 4096` | **~4× prefill regression** (9.6K → 2.6K t/s) for +28K ctx. Keep 8192. |
| `VLLM_TQ_KV_SPLITS` < 32 | Hurts *both* c1 (139→132) and c8. Not the batching bottleneck. |
| `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass` | Already the nightly default. No-op. |
| **LMCache + `turboquant_k8v4`** (persisted tier) | **Composes, but the persisted L2 tier is lossy — not shipped.** k8v4 KV builds and runs under LMCache and stores land, but the L2 SSD reload silently drops long-context needles (7/7 miss) — the format-10 kernel copies the standard 512-wide layout, not k8v4's 262-wide packed one. LMCache persistence only round-trips faithfully with fp8 KV. Full write-up: [`LMCACHE.md`](LMCACHE.md#lmcache--k8v4-composes-but-the-persisted-tier-is-lossy--not-shipped). |
