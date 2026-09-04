# Qwen3.8-27B on RTX 5090

Serving configuration, vLLM patches, launch scripts and measurements for running [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) on one or two RTX 5090 cards with 262K context. The target workload is a few concurrent coding agents with long contexts, plus interactive chat with vision, reasoning, tool calling and structured output all enabled.

Every number in this repo was measured on one machine, on the date given, and the raw results directory is named next to it. Nothing here is a projection.

## Numbers

The served configuration since 2026-09-04: two RTX 5090, [RedHatAI/Qwen3.8-27B-NVFP4](https://huggingface.co/RedHatAI/Qwen3.8-27B-NVFP4) weights, [vLLM](https://github.com/vllm-project/vllm) 0.29 with an NVFP4 KV cache, the [DFlash2](https://inco.ai/blog/dflash2/) drafter in CUDA graphs, launcher [scripts/serve-r168-daily.sh](scripts/serve-r168-daily.sh). Each row links to its write-up in [bench/RESULTS.md](bench/RESULTS.md), which names the raw results directory on the serving host, and to the driver script that produced it.

| | value | measured |
|---|---|---|
| context length | 262,144 tokens | |
| KV pool on the GPUs | 1,020,596 tokens, pinned at 13.98 GB per card, 16 sequences. A short request holds 3.4% of it, a 100K prompt 15.9%; five 100K contexts ran together at 79% | 2026-09-04, [R182](bench/RESULTS.md#r182-the-gdn-state-cached-in-bf16-promoted-pool-1020596-tiers-needles-tool-eval-2026-09-04-results2026-09-04-r182-promote-ssm-bf16-scriptsr182-promote-ssm-bf16sh), [driver](scripts/r182-promote-ssm-bf16.sh) |
| KV tiers behind the pool | 16 GiB host RAM, then 300 GB of disk with LRU eviction, kept across restarts. A 131K or 220K prompt evicted from the GPUs is served again from the tiers in 1.4 to 2.7 s, against 25 to 56 s cold | 2026-09-04, [R182](bench/RESULTS.md#r182-the-gdn-state-cached-in-bf16-promoted-pool-1020596-tiers-needles-tool-eval-2026-09-04-results2026-09-04-r182-promote-ssm-bf16-scriptsr182-promote-ssm-bf16sh), [driver](scripts/r182-promote-ssm-bf16.sh) |
| decode, 1 stream | code 285 t/s, prose 158 t/s, prose at 30K context 153 t/s. The rate follows draft acceptance: 0.36 accepted per draft token on code, 0.15 on prose | 2026-09-04, [R183](bench/RESULTS.md#r183-a-decode-step-profile-of-the-served-route-and-a-20-arm-lever-ladder-gemm-kernel-all-reduce-backends-fusion-passes-prefill-chunk-speculation-policy-dp2-the-two-card-decode-tax-is-the-custom-all-reduce-not-nccl-and-no-arm-beats-the-replicate-band-2026-09-04-results2026-09-04-r183-next-levers-scriptsr183-next-leverssh), [driver](scripts/r183-next-levers.sh) (BASE arm) |
| decode, 8 streams | 1,327 and 1,415 t/s aggregate (two boots, two runs each) | 2026-09-04, [R183](bench/RESULTS.md#r183-a-decode-step-profile-of-the-served-route-and-a-20-arm-lever-ladder-gemm-kernel-all-reduce-backends-fusion-passes-prefill-chunk-speculation-policy-dp2-the-two-card-decode-tax-is-the-custom-all-reduce-not-nccl-and-no-arm-beats-the-replicate-band-2026-09-04-results2026-09-04-r183-next-levers-scriptsr183-next-leverssh), [driver](scripts/r183-next-levers.sh); [R182](bench/RESULTS.md#r182-the-gdn-state-cached-in-bf16-promoted-pool-1020596-tiers-needles-tool-eval-2026-09-04-results2026-09-04-r182-promote-ssm-bf16-scriptsr182-promote-ssm-bf16sh), [driver](scripts/r182-promote-ssm-bf16.sh) |
| decode, 16 streams | 1,738 t/s aggregate, 109 per stream | 2026-09-04, [R183](bench/RESULTS.md#r183-a-decode-step-profile-of-the-served-route-and-a-20-arm-lever-ladder-gemm-kernel-all-reduce-backends-fusion-passes-prefill-chunk-speculation-policy-dp2-the-two-card-decode-tax-is-the-custom-all-reduce-not-nccl-and-no-arm-beats-the-replicate-band-2026-09-04-results2026-09-04-r183-next-levers-scriptsr183-next-leverssh), [driver](scripts/r183-next-levers.sh) |
| prefill, cold prompt, 1 request | 2K-token prompt 8.8K t/s ([llama-benchy](https://github.com/eugr/llama-benchy)); 6.7K 8.3K; 30K 7.9K; 100K 5.8K; 200K 4.2K. First token after 0.8 s at 6.7K, 3.8 s at 30K, 17.1 s at 100K, 47.7 s at 200K | 2026-09-04, [R183](bench/RESULTS.md#r183-a-decode-step-profile-of-the-served-route-and-a-20-arm-lever-ladder-gemm-kernel-all-reduce-backends-fusion-passes-prefill-chunk-speculation-policy-dp2-the-two-card-decode-tax-is-the-custom-all-reduce-not-nccl-and-no-arm-beats-the-replicate-band-2026-09-04-results2026-09-04-r183-next-levers-scriptsr183-next-leverssh), [driver](scripts/r183-next-levers.sh) (request latency at 8 output tokens); 2K on llama-benchy, fp32 state: `results/2026-09-04-r173-c1-opt` |
| [SWE-Bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified), [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent) 2.4.6, one attempt, [official harness](https://github.com/SWE-bench/SWE-bench) | 388/500 = 77.6% (the fp8-KV shape the day before: 386/500, and paired per instance the two score the same) | 2026-09-04, [R175](bench/RESULTS.md#r175-swe-bench-verified-on-the-served-route-388500--776-paired-with-the-fp8-shape-2026-09-04-results2026-09-02-miniswe-rh-r174-nvfp4-scriptsminiswe-fullsh), [driver](scripts/miniswe-full.sh), [package](bench/README.md); fp32 state, see below |
| [tool-eval](https://github.com/SeraphimSerapis/tool-eval-bench), 69 scenarios × 4 trials | 90.5 ± 2.1 | 2026-09-04, [R182](bench/RESULTS.md#r182-the-gdn-state-cached-in-bf16-promoted-pool-1020596-tiers-needles-tool-eval-2026-09-04-results2026-09-04-r182-promote-ssm-bf16-scriptsr182-promote-ssm-bf16sh), [driver](scripts/r182-promote-ssm-bf16.sh) |
| GSM8K cot zero-shot ([lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)), n=120, temperature 0 | 0.85 ± 0.03 | 2026-09-04, [R177](bench/RESULTS.md#r177-the-served-route-at-16-sequences-on-the-r142-matrix-instrument-2026-09-04-results2026-09-04-r177-matrix-scriptsr177-matrixsh), [driver](scripts/r177-matrix.sh); fp32 state, see below |
| fidelity vs the [bf16 model](https://huggingface.co/Qwen/Qwen3.8-27B), dense text, 555,549 positions | top-1 agreement 92.77%, perplexity +0.74%, truncated KL 0.0141 | 2026-09-04, [R180](bench/RESULTS.md#r179-to-r181-pool-cost-knobs-the-bf16-gdn-state-halves-the-per-request-cost-its-fidelity-on-the-dense-agentic-and-decode-rulers-2026-09-04-results2026-09-04-r180-pool-cost-clean-results2026-09-04-r181-ssm-bf16-ruler-scriptsr180-pool-cost-cleansh-scriptsr181-ssm-bf16-rulersh), [driver](scripts/r180-pool-cost-clean.sh); [R183](bench/RESULTS.md#r183-a-decode-step-profile-of-the-served-route-and-a-20-arm-lever-ladder-gemm-kernel-all-reduce-backends-fusion-passes-prefill-chunk-speculation-policy-dp2-the-two-card-decode-tax-is-the-custom-all-reduce-not-nccl-and-no-arm-beats-the-replicate-band-2026-09-04-results2026-09-04-r183-next-levers-scriptsr183-next-leverssh), [driver](scripts/r183-next-levers.sh); [docs/FIDELITY.md](docs/FIDELITY.md) |
| fidelity vs the bf16 model, agentic turns, 57,972 positions | top-1 agreement 95.58%, perplexity +2.68% | 2026-09-04, [R180](bench/RESULTS.md#r179-to-r181-pool-cost-knobs-the-bf16-gdn-state-halves-the-per-request-cost-its-fidelity-on-the-dense-agentic-and-decode-rulers-2026-09-04-results2026-09-04-r180-pool-cost-clean-results2026-09-04-r181-ssm-bf16-ruler-scriptsr180-pool-cost-cleansh-scriptsr181-ssm-bf16-rulersh), [driver](scripts/r180-pool-cost-clean.sh); [docs/FIDELITY.md](docs/FIDELITY.md) |

Decode numbers are steady-state tokens per second from [scripts/decode_ss.py](scripts/decode_ss.py) (512 generated tokens through R182, 1,024 from R183 on; content stated in the results directory), not [llama-benchy](https://github.com/eugr/llama-benchy) means. Single-stream decode swings up to 20% between runs on this box; the 8-stream cell shows the spread between two boots. The `r183` BASE boot carried an idle torch-profiler configuration, and its 8-stream number is the lower of the two.

Footnote on the state precision: the linear-attention state has been cached in bf16 since 2026-09-04 16:24 UTC ([R182](bench/RESULTS.md#r182-the-gdn-state-cached-in-bf16-promoted-pool-1020596-tiers-needles-tool-eval-2026-09-04-results2026-09-04-r182-promote-ssm-bf16-scriptsr182-promote-ssm-bf16sh), [docs/HISTORY.md](docs/HISTORY.md)); it halves what a request costs in the pool. SWE-Bench, GSM8K and the 2K prefill were scored before that change, on the same weights, KV dtype and drafter with the fp32 state (pool 903,793). The bf16-state SWE-Bench run is in progress ([scripts/miniswe-full.sh](scripts/miniswe-full.sh), reproduction package in [bench/](bench/README.md)). The rulers put the two states 0.05 points of top-1 agreement apart on dense text and 0.1 on agentic turns ([docs/FIDELITY.md](docs/FIDELITY.md)).

The fidelity rows are what the whole stack costs against the unquantized model: a different NVFP4 checkpoint of the same model ([gittensor](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090-LMHead4)), served here until 2026-09-02, sat 4.5 points of top-1 agreement lower, and no task benchmark in this repo detected it. [docs/FIDELITY.md](docs/FIDELITY.md) has the rulers for every checkpoint and KV dtype.

## What the stack is

- **Weights**: [RedHatAI NVFP4](https://huggingface.co/RedHatAI/Qwen3.8-27B-NVFP4), W4A4 from [llm-compressor](https://github.com/vllm-project/llm-compressor) with 303 modules kept at 8 bit and an fp8 `lm_head`. Chosen by a bf16-anchored fidelity ladder over nine NVFP4 checkpoints ([docs/FIDELITY.md](docs/FIDELITY.md), [docs/R156-DECISION.md](docs/R156-DECISION.md)): +0.38% perplexity from bf16 on dense text; the quantizer recipe mattered more than the bit width.
- **Engine**: [vLLM v0.29.0rc2](https://github.com/vllm-project/vllm/releases) with the [patches-v0290/](patches-v0290/) chain (0101 to 0137): NVFP4 KV on `sm_120`, which upstream gates to SM100; DFlash2 with quantized drafters in CUDA graphs; a pooled FlashInfer workspace; prefix-cache reuse under DFlash; the embedding table in pinned host RAM; LRU eviction for the disk tier, which upstream lacks. [FlashInfer](https://github.com/flashinfer-ai/flashinfer) is pinned at 0.6.16.post3, because 0.6.18 drops decode at 30K context from 143 to 26.5 t/s ([scripts/r168-deep-decode.sh](scripts/r168-deep-decode.sh)). Built by [scripts/build-v0290rc2.sh](scripts/build-v0290rc2.sh); each patch has a design note next to its diff and a provenance line in [THIRD_PARTY.md](THIRD_PARTY.md).
- **Drafter**: [syvai/Qwen3.8-27B-DFlash2-W4A16](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16), 9 draft tokens, tensor-parallel 2, in CUDA graphs ([DFlash2](https://inco.ai/blog/dflash2/), [vLLM PR #52816](https://github.com/vllm-project/vllm/pull/52816)). 7 draft tokens gave +12.5% at 8 streams but sat twice as far from the bf16 decode reference ([scripts/r173c-bf16-decode.sh](scripts/r173c-bf16-decode.sh)), so 9 stayed.
- **KV cache**: NVFP4 KV (+43% pool over fp8 at the same VRAM, for 0.4 points of perplexity; the sm120 port is in [patches-v0290/](patches-v0290/), provenance in [THIRD_PARTY.md](THIRD_PARTY.md)); the pool pinned in bytes so every boot has the same size ([R178](bench/RESULTS.md#r178-the-concurrency-ceiling-is-the-kv-pool-what-a-request-costs-of-it-2026-09-04-results2026-09-04-r178-seqs-ladder-scriptsr178-seqs-laddersh-scriptskvcapacityprobepy)); the linear-attention state in bf16 ([R182](bench/RESULTS.md#r182-the-gdn-state-cached-in-bf16-promoted-pool-1020596-tiers-needles-tool-eval-2026-09-04-results2026-09-04-r182-promote-ssm-bf16-scriptsr182-promote-ssm-bf16sh)); a 16 GiB CPU tier and a 300 GB disk tier ([scripts/setup-native-l2.sh](scripts/setup-native-l2.sh), [scripts/tier-evict.sh](scripts/tier-evict.sh)).
- **Guard rails**: the launcher asserts the image, the vLLM and FlashInfer versions, the store overlay, the drafter graphs, the pool size, free VRAM after pre-warm and the tier state, and refuses to serve otherwise. [docs/CONFIG.md](docs/CONFIG.md) explains every flag and what breaks without it.

## Hardware

- ASRock X870 Taichi Creator, Ryzen 7 9800X3D, 64 GB DDR5-6000, Ubuntu 24.04 HWE.
- Two RTX 5090 32 GB (`sm_120`): ASUS at 600 W and HP OEM at 575 W, PCIe Gen5 x8/x8.
- NVIDIA driver 610.57.04 with the [QuixiAI open kernel modules](https://github.com/QuixiAI/open-gpu-kernel-modules) for GPU peer-to-peer ([scripts/gpu-p2p-610.sh](scripts/gpu-p2p-610.sh)).
- Memory clock offset +4500 MHz on both cards, core clock stock ([scripts/gpu-tune.sh](scripts/gpu-tune.sh)), worth about 4% decode. All throughput numbers include it.
- One Gen5 x4 NVMe for the model weights and a 393 GB loopback image for the KV disk tier.

The one-card configuration ran on this same host before the second card was added. The host RAM it needs was not measured: its container is capped at 52 GB (`--memory`) with a 4 GiB CPU KV staging buffer inside that, and the peak is the first boot's kernel JIT, which once took all 64 GB before the compile-job caps and persisted caches bounded it ([docs/CONFIG.md](docs/CONFIG.md)).

## Quick start

The scripts assume the host layout used here: models under `/srv/qwen5090/models`, compile caches under `/srv/qwen5090/cache`, the disk tier at `/srv/qwen5090/native-l2`, [scripts/serve-v0280-daily.sh](scripts/serve-v0280-daily.sh) installed as `/srv/qwen5090/launch-daily-v0280.sh` and [scripts/serve-r156-daily.sh](scripts/serve-r156-daily.sh) as `/srv/qwen5090/launch-daily-redhat-fp8-0902.sh`. Adjust the paths at the top of each script if your layout differs.

```bash
# 1. weights (22 GB on disk) and the DFlash2 drafter (1.2 GB, two-card configurations only)
huggingface-cli download RedHatAI/Qwen3.8-27B-NVFP4 --local-dir /srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4
huggingface-cli download syvai/Qwen3.8-27B-DFlash2-W4A16 --local-dir /srv/qwen5090/models/dflash2-qwen38-syvai-w4a16

# 2. the disk KV tier: a fixed-size loopback filesystem, so the cache cannot fill the root disk.
#    The script makes 200 GB; the served launcher's 300 GB cap only engages on a larger image,
#    so either raise SIZE in the script or pass TIER_CAP_GB below the image size.
sudo bash scripts/setup-native-l2.sh

# 3a. the served image: vLLM v0.29.0rc2 + patches-v0290 + FlashInfer 0.6.16.post3, from a pinned vLLM nightly base
bash scripts/build-v0290rc2.sh          # CPU only; about an hour
# 3b. the v0.28.0 image for the rollback and the one-card shapes
docker build -f patches-v0280/Dockerfile.v0280-nvfp4kv -t vllm-qwen38:v0280-nvfp4kv patches-v0280

# 4a. two cards, the served configuration
bash scripts/serve-r168-daily.sh
# 4b. two cards, the fp8 rollback
bash scripts/serve-r156-daily.sh
# 4c. one card
MODEL_DIR=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4 PORT=8020 NAME=vllm-27b bash scripts/serve-v0280-daily.sh
```

The endpoint is OpenAI-compatible at `http://<host>:8020/v1`, model name `qwen3.8-27b` (alias `qwen3.6-27b`):

```bash
curl -s http://localhost:8020/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "Write a Python function that parses ISO-8601 durations."}]
}' | jq -r '.choices[0].message.content'
```

Reasoning is on by default at effort `medium`. Tool calls, JSON-schema structured output and up to 16 images per request work without extra flags. Clients that resend prior assistant turns should include the `reasoning` field to keep earlier thinking blocks in context.

## Other configurations

Three shapes remain runnable and documented:

- **Two cards, fp8 KV, DFlash2 on vLLM 0.28.0** ([scripts/serve-r156-daily.sh](scripts/serve-r156-daily.sh)): the rollback. Served 2026-09-02 to 09-04 on the same checkpoint with pool 654,491, tool-eval 90.8 ± 0.5 over its life, SWE-Bench Verified 386/500. Its disk tier never served a revisit, because a tier hit must fit the CPU tier whole (`results/2026-09-04-r172-cputier`).
- **One card, nvfp4 KV, MTP on vLLM 0.28.0** ([scripts/serve-v0280-daily.sh](scripts/serve-v0280-daily.sh)): the shape for a single RTX 5090.
- **Two cards, nvfp4 KV, MTP on vLLM 0.28.0** (`serve-v0280-daily.sh` with `TP=2`): the capacity shape, 1,508,519 tokens of pool and 2,007 t/s aggregate at 16 streams, at 225 t/s single stream.

The three measured on 2026-08-31 on the [gittensor](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090-LMHead4) checkpoint, same day, same harness, `results/2026-08-31-r142-matrix`. The RedHatAI checkpoint costs about 6% decode, 14% prefill and 12% pool on the fp8 shape relative to these numbers. The last column is the served route on 2026-09-04 on the RedHatAI checkpoint; cells marked † were read with the fp32 state (`results/2026-09-04-r177-matrix`), the rest with the bf16 state (`results/2026-09-04-r183-next-levers`, `results/2026-09-04-r182-promote-ssm-bf16`). Checkpoint and day differ from the other three columns.

| | one card | two cards, DFlash2, fp8 KV | two cards, MTP, nvfp4 KV | served: two cards, DFlash2, nvfp4 KV, vLLM 0.29 (2026-09-04) |
|---|---|---|---|---|
| KV pool at 262K | 381,300 | 746,849 | 1,508,519 | 1,020,596 |
| decode, code, 1 stream | 175.0 t/s | 298.9 | 225.3 | 285 |
| decode, code, 8 streams | 1,187 | 1,289 | 1,349 | 1,327 and 1,415 (two boots) |
| decode, code, 16 streams | not admitted | 1,522 | 2,007 | 1,738 |
| decode at 100K context | 106.7 | 174.4 | 137.5 | 152.8 † |
| prefill at 8K | 11.9K t/s | 9.3K | 9.0K | 8.1K † |
| prefill at 100K | 4.7K | 7.0K | 6.3K | 6.4K † |
| tool-eval ×4 | 89.2 ± 1.7 | 89.8 ± 1.3 | 90.2 ± 1.0 | 90.5 ± 2.1 |

DFlash2 accepts few draft tokens per step, so its decode is bound by weight bandwidth, which the second card doubles. MTP accepts more, amortizes the weight reads, and turns the second card into KV space instead. Tool-eval does not separate the three; the bf16 rulers do, by the KV dtype ([docs/FIDELITY.md](docs/FIDELITY.md)).

## Findings that transfer

- The NVFP4 store overlay ([patches-v0280/README-sm120-nvfp4.md](patches-v0280/README-sm120-nvfp4.md)) is required on `sm_120` and its absence is invisible to behavioural tests. Without it the engine is fluent, passes needle tests, and has 2.7 to 10 times the attention error. Only a numeric diagnostic catches it.
- Task benchmarks cannot rank quantized checkpoints ([docs/FIDELITY.md](docs/FIDELITY.md)). GSM8K at n=250 resolves about 8 percentage points; the nine checkpoints differ by less than one, and by 4.5 points of top-1 agreement against bf16.
- Prefill-only fidelity rulers cannot see decode kernels or the draft length. Greedy continuations with 7 and 9 draft tokens diverge on 19 of 20 chunks; validate a decode-path change with [scripts/decode_fidelity.py](scripts/decode_fidelity.py) against the bf16 decode reference ([docs/FIDELITY.md](docs/FIDELITY.md)).
- A FlashInfer bump can change deep-context decode 5x without touching short prompts. Measure decode at 30K context after every library change ([scripts/r168-deep-decode.sh](scripts/r168-deep-decode.sh)).
- A request costs more of the pool than its token count: the linear-attention state is paid per sequence, so the state dtype, not the attention block, sets the per-request floor ([docs/DESIGN.md](docs/DESIGN.md#what-a-request-costs-in-the-pool), [scripts/kv_capacity_probe.py](scripts/kv_capacity_probe.py)).
- On two cards, the decode-step cost that grows with concurrency is the custom all-reduce, from 15% of the step at 1 stream to 33% at 16; no NCCL all-reduce runs in decode (NCCL takes the prefill chunks above the 8 MiB custom-all-reduce cap), but about 3 NCCL all-gathers per step remain, 0.2 / 1.0 / 2.0 ms at 1 / 8 / 16 streams ([R183](bench/RESULTS.md#r183-a-decode-step-profile-of-the-served-route-and-a-20-arm-lever-ladder-gemm-kernel-all-reduce-backends-fusion-passes-prefill-chunk-speculation-policy-dp2-the-two-card-decode-tax-is-the-custom-all-reduce-not-nccl-and-no-arm-beats-the-replicate-band-2026-09-04-results2026-09-04-r183-next-levers-scriptsr183-next-leverssh), [scripts/prof_decode_split.py](scripts/prof_decode_split.py)). FlashInfer's `pcie_ipc` all-reduce (main only, [PR #4393](https://github.com/flashinfer-ai/flashinfer/pull/4393)) is 24% to 36% faster than the served kernel at the decode row counts on this box and every kernel hits the PCIe floor at 84 MB; the ceiling for a kernel swap is about 8% of the decode step at 8 and 16 streams ([R184](bench/RESULTS.md#r184-an-all-reduce-microbenchmark-at-the-served-decode-shapes-nccl-vs-vllms-custom-all-reduce-vs-flashinfers-pcie_ipc-kernel-on-the-two-rtx-5090s-pcie_ipc-is-36-faster-than-the-served-kernel-at-10-rows-and-24-at-160-the-served-kernel-is-slower-than-nccl-from-80-rows-up-and-all-three-sit-at-the-pcie-floor-at-80-mb-2026-09-04-results2026-09-04-r184-arbench-scriptsar_benchpy-scriptsar-benchsh), [scripts/ar_bench.py](scripts/ar_bench.py)).
- FlashInfer's `pcie_ipc` all-reduce, vendored as an opt-in layer (patch 0138, not served), buys 4.6% code and 5.4% prose single-stream decode, 3.1% at 30K context and 2.6% at 16 streams over the same image with the kernel off, with numerics identical on every paired ruler; the 8-stream tokens/s read is flat because that boot's draft acceptance was lower, and the step rate there is +4.9% (2026-09-04, [R185](bench/RESULTS.md#r185-flashinfers-pcie_ipc-all-reduce-vendored-into-the-served-image-as-an-opt-in-tp2-all-reduce-patch-0138-46-code-and-54-prose-single-stream-decode-31-at-30k-context-26-at-16-streams-the-8-stream-tokenss-flat-at-49-stepss-numerics-identical-to-the-kernel-off-control-2026-09-04-results2026-09-05-r185-pcieipc-scriptsr185-pcieipcsh-scriptsr185b-pcieoff-fidsh)).
- The NVFP4 GEMM kernel is a fidelity knob. The Marlin kernel (FP4 weights dequantized, bf16 activations) is +0.207% perplexity from bf16 where the served kernel is +0.744%, at a cost of 7.8% of 8-stream decode, 17.3% of 16-stream and 32% added to the time to first token at 100K (2026-09-04, [R183b](bench/RESULTS.md#r183b-the-nvfp4-gemm-kernel-ladder-walked-on-the-automatic-path-and-five-kernel-count-fusion-passes-the-marlin-kernel-is-0207-perplexity-from-bf16-against-the-served-kernels-0744-and-costs-78-of-8-stream-decode-173-of-16-stream-and-adds-32-to-the-time-to-first-token-at-100k-2026-09-04-results2026-09-04-r183-next-levers-scriptsr183b-kernelssh)).
- A disk-tier hit is served only if the whole prompt fits the CPU tier. Size the CPU tier for the longest prompt you expect to revisit, and test the tier with a needle retrieved after a restart ([scripts/needle_gate.sh](scripts/needle_gate.sh), [scripts/r172-cputier.sh](scripts/r172-cputier.sh)).
- Do not pass `--no-async-scheduling` on vLLM 0.28 or later. It costs 21% to 29% single-stream decode ([docs/REJECTED.md](docs/REJECTED.md)).
- Single-stream decode with speculative decoding varies between boots and between runs. Compare within one boot, or normalize by accepted tokens per step ([scripts/decode_ss.py](scripts/decode_ss.py) reports both).

The full list is in [docs/GOTCHAS.md](docs/GOTCHAS.md); the configurations tried and rejected, with the number that rejected them, in [docs/REJECTED.md](docs/REJECTED.md).

## Repository map

| path | contents |
|---|---|
| [scripts/](scripts/) | Launchers (`serve-*.sh`), image builds (`build-v0290rc*.sh`), host setup, measurement probes (`decode_ss.py`, `decode_fidelity.py`, `fidelity_compare.py`, `kv_capacity_probe.py`, `needle_gate.sh`) and one driver per results directory (`r1xx-*.sh`). |
| [patches-v0290/](patches-v0290/) | The served patch chain on vLLM v0.29.0rc2, its Dockerfiles, verification scripts and design notes. [patches-v0280/](patches-v0280/README-sm120-nvfp4.md) is the v0.28.0 generation, one README per hunk, which the rollback and the one-card shapes run. |
| [bench/RESULTS.md](bench/RESULTS.md) | Every measurement, newest first, with the SWE-Bench reproduction package in [bench/](bench/README.md). |
| [docs/CONFIG.md](docs/CONFIG.md) | Every flag of the served and the rollback configuration, why it is set, and what breaks without it. |
| [docs/FIDELITY.md](docs/FIDELITY.md) | The bf16 rulers: checkpoints, KV dtypes, state precision, draft length. |
| [docs/DESIGN.md](docs/DESIGN.md) | Why W4A4 weights, why this model fits, where the VRAM goes, what a request costs in the pool, why two cards help the way they do. |
| [docs/GOTCHAS.md](docs/GOTCHAS.md), [docs/REJECTED.md](docs/REJECTED.md) | Failure modes found on this stack, and configurations rejected with numbers. |
| [docs/HISTORY.md](docs/HISTORY.md) | Lineage of the served configuration since 2026-06, reversals included; [docs/R156-DECISION.md](docs/R156-DECISION.md) is the checkpoint decision record. |
| [THIRD_PARTY.md](THIRD_PARTY.md) | Per-file provenance of every patch and idea taken from upstream PRs and other repos. |

## License

MIT ([LICENSE](LICENSE)) for the original work: documentation, scripts, probes and patch tooling. Everything derived from vLLM, LMCache or FlashInfer, including redistributed PR diffs and patched files inside built images, stays Apache-2.0-derived. Per-file inventory: [THIRD_PARTY.md](THIRD_PARTY.md).

## Credits

- [vLLM](https://github.com/vllm-project/vllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [LMCache](https://github.com/LMCache/LMCache).
- ch2lab for [vLLM PR #49891](https://github.com/vllm-project/vllm/pull/49891), the sm120 NVFP4 KV routing.
- drowzeys for the linear V-scale writer fix ([DGX Spark repo](https://github.com/drowzeys/keys-vLLm.0.27-Qwen3.8-27B-ADay777Ablit-NVFP4-A4Q-NVFP4-KV-4M-KV-token-pool-MTP3-Single-DGX-Spark)).
- The author of [vllm#49011](https://github.com/vllm-project/vllm/issues/49011) for demonstrating XQA-NVFP4 decode with FA2 prefill on sm120, and [hikarioyama](https://github.com/hikarioyama/vllm-nvfp4-kv-sm120) for the FA2 SF-stride prior art.
- [seanyourhighness](https://github.com/seanyourhighness/vllm-sm12x-nvfp4-dflash2) for the DFlash2-on-NVFP4 overlay, the `--kv-cache-memory-bytes` pinning and the masked NVFP4 XQA verification of [vLLM PR #53543](https://github.com/vllm-project/vllm/pull/53543) (patch 0132).
- F21HGG for [vLLM PR #54181](https://github.com/vllm-project/vllm/pull/54181), the packed GDN decode launch (0133); Ledgero for [vLLM PR #54163](https://github.com/vllm-project/vllm/pull/54163), prefix-cache reuse under DFlash drafters (0134); waizuichougou for [vLLM PR #53981](https://github.com/vllm-project/vllm/pull/53981), the embedding-table UVA offload (0135).
- The v0.29 rebase of the chain and the disk-tier eviction patch (0137) were produced with OpenAI's codex from local source dumps; every build and measurement ran on the host.
- [RedHatAI](https://huggingface.co/RedHatAI/Qwen3.8-27B-NVFP4) for the served weights; [unsloth](https://huggingface.co/unsloth) and [kelnei](https://huggingface.co/kelnei) for the two checkpoints that tie it on fidelity; [sakamakismile](https://huggingface.co/sakamakismile/Qwen3.8-27B-MTP-NVFP4) and [gittensor](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090-LMHead4) for the earlier served checkpoints; [syv-ai](https://huggingface.co/syvai) for the quantized DFlash2 drafter; z-lab and inco.ai for DFlash2 and [vLLM PR #52816](https://github.com/vllm-project/vllm/pull/52816).

## Links

Models:

- [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), the bf16 model every ruler is measured against.
- [RedHatAI/Qwen3.8-27B-NVFP4](https://huggingface.co/RedHatAI/Qwen3.8-27B-NVFP4), the served weights; [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) and [kelnei/Qwen3.8-27B-NVFP4](https://huggingface.co/kelnei/Qwen3.8-27B-NVFP4) tie it on fidelity.
- [syvai/Qwen3.8-27B-DFlash2-W4A16](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16), the served drafter; [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2), the original bf16 drafter.
- [gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090-LMHead4](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090-LMHead4) and [sakamakismile/Qwen3.8-27B-MTP-NVFP4](https://huggingface.co/sakamakismile/Qwen3.8-27B-MTP-NVFP4), the checkpoints served before.

Software:

- [vLLM](https://github.com/vllm-project/vllm) and its [releases](https://github.com/vllm-project/vllm/releases); the patch chains here are [patches-v0290/](patches-v0290/) and [patches-v0280/](patches-v0280/README-sm120-nvfp4.md).
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer), pinned at 0.6.16.post3 in the served image.
- [llm-compressor](https://github.com/vllm-project/llm-compressor), the quantizer behind the served weights.
- [DFlash2](https://inco.ai/blog/dflash2/) and its [vLLM PR #52816](https://github.com/vllm-project/vllm/pull/52816).
- [QuixiAI open GPU kernel modules](https://github.com/QuixiAI/open-gpu-kernel-modules), for peer-to-peer between the two cards.
- [LMCache](https://github.com/LMCache/LMCache), the KV tier of the 2026-07 generation ([docs/archive/](docs/archive/)).

Benchmarks and instruments:

- [SWE-Bench](https://github.com/SWE-bench/SWE-bench) and the [SWE-Bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) split; [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent), the scaffold; [bench/](bench/README.md), the reproduction package.
- [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench), the 69-scenario tool-calling suite.
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), for GSM8K.
- [llama-benchy](https://github.com/eugr/llama-benchy), for prefill and the profiler captures.
- The probes in this repo: [decode_ss.py](scripts/decode_ss.py) (steady-state decode), [decode_fidelity.py](scripts/decode_fidelity.py) (decode ruler vs bf16), [fidelity_ladder.py](scripts/fidelity_ladder.py) and [fidelity_compare.py](scripts/fidelity_compare.py) (dense and agentic rulers), [kv_capacity_probe.py](scripts/kv_capacity_probe.py) (pool cost per request), [needle_gate.sh](scripts/needle_gate.sh) (tier revisits), [prof_decode_split.py](scripts/prof_decode_split.py) (decode-step profile).

This repo:

- [bench/RESULTS.md](bench/RESULTS.md), every measurement newest first; [docs/HISTORY.md](docs/HISTORY.md), the lineage of the served configuration.
- [docs/CONFIG.md](docs/CONFIG.md), every flag; [docs/DESIGN.md](docs/DESIGN.md), why it fits; [docs/FIDELITY.md](docs/FIDELITY.md), the bf16 rulers; [docs/R156-DECISION.md](docs/R156-DECISION.md), the checkpoint decision.
- [docs/GOTCHAS.md](docs/GOTCHAS.md), failure modes; [docs/REJECTED.md](docs/REJECTED.md), what was tried and rejected.
- [scripts/serve-r168-daily.sh](scripts/serve-r168-daily.sh), the served launcher; [scripts/serve-r156-daily.sh](scripts/serve-r156-daily.sh), the fp8 rollback; [scripts/serve-v0280-daily.sh](scripts/serve-v0280-daily.sh), the one-card and MTP shapes; [scripts/build-v0290rc2.sh](scripts/build-v0290rc2.sh), the image build.
- [THIRD_PARTY.md](THIRD_PARTY.md), provenance of every patch and idea; [LICENSE](LICENSE).
