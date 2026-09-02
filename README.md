# Qwen3.8-27B on RTX 5090

Serving configuration, vLLM patches, launch scripts and measurements for running [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) on one or two RTX 5090 cards with 262K context. The target workload is a few concurrent coding agents with long contexts, plus interactive chat with vision, reasoning, tool calling and structured output all enabled.

Every number in this repo was measured on one machine, on the date given, and the raw results directory is named next to it. Nothing here is a projection.

## What you get

The served configuration (two cards, since 2026-09-02):

| | value |
|---|---|
| context length | 262,144 tokens |
| KV pool on the GPUs | 654,491 tokens (fp8 KV), plus a 200 GB disk tier that survives restarts |
| decode, single stream, code | 318.8 t/s (llama-benchy, T=0.6, pp2048/tg256) |
| decode, 8 streams, code, steady state | 1,212 t/s aggregate |
| prefill at 2K prompt | 8,741 t/s |
| tool-calling benchmark (tool-eval, 69 cases × 4 runs) | 90.8 ± 0.5 |
| perplexity gap vs the bf16 model | +0.38% |
| warm revisit of a 32K context from the disk tier | 0.45 s instead of 7.5 s |

Agentic benchmarks, measured on earlier one-card configurations: SWE-Bench Verified 331/500 = 66.2% (2026-08-21, saka checkpoint, NVFP4 KV with LMCache tiers, R2E-Gym scaffold, official harness) and Terminal-Bench 2.1 50/89 = 56.2% (2026-08-23, gittensor checkpoint, same one-card stack, Harbor with terminus-2). Details in [bench/RESULTS.md](bench/RESULTS.md).

## Hardware

- ASRock X870 Taichi Creator, Ryzen 7 9800X3D, 64 GB DDR5-6000, Ubuntu 24.04 HWE.
- Two RTX 5090 32 GB (`sm_120`): ASUS at 600 W and HP OEM at 575 W, PCIe Gen5 x8/x8.
- NVIDIA driver 610.57.04 with the [QuixiAI open kernel modules](https://github.com/QuixiAI/open-gpu-kernel-modules) for GPU peer-to-peer ([scripts/gpu-p2p-610.sh](scripts/gpu-p2p-610.sh)).
- Memory clock offset +4500 MHz on both cards, core clock stock ([scripts/gpu-tune.sh](scripts/gpu-tune.sh)). Decode is memory-bound, so this is worth about 4% decode. All throughput numbers include it.
- One Gen5 x4 NVMe for the model weights and the KV disk tier.

The one-card configuration needs only the first RTX 5090 and 64 GB of RAM.

## The three configurations

All three run the same image (vLLM v0.28.0 plus [patches-v0280/](patches-v0280/)), the same disk KV tier, and the same sampling settings. Numbers measured 2026-08-31 on the previous checkpoint (gittensor), same day, same harness, `results/2026-08-31-r142-matrix`. The current RedHatAI checkpoint costs about 6% decode, 14% prefill and 12% pool on the two-card DFlash2 shape and gains fidelity (see below).

| | one card | two cards, DFlash2 (served) | two cards, MTP |
|---|---|---|---|
| launcher | [serve-v0280-daily.sh](scripts/serve-v0280-daily.sh) | [serve-r156-daily.sh](scripts/serve-r156-daily.sh) | [serve-v0280-daily.sh](scripts/serve-v0280-daily.sh) with `TP=2` |
| KV cache | nvfp4 (4-bit) | fp8 | nvfp4 (4-bit) |
| speculative decoding | MTP head, 4 tokens | DFlash2 drafter, 9 tokens | MTP head, 4 tokens |
| KV pool at 262K | 381,300 | 746,849 | 1,508,519 |
| decode, code, 1 stream | 175.0 t/s | 298.9 | 225.3 |
| decode, code, 8 streams | 1,187 | 1,289 | 1,349 |
| decode, code, 16 streams | not admitted | 1,522 | 2,007 |
| decode at 100K context | 106.7 | 174.4 | 137.5 |
| prefill at 8K | 11.9K t/s | 9.3K | 9.0K |
| prefill at 100K | 4.7K | 7.0K | 6.3K |
| tool-eval ×4 | 89.2 ± 1.7 | 89.8 ± 1.3 | 90.2 ± 1.0 |

How to read it: the two-card DFlash2 shape has the fastest single stream and the fastest prefill past 8K. The two-card MTP shape has the largest pool and the best aggregate throughput at 16 streams. Quality is the same across the three. The reason the split lands this way: DFlash2 accepts few draft tokens per step, so its decode is bound by weight bandwidth, which the second card doubles. MTP accepts more, amortizes the weight reads, and turns the second card into KV space instead.

## The checkpoint

The served weights are [RedHatAI/Qwen3.8-27B-NVFP4](https://huggingface.co/RedHatAI/Qwen3.8-27B-NVFP4): NVFP4 weights and activations (W4A4) from llm-compressor, 303 modules kept at 8-bit, fp8 `lm_head`. The DFlash2 drafter is [syvai/Qwen3.8-27B-DFlash2-W4A16](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16).

Nine NVFP4 checkpoints were compared against the bf16 model on 725K teacher-forced text positions and on 58K positions of bf16-generated agentic turns (`results/2026-09-01-r156-bf16-ladder`). Task benchmarks cannot see differences of this size: GSM8K at n=250 resolves about 8 percentage points, and the checkpoints differ by less than one. Findings that generalize:

- The quantizer recipe matters more than the bit width. Three checkpoints at the same 4-bit width sit at +0.37% perplexity from bf16; one sits at +4.46%.
- A 4-bit `lm_head` alone costs about 0.85 percentage points of perplexity.
- fp8 KV costs +0.13 points and does not grow with context out to 171K. 4-bit KV costs +0.76.
- Draft acceptance measures agreement between drafter and target, not quality. Re-run the drafter A/B after every checkpoint change.

Full method and tables: [bench/RESULTS.md](bench/RESULTS.md), and the decision record in [docs/R156-DECISION.md](docs/R156-DECISION.md).

## Quick start

The scripts assume the host layout used here: models under `/srv/qwen5090/models`, compile caches under `/srv/qwen5090/cache`, the disk tier at `/srv/qwen5090/native-l2`, and [scripts/serve-v0280-daily.sh](scripts/serve-v0280-daily.sh) installed as `/srv/qwen5090/launch-daily-v0280.sh`. Adjust the paths at the top of each script if your layout differs.

```bash
# 1. weights (22 GB on disk) and the DFlash2 drafter (1.2 GB, two-card config only)
huggingface-cli download RedHatAI/Qwen3.8-27B-NVFP4 --local-dir /srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4
huggingface-cli download syvai/Qwen3.8-27B-DFlash2-W4A16 --local-dir /srv/qwen5090/models/dflash2-qwen38-syvai-w4a16

# 2. the disk KV tier: a fixed-size loopback filesystem, so the cache cannot fill the root disk
sudo bash scripts/setup-native-l2.sh

# 3. the image: vLLM v0.28.0 + the sm120 patches (no GPU needed to build)
docker build -f patches-v0280/Dockerfile.v0280-nvfp4kv -t vllm-qwen38:v0280-nvfp4kv patches-v0280

# 4a. two cards (the served configuration)
bash scripts/serve-r156-daily.sh

# 4b. one card
MODEL_DIR=/srv/qwen5090/models/qwen3.8-27b-redhat-nvfp4 PORT=8020 NAME=vllm-27b bash scripts/serve-v0280-daily.sh
```

The launcher refuses to serve if a load-bearing property is missing: the tier connector, the KV pool inside its expected band, enough free host memory, enough free space on the tier, and on the one-card NVFP4 configuration the store overlay line in the boot log and the XQA decode backend.

The endpoint is OpenAI-compatible at `http://<host>:8020/v1`, model name `qwen3.8-27b`:

```bash
curl -s http://localhost:8020/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "Write a Python function that parses ISO-8601 durations."}]
}' | jq -r '.choices[0].message.content'
```

Reasoning is on by default at effort `medium`. Tool calls, JSON-schema structured output and up to four images per request work without extra flags. Clients that resend prior assistant turns should include the `reasoning` field to keep earlier thinking blocks in context.

## Repository map

| path | contents |
|---|---|
| [scripts/](scripts/) | Launchers (`serve-*.sh`), host setup (`setup-native-l2.sh`, `gpu-tune.sh`, `gpu-p2p-610.sh`, `setup-earlyoom.sh`, `tier-evict.sh`), measurement probes (`decode_ss.py`, `fidelity.py`, `needle_depth.py`, `nvfp4kv-gauntlet.sh`) and the per-experiment drivers (`r1xx-*.sh`). |
| [patches-v0280/](patches-v0280/README-sm120-nvfp4.md) | The current patch stack on vLLM v0.28.0: NVFP4 KV cache on `sm_120` (FA2 routing, linear V-scale store overlay, XQA decode, drafter CUDA graphs), DFlash2 with quantized drafters, GDN kernel hardening. One README per hunk. |
| [docs/CONFIG.md](docs/CONFIG.md) | Every flag of the served configuration, why it is set, and what breaks without it. |
| [docs/GOTCHAS.md](docs/GOTCHAS.md) | Failure modes found on this stack. Read before changing anything. |
| [docs/REJECTED.md](docs/REJECTED.md) | Configurations that were tried and rejected, with the number that rejected them. |
| [docs/DESIGN.md](docs/DESIGN.md) | Why W4A4 weights, why a hybrid model, where the VRAM goes, why two cards help the way they do. |
| [docs/HISTORY.md](docs/HISTORY.md) | Lineage of the served configuration since 2026-06, reversals included. |
| [docs/archive/](docs/archive/) | Documents of earlier stack generations (LMCache tiers, the first NVFP4 KV port, the V2 runner, the first DFlash2 audition). |
| [bench/](bench/README.md) | [RESULTS.md](bench/RESULTS.md) with every measurement newest first, the probe scripts, and the SWE-Bench reproduction package. |
| [patches/](patches/README.md), [patches-nvfp4kv/](patches-nvfp4kv/README.md), [patches-dflash2/](patches-dflash2/README.md) | Earlier patch generations, kept so the configurations in HISTORY.md stay reproducible. |
| [THIRD_PARTY.md](THIRD_PARTY.md) | Per-file provenance of every patch and idea taken from upstream PRs and other repos. |

## Before you change anything

- The NVFP4 store overlay is required on `sm_120` and its absence is invisible to behavioural tests. Without it the engine is fluent, passes needle tests, and has 2.7 to 10 times the attention error. Only a numeric diagnostic catches it.
- Do not pass `--no-async-scheduling` on vLLM 0.28 or later. It costs 21% to 29% single-stream decode. The bug it once guarded is fixed upstream.
- Single-stream decode with speculative decoding varies about 10% between boots on identical prompts. Compare within one boot, or normalize by accepted tokens per step.
- The disk tier has no eviction. Bound it with a fixed-size filesystem and wipe stale namespaces, or use [scripts/tier-evict.sh](scripts/tier-evict.sh).

The full list is in [docs/GOTCHAS.md](docs/GOTCHAS.md).

## License

MIT ([LICENSE](LICENSE)) for the original work: documentation, scripts, probes and patch tooling. Everything derived from vLLM, LMCache or FlashInfer, including redistributed PR diffs and patched files inside built images, stays Apache-2.0-derived. Per-file inventory: [THIRD_PARTY.md](THIRD_PARTY.md).

## Credits

- [vLLM](https://github.com/vllm-project/vllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [LMCache](https://github.com/LMCache/LMCache).
- ch2lab for [vLLM PR #49891](https://github.com/vllm-project/vllm/pull/49891), the sm120 NVFP4 KV routing.
- drowzeys for the linear V-scale writer fix ([DGX Spark repo](https://github.com/drowzeys/keys-vLLm.0.27-Qwen3.8-27B-ADay777Ablit-NVFP4-A4Q-NVFP4-KV-4M-KV-token-pool-MTP3-Single-DGX-Spark)).
- The author of [vllm#49011](https://github.com/vllm-project/vllm/issues/49011) for demonstrating XQA-NVFP4 decode with FA2 prefill on sm120, and [hikarioyama](https://github.com/hikarioyama/vllm-nvfp4-kv-sm120) for the FA2 SF-stride prior art.
- [seanyourhighness](https://github.com/seanyourhighness/vllm-sm12x-nvfp4-dflash2) for the DFlash2-on-NVFP4 overlay and the `--kv-cache-memory-bytes` pinning.
- [RedHatAI](https://huggingface.co/RedHatAI/Qwen3.8-27B-NVFP4) for the served weights; [unsloth](https://huggingface.co/unsloth) and [kelnei](https://huggingface.co/kelnei) for the two checkpoints that tie it on fidelity; [sakamakismile](https://huggingface.co/sakamakismile/Qwen3.8-27B-MTP-NVFP4) and [gittensor](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090-LMHead4) for the earlier served checkpoints; [syv-ai](https://huggingface.co/syvai) for the quantized DFlash2 drafter; z-lab and inco.ai for DFlash2 and [vLLM PR #52816](https://github.com/vllm-project/vllm/pull/52816).
