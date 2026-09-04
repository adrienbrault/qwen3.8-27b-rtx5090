# Qwen3.8-27B on RTX 5090

Serving configuration, vLLM patches, launch scripts and measurements for running [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) on one or two RTX 5090 cards with 262K context. The target workload is a few concurrent coding agents with long contexts, plus interactive chat with vision, reasoning, tool calling and structured output all enabled.

Every number in this repo was measured on one machine, on the date given, and the raw results directory is named next to it. Nothing here is a projection.

## The served configuration

Since 2026-09-04: vLLM 0.29 with an NVFP4 KV cache on two cards, launcher [scripts/serve-r168-daily.sh](scripts/serve-r168-daily.sh), promotion record in [bench/RESULTS.md](bench/RESULTS.md) R168 to R174.

| | value | measured |
|---|---|---|
| context length | 262,144 tokens | |
| KV pool on the GPUs | 937,795 tokens, pinned at 14.5 GB per card | every boot since 2026-09-03 (`results/2026-09-04-r173b-ns-confirm`, `results/2026-09-04-r174-promote`) |
| CPU KV tier | 16 GiB, 1,013 blocks | 2026-09-04, `results/2026-09-04-r172-cputier` |
| disk KV tier | 300 GB cap with LRU eviction, survives restarts | 2026-09-04, `results/2026-09-03-r170-tier-evict` |
| decode, code, 1 stream | 244 t/s (two runs, 226 and 261) | 2026-09-04, `results/2026-09-04-r174-promote` |
| decode, prose, 1 stream | 172 t/s (two boots, 171.6 and 171.1) | 2026-09-04, `results/2026-09-04-r173b-ns-confirm` |
| decode, prose, 1 stream at 30K context | 147 t/s (two boots, 147.6 and 146.4) | same |
| decode, code, 8 streams | 1,134 and 1,146 t/s aggregate (two boots) | same |
| prefill, 2K prompt, 1 request | 8,757 t/s (three runs, sd 36) | 2026-09-04, `results/2026-09-04-r173-c1-opt`, llama-benchy |
| tool-eval, 69 scenarios × 4 trials, parallel 8 | 91, one run | 2026-09-04, `results/2026-09-04-r174-promote` |
| dense text vs the bf16 model: top-1 agreement / perplexity delta / truncated KL | 92.80% / +0.75% / 0.0140 | 2026-09-04, `results/2026-09-04-r168e-splitkv-bf16` |
| agentic turns vs the bf16 model: top-1 / perplexity delta | 95.67% / +2.61% | same |
| SWE-Bench Verified | not run on this route | |

Decode numbers are steady-state tokens per second from [scripts/decode_ss.py](scripts/decode_ss.py) (512 generated tokens, content stated in the results directory), not llama-benchy means. Single-stream decode swings up to 20% between runs inside one boot on this box, so the one-stream cells are ranges, not points.

What the route trades against the two-card fp8 shape it replaced, measured the same day on the same box:

- The pool grows from 654,491 (or 628,798, the fp8 shape's profiler was bimodal) to 937,795, and the pin makes it the same on every boot.
- The disk tier serves. On the fp8 shape every revisit of a 131K prompt was recomputed: a tier hit is promoted through the CPU tier, 331 fp8 blocks did not fit its 4 GiB, and with 16 GiB the v0.28 lookup still served 0 of 4 needles. On the served route, needles at 131K and 220K were served 4 of 4 after a 12-prompt flood and 4 of 4 after a fresh boot, in 1.4 to 2.9 s against 25 to 57 s cold (`results/2026-09-04-r172-cputier`).
- The fidelity cost is the NVFP4 KV cost measured in R156 and unchanged across vLLM versions: 0.3 points of top-1 agreement and 0.4 points of perplexity against bf16. The rc2 image with fp8 KV scores the same as the v0.28 fp8 daily (table below), so the 0.29 chain itself is fidelity-neutral.
- Decode, paired against the rc2 image with fp8 KV (`results/2026-09-03-r169-rc2`, decode_ss): code 1 stream 245 vs 227, 30K context 155 vs 164, code 8 streams 1,110 vs 1,153 (`results/2026-09-04-r173-c1-opt`). Prefill at 2K: 8,757 vs 8,548 (`results/2026-09-03-r166-gates`). Tool-eval on the same instrument: 91 vs 88.

The fp8 shape scored SWE-Bench Verified 386/500 = 77.2% on 2026-09-03 (mini-SWE-agent 2.4.6, official harness, one attempt, `results/2026-09-02-miniswe-rh`). That number belongs to the rollback configuration, not to the served one. Earlier one-card configurations scored 331/500 = 66.2% (2026-08-21, saka checkpoint, R2E-Gym scaffold) and Terminal-Bench 2.1 50/89 = 56.2% (2026-08-23, gittensor checkpoint, Harbor with terminus-2).

## Hardware

- ASRock X870 Taichi Creator, Ryzen 7 9800X3D, 64 GB DDR5-6000, Ubuntu 24.04 HWE.
- Two RTX 5090 32 GB (`sm_120`): ASUS at 600 W and HP OEM at 575 W, PCIe Gen5 x8/x8.
- NVIDIA driver 610.57.04 with the [QuixiAI open kernel modules](https://github.com/QuixiAI/open-gpu-kernel-modules) for GPU peer-to-peer ([scripts/gpu-p2p-610.sh](scripts/gpu-p2p-610.sh)).
- Memory clock offset +4500 MHz on both cards, core clock stock ([scripts/gpu-tune.sh](scripts/gpu-tune.sh)). Decode is memory-bound, so this is worth about 4% decode. All throughput numbers include it.
- One Gen5 x4 NVMe for the model weights and a 393 GB loopback image for the KV disk tier.

The one-card configuration needs only the first RTX 5090 and 64 GB of RAM.

## How the served image is built and what the flags do

The image `vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616` is built by [scripts/build-v0290rc2.sh](scripts/build-v0290rc2.sh) on the CPU: a pinned `vllm/vllm-openai` nightly whose dependency set matches v0.29.0rc2, the rc2 wheel from `wheels.vllm.ai` over it, the [patches-v0290/](patches-v0290/) chain 0101 to 0137 applied with `--fuzz=0`, and a final layer that swaps FlashInfer to 0.6.16.post3. The chain carries the sm120 NVFP4 KV port (FA2 routing, linear V-scale store overlay, XQA decode), DFlash2 with quantized drafters and CUDA graphs, GDN kernel hardening, a pooled FlashInfer workspace (0131), prefix-cache reuse under DFlash (0134), embedding-table offload (0135), an opt-in split-KV knob (0136, unused on this image) and LRU eviction for the disk tier (0137). Design notes per patch are in the `NOTES*.md` files next to the diffs; the v0.28 generation has one README per hunk in [patches-v0280/README-sm120-nvfp4.md](patches-v0280/README-sm120-nvfp4.md).

The flags the launcher asserts at boot, and why ([docs/CONFIG.md](docs/CONFIG.md) has the full table):

- `--kv-cache-dtype nvfp4`: 2,944-token blocks, +43% pool over fp8 at the same VRAM.
- `--kv-cache-memory-bytes 14500000000` with `--gpu-memory-utilization 0.88`: the pool is pinned in bytes, per `--max-num-seqs`, because the utilization path sizes it before CUDA-graph capture and the first request then runs out of memory. The boot fails under 512 MiB free after pre-warm, since a long-prompt logprobs request once killed the fp8 daily with 100 MiB free.
- `VLLM_SM12X_NVFP4_XQA=0`, `--max-num-batched-tokens 8192`, a 512 MiB FlashInfer workspace: NVFP4 prefill chunks above about 4,929 tokens corrupt under XQA decode on this stack, so decode goes through FlashInfer FA2 instead.
- FlashInfer 0.6.16.post3 rather than the 0.6.18 the nightly ships: 0.6.18 force-disables split-KV for packed NVFP4 in its paged-prefill wrapper, and every speculative verification step is a 10-token prefill through that wrapper, so decode at 30K context dropped from 143 to 26.5 t/s. The swap restores it (`scripts/r168-deep-decode.sh`, RESULTS R168).
- DFlash2 drafter, 9 draft tokens, drafter tensor-parallel 2, in CUDA graphs (`VLLM_SM12X_DFLASH_GRAPHS=1`): 7 draft tokens gave +12.5% at 8 streams and 4.6% more pool but sat twice as far from the bf16 decode reference at 30K, so it was retracted (`results/2026-09-04-r173c-bf16-decode`).
- Embedding table in pinned host RAM (`--offload-backend uva --cpu-offload-params embed_tokens`): +9% pool, no measured cost (`results/2026-09-03-r167-embed`).
- CPU tier 16 GiB: a disk-tier hit is served only if every block of the prompt fits the CPU tier at once.
- Disk tier cap 300 GB, 40 GB minimum free, patch 0137: upstream vLLM has no eviction in its filesystem tier and the tier stranded the engine at 100% on 2026-09-01.

## What the measurements established

Nine NVFP4 checkpoints were compared against the bf16 model on 725K teacher-forced text positions and 58K positions of bf16-generated agentic turns (`results/2026-09-01-r156-bf16-ladder`, [docs/R156-DECISION.md](docs/R156-DECISION.md)). Task benchmarks cannot see differences of this size: GSM8K at n=250 resolves about 8 percentage points, and the checkpoints differ by less than one.

- The quantizer recipe matters more than the bit width. Three checkpoints at the same 4-bit width sit at +0.37% perplexity from bf16; one sits at +4.46%.
- A 4-bit `lm_head` alone costs about 0.85 percentage points of perplexity.
- fp8 KV costs +0.13 points of perplexity and does not grow with context out to 171K. 4-bit KV costs +0.76.
- Draft acceptance measures agreement between drafter and target, not quality. Re-run the drafter A/B after every checkpoint change.
- The draft length is a numerics change. Greedy continuations with 7 and 9 draft tokens diverge on 19 of 20 chunks at both 0 and 30K context, the same magnitude as fp8 versus nvfp4 KV, so a prefill-only ruler cannot rank draft lengths (`results/2026-09-04-r173b-ns-confirm`).
- Two boots of the same configuration are bit-identical on 725K positions and on 20 greedy decode chunks, so any difference between arms is the arm (`results/2026-09-04-r168e-splitkv-bf16`, `results/2026-09-04-r168d-splitkv-ref`).

The rulers on the served configuration, its rollback and the previous checkpoint. Dense: 693 documents, 555,549 scored positions. Agentic: 72 bf16-generated turns, 57,972 positions.

| vs bf16 | served: RedHatAI, nvfp4 KV, vLLM 0.29rc2, FlashInfer 0.6.16 (`r168e`) | rollback: RedHatAI, fp8 KV, vLLM 0.28.0 (`r156-bf16-ladder`) | RedHatAI, fp8 KV, vLLM 0.29rc2 (`r169-rc2`) | gittensor, fp8 KV, vLLM 0.28.0, served until 09-02 (`r156-bf16-ladder`) | one configuration, two boots (`r169` vs `r168e`) |
|---|---|---|---|---|---|
| dense top-1 agreement | 92.80% | 93.07% | 93.08% | 88.56% | 92.773% both |
| dense perplexity delta | +0.75% | +0.38% | +0.36% | +4.46% | +0.789% both |
| dense truncated KL | 0.0140 | 0.0129 | 0.0126 | 0.0349 | 0.0144 both |
| agentic top-1 agreement | 95.67% | 95.95% | 95.93% | 92.60% | 95.603% both |
| agentic perplexity delta | +2.61% | +2.41% | +2.38% | +6.56% | +2.586% both |

The served weights are [RedHatAI/Qwen3.8-27B-NVFP4](https://huggingface.co/RedHatAI/Qwen3.8-27B-NVFP4): W4A4 NVFP4 from llm-compressor, 303 modules kept at 8-bit, fp8 `lm_head`. The drafter is [syvai/Qwen3.8-27B-DFlash2-W4A16](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16). The checkpoint served before it cost 4.5 points of top-1 agreement on dense text and 3.4 on agentic turns, which no task benchmark in this repo detected.

## Other configurations

Three shapes remain runnable and documented:

- **Two cards, fp8 KV, DFlash2 on vLLM 0.28.0** ([scripts/serve-r156-daily.sh](scripts/serve-r156-daily.sh)): the rollback. Served 2026-09-02 to 09-04 on the RedHatAI checkpoint with pool 654,491 or 628,798, tool-eval 90.8 ± 0.5 over its life, SWE-Bench Verified 386/500. Its disk tier never served a revisit (`results/2026-09-04-r172-cputier`, arm D16).
- **One card, nvfp4 KV, MTP on vLLM 0.28.0** ([scripts/serve-v0280-daily.sh](scripts/serve-v0280-daily.sh)): the shape for a single RTX 5090.
- **Two cards, nvfp4 KV, MTP on vLLM 0.28.0** (`serve-v0280-daily.sh` with `TP=2`): the capacity shape, largest pool and best aggregate throughput at 16 streams.

The three measured on 2026-08-31 on the gittensor checkpoint, same day, same harness, `results/2026-08-31-r142-matrix`. The RedHatAI checkpoint costs about 6% decode, 14% prefill and 12% pool on the fp8 shape relative to these numbers.

| | one card | two cards, DFlash2, fp8 KV | two cards, MTP, nvfp4 KV |
|---|---|---|---|
| KV pool at 262K | 381,300 | 746,849 | 1,508,519 |
| decode, code, 1 stream | 175.0 t/s | 298.9 | 225.3 |
| decode, code, 8 streams | 1,187 | 1,289 | 1,349 |
| decode, code, 16 streams | not admitted | 1,522 | 2,007 |
| decode at 100K context | 106.7 | 174.4 | 137.5 |
| prefill at 8K | 11.9K t/s | 9.3K | 9.0K |
| prefill at 100K | 4.7K | 7.0K | 6.3K |
| tool-eval ×4 | 89.2 ± 1.7 | 89.8 ± 1.3 | 90.2 ± 1.0 |

DFlash2 accepts few draft tokens per step, so its decode is bound by weight bandwidth, which the second card doubles. MTP accepts more, amortizes the weight reads, and turns the second card into KV space instead. Tool-eval does not separate the three; the bf16 rulers above do, by the KV dtype.

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

Every launcher refuses to serve when a load-bearing property is missing: the image and vLLM version, the tier connector, the KV pool inside its band, free VRAM after pre-warm, free host memory, free space on the tier. The served launcher additionally checks the FlashInfer version, the pin on the container, the drafter graphs, the embedding offload, the CPU-tier size and the tier cap. The one-card launcher checks the store overlay line and the XQA decode backend.

The endpoint is OpenAI-compatible at `http://<host>:8020/v1`, model name `qwen3.8-27b` (alias `qwen3.6-27b`):

```bash
curl -s http://localhost:8020/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "Write a Python function that parses ISO-8601 durations."}]
}' | jq -r '.choices[0].message.content'
```

Reasoning is on by default at effort `medium`. Tool calls, JSON-schema structured output and up to 16 images per request work without extra flags. Clients that resend prior assistant turns should include the `reasoning` field to keep earlier thinking blocks in context.

## Repository map

| path | contents |
|---|---|
| [scripts/](scripts/) | Launchers (`serve-*.sh`), image builds (`build-v0290rc*.sh`), host setup (`setup-native-l2.sh`, `gpu-tune.sh`, `gpu-p2p-610.sh`, `setup-earlyoom.sh`, `tier-evict.sh`), measurement probes (`decode_ss.py`, `decode_fidelity.py`, `fidelity_ladder.py`, `fidelity_compare.py`, `needle_depth.py`, `nvfp4kv-gauntlet.sh`) and the per-experiment drivers (`r1xx-*.sh`), one per results directory. |
| [patches-v0290/](patches-v0290/) | The served patch chain on vLLM v0.29.0rc2, the Dockerfiles that build it, the verification scripts and the design notes per patch. |
| [patches-v0280/](patches-v0280/README-sm120-nvfp4.md) | The v0.28.0 generation of the same chain, one README per hunk; the rollback and the one-card shapes run it. |
| [docs/CONFIG.md](docs/CONFIG.md) | Every flag of the served and the rollback configuration, why it is set, and what breaks without it. |
| [docs/GOTCHAS.md](docs/GOTCHAS.md) | Failure modes found on this stack. Read before changing anything. |
| [docs/REJECTED.md](docs/REJECTED.md) | Configurations that were tried and rejected, with the number that rejected them. |
| [docs/DESIGN.md](docs/DESIGN.md) | Why W4A4 weights, why a hybrid model, where the VRAM goes, why two cards help the way they do. |
| [docs/HISTORY.md](docs/HISTORY.md) | Lineage of the served configuration since 2026-06, reversals included. |
| [docs/R156-DECISION.md](docs/R156-DECISION.md) | The checkpoint decision record, with the reliability review in R156-REVIEW.md. |
| [docs/archive/](docs/archive/) | Documents of earlier stack generations (LMCache tiers, the first NVFP4 KV port, the V2 runner, the first DFlash2 audition). |
| [bench/](bench/README.md) | [RESULTS.md](bench/RESULTS.md) with every measurement newest first, the probe scripts, and the SWE-Bench reproduction package. |
| [patches/](patches/README.md), [patches-nvfp4kv/](patches-nvfp4kv/README.md), [patches-dflash2/](patches-dflash2/README.md) | Earlier patch generations, kept so the configurations in HISTORY.md stay reproducible. |
| [THIRD_PARTY.md](THIRD_PARTY.md) | Per-file provenance of every patch and idea taken from upstream PRs and other repos. |

## Before you change anything

- The NVFP4 store overlay is required on `sm_120` and its absence is invisible to behavioural tests. Without it the engine is fluent, passes needle tests, and has 2.7 to 10 times the attention error. Only a numeric diagnostic catches it.
- Prefill-only fidelity rulers cannot see decode kernels or the draft length. Validate a decode-path change with [scripts/decode_fidelity.py](scripts/decode_fidelity.py) against the bf16 decode reference.
- A FlashInfer bump can change the deep-context decode rate 5x without touching short prompts. Measure decode at 30K context after every library change.
- A disk-tier hit is served only if the whole prompt fits the CPU tier. Size `cpu_bytes_to_use` for the longest prompt you expect to revisit, and test the tier with a needle retrieved after a restart.
- Do not pass `--no-async-scheduling` on vLLM 0.28 or later. It costs 21% to 29% single-stream decode.
- Single-stream decode with speculative decoding varies between boots and between runs. Compare within one boot, or normalize by accepted tokens per step.
- Upstream vLLM's filesystem tier has no eviction. The served image evicts through patch 0137; the v0.28 shapes rely on the fixed-size filesystem and [scripts/tier-evict.sh](scripts/tier-evict.sh).

The full list is in [docs/GOTCHAS.md](docs/GOTCHAS.md).

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
