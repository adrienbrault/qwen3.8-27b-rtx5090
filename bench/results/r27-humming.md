# Humming 0.1.18 reduces single-stream decode on the served 27B configuration (2026-10-21)

Humming 0.1.18 was not selected for serving. Its NVFP4 W4A4 path reduces single-stream code decode by about 8% and prose by about 12% relative to the current daily controls. Four- and sixteen-stream results do not show a consistent gain. The served launcher and image are unchanged.

## NVFP4-only selection preserves the serving configuration

Measured on 2026-10-21, results `2026-10-21-072610-r27-humming-screen` (0.1.16 with upstream device fixes) and `2026-10-21-074028-r27-humming-screen` (official 0.1.18). Both use the served image as their base: vLLM 0.31.0, FlashInfer 0.7.0.post1, NVIDIA Qwen3.8-27B NVFP4 weights, TP2 on two RTX 5090s, MTP three draft tokens, BA4 active on 96/96 rank-layers, L1a, no block drop, NVFP4 KV pinned at 15.90 GB per card, bf16 SSM, 16 sequences and 262,144 maximum length. Measurements use stock power limits of 600/575 W and memory offset +4500 MHz; the daily returns to 400 W per card afterward.

The candidate argument is `--kernel-config '{"linear_backend_per_quant":{"nvfp4_w4a4":"humming"}}'`. FP8 linear selection stays automatic. The ModelOpt loader creates `input_global_scale`; the Humming adapter uses it to select NVFP4 activations, so this is W4A4 rather than the older W4A16 experiment. Each candidate uses a cleared scratch tier; it cannot read or write the daily's disk-tier blocks.

Each complete comparison has a daily control before and after one candidate boot. The 0.1.18 comparison reuses the preceding comparison's completed closing-control timing data as its opening control, then refreshes greedy dumps before booting the candidate. Controls use the normal daily endpoint; external API traffic was not blocked. The candidate endpoint is loopback-only.

## The official 0.1.18 release loses at one stream

Per-stream steady-state decode tokens/s, three 1,024-token runs per cell, temperature 0.6, fixed prompt-seed prefix `humming-20261021-`, measured by `decode_ss.py` over the samples with all requested streams running. Each number is that boot's median. The last column compares Humming with the arithmetic mean of the two control medians. Raw records also contain aggregate throughput, acceptance and individual-run ranges; these are short screens rather than confidence intervals.

| Kind | Streams | Daily before | Humming | Daily after | vs control mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| code | 1 | 284.4 | 261.2 | 283.3 | -8.0% |
| code | 4 | 230.8 | 217.5 | 222.6 | -4.1% |
| code | 16 | 167.6 | 170.6 | 169.7 | +1.2% |
| prose | 1 | 225.1 | 198.9 | 225.7 | -11.8% |
| prose | 4 | 188.3 | 190.0 | 190.3 | +0.4% |
| prose | 16 | 140.3 | 139.2 | 140.4 | -0.8% |

## The metadata-only backport also loses at one stream

Humming 0.1.16 with only upstream DeviceInfo fixes, same instrument and serving configuration, results `2026-10-21-072610-r27-humming-screen`:

| Kind | Streams | Daily before | Humming | Daily after | vs control mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| code | 1 | 284.4 | 253.2 | 284.4 | -11.0% |
| code | 4 | 228.7 | 220.1 | 230.8 | -4.2% |
| code | 16 | 171.6 | 166.8 | 167.6 | -1.7% |
| prose | 1 | 225.2 | 199.3 | 225.1 | -11.5% |
| prose | 4 | 187.3 | 190.6 | 188.3 | +1.5% |
| prose | 16 | 139.1 | 139.8 | 140.3 | +0.1% |

## Cold prefill does not show a gain

One fresh prompt per size per boot in the 0.1.18 comparison, `kv_capacity_probe.py`. Requested approximate context sizes are 30K and 120K; actual tokenized prompts are about 25K and 100K. Prompts differ to avoid prefix hits. These single observations do not establish a small percentage difference.

| Approximate actual prompt tokens | Daily before TTFT | Humming TTFT | Daily after TTFT |
| --- | ---: | ---: | ---: |
| 25K | 3.0 s | 3.2 s | 3.2 s |
| 100K | 17.3 s | 18.4 s | 17.4 s |

## Capacity and functional checks pass; quality was not qualified

Both candidates retain the 1,490,437-token pool. Minimum free VRAM after startup is 2,827 MiB on the metadata-only backport and 2,451 MiB on 0.1.18. Both pass arithmetic, a parsed tool call with checked arguments, and a completion with finite decode logprobs. No GPU Xid was recorded during the workflow. The final daily is healthy with no container restart or OOM, at 400 W per card, and the experiment queue is empty.

The short greedy comparison is not bit-identical: 0/6 continuations fully agree for each Humming candidate. A daily-versus-daily comparison in the first complete screen also agrees on 0/6 short continuations, so divergence alone does not diagnose a new correctness bug. At long context, several raw-completion prompts terminate after one token; the apparent long-context agreement is not a qualification gate. Top-2 alternatives are retained for the 0.1.18 comparison. No near-tie waiver, task-quality qualification or serving promotion follows from these checks. The throughput loss ends this screen without spending GPU time on qualification.

## The installed package predates the startup fixes

The initial unmodified Humming 0.1.16 boot, results `2026-10-21-071624-r27-humming-screen`, selects `HummingNvFp4LinearKernel` and fails under Dynamo with `RecursionError: maximum recursion depth exceeded` in `humming/device.py`. No candidate throughput is measured on that failed attempt. A CPU test reproduces the real class's missing-slot introspection failure before the patch and passes afterward, including copy/deepcopy and a fullgraph Dynamo call. Only CUDA metadata is substituted in that CPU test; the two successful serving boots provide the GPU integration checks.

The intermediate image backports upstream PRs #148/#149 into 0.1.16 without changing GEMM kernels. The second image installs the official 0.1.18 x86_64 wheel by exact URL and SHA256 without changing other dependencies. The served image is never rebuilt or retagged. The GPU workflow runs from 07:16:24 to 07:51:58 UTC, about 36 minutes including the failed boot and daily restores; the initial estimate was 25–35 minutes, later extended by roughly ten minutes for the official-release comparison.

Raw measurements and exact probe inputs: [`2026-10-21-humming/`](2026-10-21-humming/). Experiment images, dependency pins and source attribution: [`patches-v0310/humming/`](../../patches-v0310/humming/) and [THIRD_PARTY](../../THIRD_PARTY.md#humming-comparison-2026-10-21). The upstream announcement is [Humming](https://vllm.ai/blog/2026-10-21-humming); its kernel microbenchmarks are not end-to-end numbers for this serving configuration.
