# R183, a decode-step profile of the served route and a 20-arm lever ladder (GEMM kernel, all-reduce backends, fusion passes, prefill chunk, speculation policy, DP2): the two-card decode tax is the custom all-reduce, not NCCL, and no arm beats the replicate band (2026-09-04, `results/2026-09-04-r183-next-levers`, `scripts/r183-next-levers.sh`)

[← all results](../RESULTS.md)

One chain on the bf16-state daily flags (`EXP=1` path of `scripts/serve-r168-daily.sh`, pin 13.98 GB, 16 sequences, tier wiped before every arm). Every arm boots, then runs decode_ss (code 1 stream ×2, prose 1 stream ×2, code 8 streams ×2, code 16 streams ×1), a cold TTFT ladder ([scripts/kv_capacity_probe.py](../../scripts/kv_capacity_probe.py) at 8 output tokens; prompts of 6.7K, 30K, 100K and 200K tokens), the decode ruler at 0 context, and the dense ruler against bf16. The BASE arm carried an idle torch-profiler configuration and was profiled separately at 1, 8 and 16 streams with llama-benchy (pp2048 tg256).

**BASE** (the served flags): code 1 stream 285.0 (281 to 289), prose 158.1, prose at 30K context 152.9, code 8 streams 1,327 (1,312 to 1,342), code 16 streams 1,738 (109 per stream; all 16 admitted with the bf16 state). TTFT 6.7K 0.8 s, 30K 3.8 s, 100K 17.1 s, 200K 47.7 s. Dense ruler 92.771% / +0.744% / KL 0.014082, identical to `r180`; decode ruler median 0.00039. Engine error lines 0.

**The decode-step profile** (torch profiler traces, decode-only window after the last NCCL all-reduce, [scripts/prof_decode_split.py](../../scripts/prof_decode_split.py); steps counted by the GDN update kernel):

| streams | ms per step | GEMM | custom all-reduce | drafter | GDN | attention | inter-kernel gaps |
|---|---|---|---|---|---|---|---|
| 1 | 18.4 (14.4 busy) | 7.1 (265 calls, 27 µs) | 2.7 (143 calls, 19 µs) | 0.9 | 0.7 | 0.6 | 22%; about 1,400 small kernels per step (elementwise, Triton, FP4 quant, copies), about 2,000 device events in total |
| 8 | 24.9 | 8.4 | 6.8 (49 µs per call) | | | | 12% |
| 16 | 38.1 | 11.1 | 12.6 (90 µs per call, 33% of the step) | | 3.2 | | 7% |

The two ranks are symmetric. The cost that grows with concurrency is the custom all-reduce kernel, from 15% of the step at 1 stream to 33% at 16; at 1 stream 22% of the step is gaps between small kernels. NCCL all-gathers remain in decode at about 3 per step (0.18 / 1.01 / 1.99 ms at 1 / 8 / 16 streams, the vocabulary gathers); no NCCL all-reduce does.

**Correction of R158.** R158 reported NCCL at 16 ms per step, 35% of the step at 8 streams. That trace was llama-benchy's 2,048-token prefill chunks: a 21 MB all-reduce exceeds the 8 MiB cap of the custom all-reduce and goes through the NCCL ring, and the call counts (k × 128 + k) match the prefill steps ([scripts/nccl_ctx.py](../../scripts/nccl_ctx.py) shows GEMM-heavy prefill kernels on both sides of every NCCL call). Decode all-reduces stay under the cap and never touch NCCL. NCCL is not a decode cost on this box.

**Levers that do nothing on this box**: FlashInfer all-reduce (`not supported for world_size=2`), the symmetric-memory all-reduce (`capability 12.0 not supported`), so the served backends stay CUSTOM + PYNCCL; those arms run as same-configuration replicates and give the noise bar for the others. `--linear-backend b12x` refuses to boot (`Failed to find a kernel that can implement the ScaledMM linear layer`: the filter applies to the fp8 layers as well), so the kernel ladder in `r183b-kernels.sh` walks `VLLM_DISABLED_KERNELS` on the automatic path instead.

**First lever measured, `--linear-backend flashinfer_b12x`** (NVFP4 GEMM kernel FlashInferB12x instead of FlashInferCutlass; the fp8 layers keep their kernel): code 1 stream 285.3 vs 285.0, prose 1 stream 177 vs 158, code 8 streams 1,388 vs 1,327, code 16 streams 1,833 vs 1,738, TTFT 6.7K 1.1 s vs 0.8, 100K 17.1 vs 17.1; dense ruler 92.769% / +0.758% / KL 0.014081, decode ruler 0.0004; pool 1,020,596, 969 MiB free after pre-warm. The 8- and 16-stream deltas are inside one arm's run-to-run spread at 8 streams (1,312 to 1,342 on BASE, 1,336 to 1,441 on this arm) and are judged against the replicate arms when the chain completes.

**The full sheet.** Code c1 / prose c1 / code c8 / code c16 in t/s, then dense top-1 / perplexity delta vs bf16 where the ruler ran. Same-configuration replicates: BASE, AR-fi, AR-symm and FU-sptp (the requested all-reduce backends are unsupported on this box and the SP passes matched nothing, so those boots ran the served flags; the ruler is bit-identical to BASE on each). Their band: c1 285 to 308, c8 1,327 to 1,421, c16 1,738 to 1,804.

| arm | change | c1 | prose c1 | c8 | c16 | top-1 / PPL |
|---|---|---|---|---|---|---|
| BASE | served flags, idle profiler config | 285 | 158 | 1,327 | 1,738 | 92.771% / +0.744% |
| AR-fi | replicate (booted at pin 13.5 GB after a headroom failure at 13.98) | 308 † | 159 | 1,421 | 1,804 | 92.771% / +0.744% |
| AR-symm | replicate | 308 † | 159 | 1,376 | 1,764 | 92.771% / +0.744% |
| FU-sptp | replicate (`enable_sp`, `fuse_gemm_comms`: inert) | 306 | 159 | 1,363 | 1,767 | 92.771% / +0.744% |
| LB-flashinfer_b12x | NVFP4 GEMM kernel FlashInferB12x | 285 | 177 | 1,388 | 1,833 | 92.769% / +0.758% |
| LB-flashinfer_cudnn | NVFP4 GEMM kernel FlashInferCudnn | 234 | 155 | 1,339 | 1,785 | 92.759% / +0.808% |
| FU-arrms | `fuse_allreduce_rms` (matched nothing) | 253 | 168 | 1,334 | 1,785 | 92.806% / +0.756% |
| FU-arrms2 | same plus custom op `+rms_norm` | 286 | 172 | 1,398 | 1,835 | 92.773% / +0.761% |
| FU-all | all fusion passes | 257 | 169 | 1,363 | 1,811 | 92.806% / +0.756% |
| SP-argmax | `use_local_argmax_reduction` | 309 | 160 | 1,350 | 1,867 | |
| SP-dyn2 | 9 draft tokens up to batch 8, then 1 | 302 | 159 | 1,345 | | |
| SP-dtp1 | drafter tensor-parallel 1 | 308 | 158 | 1,423 | 1,833 | |

† The AR-fi and AR-symm c1 summaries are byte-identical (308.2, runs 298.8 and 317.6 on both), which two timed runs cannot produce; treated as a probe artefact and not used.

Every arm sits inside the replicate band or inside one arm's own run-to-run spread at c8 (FU-arrms2's two c8 runs read 1,312 and 1,484). The highest single c16 read, SP-argmax at 1,867, is 3.5% above the best replicate on one run. The cudnn GEMM kernel is slower and farther from bf16 on the dense ruler. No lever in this ladder is a result; the software knobs vLLM 0.29 exposes for the TP2 decode tax are exhausted on this box.

**Boots that failed, and why.** `--linear-backend b12x`, `cutlass` and `marlin` refuse to start because the filter applies to every layer type: b12x has no fp8 ScaledMM kernel, cutlass and marlin none for the drafter's W4A16 layers. `flashinfer_trtllm` (19:20 UTC) and `humming` (19:24) also failed to boot, for reasons the run did not record, so five of the seven `--linear-backend` arms never started. `--max-num-batched-tokens 16384` leaves 171 MiB free after pre-warm at the 13.98 GB pin and 35 MiB at 13.5 GB, under the 384 MiB floor; 32768 hits a warm-up CUDA error. Data-parallel 2 on `serve-v0280-daily.sh` fails at KV sizing: one replica needs 6.69 GiB of KV for a single 262K sequence and has 2.05 GiB left after the weights at utilization 0.90, so DP2 cannot serve the 262K contract on 32 GB cards. The three-range dynamic draft schedule (SP-dyn1) came up but failed the harness's post-boot check and was not re-run.

**All-reduce cost per call.** Bytes per call are rows × 5,120 × 2: 10 rows at c1 with the 9-token draft (100 KB) take 19 µs, about 5 GB/s, latency-bound; 160 rows at c16 (1.6 MB) take 90 µs, about 18 GB/s, roughly two thirds of the PCIe Gen5 x8 link. The c16 floor is closer to bandwidth than to latency. The kernel itself is the remaining lever: [R184](r184-all-reduce-microbench.md) benchmarks NCCL, vLLM's custom all-reduce and FlashInfer main's `pcie_ipc` all-reduce ([flashinfer PR #4393](https://github.com/flashinfer-ai/flashinfer/pull/4393), merged 2026-08-20, in no 0.6.x release) at these row counts.
