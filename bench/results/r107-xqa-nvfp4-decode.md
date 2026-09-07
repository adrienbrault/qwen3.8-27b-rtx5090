# XQA-NVFP4 decode wired: the nvfp4 speed penalty removed and the decode path instrumented (2026-08-28, `results/2026-08-28-r107*`, `patches-v0280/0103+0104`)

[← all results](../RESULTS.md)

FlashInfer 0.6.16.post3 ships an SM120-exclusive XQA decode kernel that reads NVFP4 KV in a linear scale-factor layout, compatible with the fixed writer, and vLLM never wired it. `0103` routes sm120 nvfp4 q_len=1 decode to it, with `VLLM_SM12X_NVFP4_XQA=0` as a runtime fallback to FA2. `0104` rebases the MTP-drafter FULL-cudagraph routing that v0.28 never absorbed. Result (async ON, MTP ns=4, aggregate t/s):

| | nvfp4 FA2 (prev) | **nvfp4 XQA** | fp8 XQA |
|---|---|---|---|
| prose c1 / c4 | 109.9 / 447.8 | **127.4 / 549.2** | 131.7 / 578.6 |
| code c1 / c4 | 142.8 / 565.4 | **192.6 / 645.4** | 195.8 / 675.4 |

nvfp4 now decodes at **95–98% of fp8** with a 1.53× KV pool (345,553 @262K with MTP; 478K at ns=0). Decomposition via the env knob: drafter cudagraphs +6%, XQA kernel +28%.

Correctness was instrumented rather than assumed. The prefill-logprob ruler is provably blind to decode-only kernels: XQA and FA2 arms produce bit-identical prefill fidelity tables, because `prompt_logprobs` never executes a decode step. A new decode-path probe (`scripts/decode_fidelity.py`: T=0 greedy, per-token logprobs, ns=0 so every step exercises the kernel) shows the XQA-vs-FA2 divergence is real kernel signal, since the FA2 kernel is bit-self-consistent across boots (20/20 chunks identical) while XQA diverges on 16/20. The signature is benign: deltas only at high-entropy positions, sign in both directions, median |Δlogprob| 4e-4, and no positional clustering. The task-accuracy discriminator settles it: GSM8K cot-zeroshot at T=0 over 250 problems measured **0.876 ± 0.021 on both kernels**. These are different but valid numerics over 4-bit KV, not misreads.
