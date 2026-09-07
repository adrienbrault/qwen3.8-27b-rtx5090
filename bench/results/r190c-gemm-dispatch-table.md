# R190c, NVFP4 GEMM dispatch table (patch 0140) in the engine: parity; and two fresh torch.compile artifacts of one configuration do not agree at temperature 0 (2026-09-05 05:58 to 06:24 UTC, `results/2026-09-05-r190c-dispatch`, `scripts/r190c-dispatch.sh`)

[← all results](../RESULTS.md)

The R190 census had FlashInfer's B12x kernel at 18 µs against 45 µs for the served Cutlass kernel on the down projection at M = 10. Patch 0140 lets a JSON table pick the kernel per module and M range. Four boots on the daily image plus the patch, pcie_ipc on, 16 sequences, 13.98 GB pin: control (table unset), B12x on down and gate_up at M ≤ 256 (4 rules, 112 modules), the 58-rule table generated from the census, and the B12x table again.

| arm | steps/s code c1 | prose c1 | code c8 | code c16 | agentic top-1 vs bf16 | agentic PPL gap |
|---|---:|---:|---:|---:|---:|---:|
| control | 74.6 | 74.3 | 368.5 | 482.4 | 95.608% | +2.666% |
| B12x table | 74.6 | 74.5 | 369.0 | 483.0 | 95.563% | +2.661% |
| B12x table, second boot | 74.8 | 73.8 | 369.1 | 482.9 | 95.563% | +2.661% |

Step rate is unchanged: at the served decode sizes the MLP GEMMs are not what a step waits on, so a kernel that is 2.5 times faster in isolation buys nothing. The 58-rule table cannot boot: the patch branches on M inside the weight-apply path and dynamo raises a constraint violation during graph capture on both pins. The route is closed.

The side finding matters more. The greedy decode ruler (20 prompts, 256 tokens, per-token logprobs) put the B12x boot 19 of 20 continuations away from the control at context 0 (median absolute delta logprob 0.00042) and 17 of 20 at 30K (0.00632); against bf16 the two boots read 0.0004 and 0.00062 at context 0. Yet the second B12x boot, which loaded the first boot's saved compile artifact, matched it 20 of 20 at both contexts, and the B12x boot also matched the control boot of the R190e run (a different image, no table) 20 of 20. Every pair that shared a saved artifact has been bitwise identical (also R191's two OFF boots); every pair that did not has differed. The FlashInfer GEMM autotuner is not the cause (the second B12x boot re-ran it, 21 profiles, and stayed identical). `VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE` is not either: vLLM applies it only to single-size compile ranges and this configuration has none. The remaining suspect is inductor's runtime Triton config autotune, which benchmarks block sizes on every fresh compile and bundles the choice into the artifact. Until R193 resolves it, a decode-ruler difference of about 0.0002 median between arms compiled separately is not evidence of an arm effect; the R188 effect (0.0042 to 0.0067 against 0.0074 at 30K, dense PPL +0.21% against +0.67%) is well above that and stands, and the agentic ruler moved 0.05 points across artifacts.
