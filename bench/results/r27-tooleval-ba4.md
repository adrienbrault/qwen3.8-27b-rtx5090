# r27 tool-eval on the served configuration with BA4 active (2026-11-26, results `2026-11-26-r27-tooleval-ba4`)

[← all results](../RESULTS.md)

2026-11-26 01:43 to 01:52 UTC, raw records in [`2026-11-26-r27-tooleval-ba4/`](2026-11-26-r27-tooleval-ba4/). One boot of the served launcher ([`scripts/serve-v031-served-daily.sh`](../../scripts/serve-v031-served-daily.sh), image `vllm-qwen38:v0310-r27c-A_S-BA4-L1a-ubplan`), nothing else using the server. The boot log reported all 96 BA4 rank-layers active and none inactive.

This run replaces the [2026-11-23 row](r27-tooleval-think.md) (94.5 ± 1.7). That boot ran without BA4, the admitted Triton M=4 kernel for the linear-attention `in_proj_ba` projection. BA4 admits each card against a recorded fingerprint, listed in rank order. On 2026-11-20 the two cards swapped PCIe slots, which reversed CUDA's device order. Each rank then met the other card's fingerprint, and all 96 rank-layers fell back to the original GEMM. The launcher now pins CUDA's device order to the admitted cards and gives BA4 compiles their own `torch.compile` cache directory.

[tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench), 69 scenarios × 4 trials, temperature 0.6, top-p 0.95, top-k 20, 8 in parallel.

| arm | score (mean ± stddev of 4 trials), median | trial points of 138, mean |
| --- | ---: | ---: |
| A: tool-eval-bench 2.1.0 | 90.8 ± 0.5, 91.0 | 124.8 |
| D: 2.1.0 + reasoning echo on every assistant turn (376 of 376 tool-calling and 335 of 335 other turns) | 91.0 ± 1.4, 91.5 | 125.5 |
| C: tool-eval-bench 3.0.0 | 95.2 ± 1.5, 95.0 | 131.8 |

- Arm D sends each earlier assistant turn's reasoning back as `reasoning_content`, as pi and Hermes do. The Qwen3.8 template keeps reasoning on every earlier turn. The shim is [`scripts/tooleval_echo.py`](../../scripts/tooleval_echo.py) with `ECHO_ALL=1`. Against A it moved the score by +0.2, within noise. The 2026-11-23 run echoed tool-calling turns only and moved the score by +1.5, also within noise.
- 3.0.0 reads 95.2 here against 94.5 without BA4. The difference is within one trial standard deviation, so BA4 is not shown to change the score. This row stands because it was measured on the served configuration.
- Most of the difference between 2.1.0 and 3.0.0 comes from 3.0.0's other changes: it re-grades some scenarios and changed how they run.
