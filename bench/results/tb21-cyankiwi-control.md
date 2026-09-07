# Terminal-Bench 2.1 control: the most faithful AWQ checkpoint scored 7 pts lower (2026-08-24/25, `results/2026-08-24-tb21-cyankiwi`)

[← all results](../RESULTS.md)

This run tested directly whether checkpoint precision is what separates this rig from Qwen's reported 73. cyankiwi/Qwen3.8-27B-AWQ-INT4 is the most faithful quant measured on this box (top-1 agreement 0.934 vs FP8, confident-flip rate half of the daily's), but it has no MTP head, ~2x slower decode and ~3x slower prefill. It ran through the identical leaderboard-legal harness: Harbor 0.18.0, terminus-2, k=1, default timeouts x1.0, n-concurrent 2 rather than 3, which if anything favours the slower engine.

The control measured **44 PASS / 15 FAIL / 28 agent-timeout / 2 env-error = 49.4%**, against 56.2% for the NVFP4 daily. Head-to-head it took 5 wins (mailman, sam-cell-seg, sqlite-db-truncate, torch-tensor-parallelism, tune-mjcf), mostly coin-flip-class tasks the daily's own k=1 run dropped, against 11 losses. 8 of the losses are PASS-to-timeout (build-cython-ext, caffe-cifar-10, compile-compcert, db-wal-recovery, largest-eigenval, password-recovery, polyglot-rust-c, rstan-to-pystan), largely the same tasks the x4-timeout diagnostic flagged as wall-clock-bound, plus 3 PASS-to-FAIL. Agent timeouts went 19 to 28.

This pairs with the one-ruler table below: fidelity is real and measurable, but on a time-budgeted agentic benchmark **throughput dominates**. The 4.5-pt agreement edge buys ~5 task wins while the missing MTP and FP4-GEMM speed loses 11. Checkpoint precision is not the term in the gap to Qwen's reported 73; that gap decomposes into best-of-k (k=5 vs k=1), the 62.9% wall-clock ceiling, and model scale.
