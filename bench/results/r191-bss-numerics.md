# R191, batch-sharded sampling is not numerically equivalent to the dense sampler at temperature 0 and is farther from bf16; rejected (2026-09-05 03:57 to 04:19 UTC, `results/2026-09-05-r191-bss-numerics`, `scripts/r191-bss-numerics.sh`)

[← all results](../RESULTS.md)

Same image and flags as the daily plus the pcie_ipc all-reduce, 16 sequences, three boots: OFF-a, ON (`--enable-batch-sharded-sampling`), OFF-b. Each boot ran the greedy decode ruler (20 prompts, 256 tokens, per-token logprobs) at context 0 and 30K, the agentic ruler against the bf16 dump, and the decode probes.

| comparison | ctx 0 agreeing chunks | ctx 0 median abs. delta logprob | ctx 30K agreeing chunks | ctx 30K median abs. delta logprob |
|---|---:|---:|---:|---:|
| OFF-b vs OFF-a (run-to-run) | 20/20 | 0.0 | 20/20 | 0.0 |
| ON vs OFF-a | 1/20 | 0.00049 | 3/20 | 0.00665 |
| OFF vs bf16 | 1/20 | 0.00039 | 3/20 | 0.00631 |
| ON vs bf16 | 1/20 | 0.00049 | 3/20 | 0.00768 |

The engine is bitwise reproducible across boots (the two OFF boots even used different KV pins), so the ruler resolves the sampler change exactly: sharded sampling diverges from the dense sampler on 19 of 20 greedy continuations at context 0 and sits 22 to 25% farther from bf16. The agentic ruler agrees: top-1 agreement with bf16 95.575% and 95.572% on the two OFF boots (2,565 and 2,567 flips) against 95.479% ON (2,621 flips); corpus perplexity delta +2.68% against +2.72%.

Step rate in steps per second (tokens per second divided by 1 + 9 times the acceptance per draft token): code c8 371.6 and 372.1 OFF against 379.0 ON (+2.0%), code c16 485.6 and 484.6 against 505.0 (+4.1%), code c1 75.7 against 75.0, prose c1 75.4 and 75.9 against 74.7. The flag stays off; the fidelity ruler is the gate and the step-rate gain does not pass it.

Caveat added 2026-09-05 06:25 UTC after R190c: the ON boot loaded a different saved torch.compile artifact than the two OFF boots, and R190c shows two fresh artifacts of one configuration disagreeing on 19 of 20 greedy continuations with a median delta of the same size as the ON-vs-OFF gap. The temperature-0 verdict above is therefore possibly an artifact effect rather than a sampler effect; R193 (`scripts/r193-determinism.sh`) repeats the comparison against a same-settings control.

Boot note for this image with the pcie_ipc all-reduce: the 13.98 GB KV pin booted once with 725 MiB free and then failed twice in the Triton warmup (CUDA invalid argument, not the headroom assertion); the 13.5 GB pin booted every time (pool 985,621 tokens, 1.5 to 1.9 GB free).
