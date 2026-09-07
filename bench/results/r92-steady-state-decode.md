# Steady-state decode of the gittensor daily (2026-08-23, `results/2026-08-23-r92-daily-perf`)

[← all results](../RESULTS.md)

Method (`scripts/decode_ss.py`): c concurrent generations with `min_tokens = max_tokens = 1024` and short prompts, vLLM `/metrics` sampled every 0.5 s, throughput taken only over samples where `num_requests_running == c`; MTP acceptance from the same counters; median of 3 runs (min–max in brackets). Cross-check: `vllm bench serve --backend vllm --endpoint /v1/completions --dataset-name random --random-input-len 1024 --random-output-len 512 --num-prompts 48 --max-concurrency 4 --ignore-eos`.

| | c1 | c2 | c4 | c8 | c1 @30K | c1 @100K |
|---|---|---|---|---|---|---|
| prose, aggregate t/s | 124 (123–149) | 270 (265–281) | 511 (505–538) | 891 (888–913) | 115 (109–120) | 103 (102–105) |
| prose, accept / draft token | 0.32 | 0.38 | 0.38 | 0.37 | 0.31 | 0.32 |
| code, aggregate t/s | 183 (155–197) | — | 639 (628–655) | — | — | — |
| code, accept / draft token | 0.61 | — | 0.54 | — | — | — |
| vllm bench serve (random tokens) | — | — | 602; TPOT 5.1 ms median / 11.6 ms p99; TTFT 221 ms | — | — | — |
| llama-benchy pp8192 tg512 (R90), mean / peak | 170 / 194 | 272 / 368 | 451 / 737 | 487 / 1198 | 187 | 193 |

Every benchy "aggregate" row elsewhere in this file is a wall-clock mean over a window dominated by the prefill ramp, and it under-reads concurrent decode by 10–45%. Its peak column is the steady state. The 187/193 "decode rises with depth" in the R90 row is MTP acceptance on benchy's repetitive filler, not a property of the engine: on prose, depth costs ~17% at 100K.
