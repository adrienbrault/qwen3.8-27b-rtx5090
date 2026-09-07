# R200 and R200b: decode and prefill at 32 and 64 streams, and the per-request cost of the speculative state ring (2026-09-05 19:54 to 20:50 UTC, `results/2026-09-05-r200-c32c64`, `results/2026-09-05-r200b-pool-cost`, [scripts/r200-c32c64.sh](../../scripts/r200-c32c64.sh), [scripts/r200b-pool-cost.sh](../../scripts/r200b-pool-cost.sh), [scripts/pool_cost_probe.py](../../scripts/pool_cost_probe.py))

[← all results](../RESULTS.md)

The served configuration was booted on the experiment port at 32 and at 64 sequences, same image, flags and 13.98 GB pin. Decode with the steady-state probe, 1,024 tokens, two runs each:

| boot | streams | code t/s aggregate (per stream) | prose t/s aggregate (per stream) | accepted per draft, code / prose |
|---|---|---|---|---|
| 32 sequences | 16 | 2,396 (150) | 1,645 (103) | 0.40 / 0.23 |
| 32 sequences | 32 | 2,758 (86) | 1,973 (62) | 0.39 / 0.24 |
| 64 sequences | 32 | 2,743 (86) | 1,935 (61) | 0.39 / 0.23 |
| 64 sequences | 64 | no steady state | no steady state | |

The 16-stream rows on the 32-sequence boot match the served rows within 3 %, so a larger sequence limit costs nothing by itself. Going from 16 to 32 streams adds 15 to 20 % aggregate throughput and halves the per-stream speed. At 64 streams the engine ran at most 36 requests and queued the other 28 ("Running: 36 reqs, Waiting: 28 reqs" in the engine log), so the 64-stream rows do not exist.

Prefill with llama-benchy (prompt 2,048 or 8,192 tokens, 32 generated, temperature 0.6, no cache) is flat in concurrency: 9,085, 9,088 and 9,015 t/s at 16, 32 and 64 streams for the 2K prompt, 8,905, 8,911 and 9,083 for the 8K prompt, against 8.8K t/s for a single request. The engine prefills one chunk at a time; concurrency only lengthens the queue (first response after 4.5 s at 32 streams with 2K prompts, 30.5 s at 64 streams with 8K prompts).

Why 36: `pool_cost_probe.py` sends identical short generations and reads `kv_cache_usage_perc` while all of them run. A request takes a fixed share of the pool as soon as it is scheduled, unchanged over 1,200 generated tokens. On the served configuration (7 draft tokens) that share is 2.72 % of the 1,052,277-token pool, 28,613 tokens-equivalent, so 36 requests fill it. The same probe on one boot per arm at 16 sequences:

| arm | pool tokens | attention block | tokens-equivalent per request | requests that fit |
|---|---|---|---|---|
| DFlash, 1 draft token | 1,138,434 | 1,440 | 8,448 | 134 |
| DFlash, 3 draft tokens | 1,108,062 | 1,472 | 15,128 | 73 |
| DFlash, 7 draft tokens (served) | 1,052,277 | 1,552 | 28,613 | 36 |
| DFlash, 7 draft tokens, fp32 state | 945,307 | 2,912 | 48,259 | 19 |
| MTP head, 3 draft tokens | 1,309,368 | 1,472 | 17,584 | 74 |

The cost is one attention block plus (1 + draft tokens) × about 3,355 tokens-equivalent, which at 13.3 KB per token is 44.6 MB per GPU per copy: the bf16 GDN state of one request split across the two cards. vLLM's hybrid KV manager (`v1/core/single_type_kv_cache_manager.py`, `mamba_utils.py`) allocates `1 + num_spec_tokens` state blocks per request in `align` mode and keeps the state after every draft position, so that accepting k tokens selects the k-th stored state instead of recomputing it. With 7 draft tokens and 16 requests running, 5.0 GB of the 13.98 GB pool holds rollback states, 376K tokens or 36 % of the pool. Recomputing the accepted state from the pre-draft state and the accepted tokens would free that space; it is an open item, not a change.
