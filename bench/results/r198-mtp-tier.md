# R198: the MTP head on this hybrid pays full prefill for every revisit and eviction, so the tiers never serve it (2026-09-05 15:09 to 15:33 UTC, results `2026-09-05-r198-mtp-tier`, `scripts/r198-mtp-tier.sh`)

[← all results](../RESULTS.md)

Why: the R197 ladder left MTP with three draft tokens on the `pcie_ipc` all-reduce (patch 0148) as the fastest 8- and 16-stream arm and the best 30K single stream, with a 24% larger pool, and one open question. Every MTP boot on this model prints that no KV-cache group can be identified as the drafter's, so every group, the three Mamba groups included, is treated as a draft group, prefix-cache reuse across requests is disabled, and an external KV tier stores without ever serving a hit. This run measures that sentence on port 8029 with the same boot as the ladder (13.98 GB pin, pool 1,309,368, attention block 1,472 at three draft tokens, drafter admitted to the `pcie_ipc` all-reduce), reading the engine's prefix-cache and tier counters from `/metrics` between steps, and running the same needle gate the seven-draft-token daily passed on port 8020 at 15:04 UTC (R197).

| step | MTP, 3 draft tokens | DFlash2, 7 draft tokens (served, same gate) |
|---|---|---|
| one 120K prompt, cold | 19.2 s prefill, 0 prefix hits | 18.5 s |
| the same 120K prompt again | 19.2 s, 0 prefix hits | evicted 131K served in 1.2 s |
| needles at 131K and 220K, cold | 4/4 hits, 26.9 s and 60.5 s | 4/4, 25.6 s and 56.7 s |
| the same needles after a 12 × 90K eviction flood | 4/4 hits, 27.1 s and 60.6 s, 0 tokens external, tier served 0 of 4 | 4/4, 1.2 s and 2.2 s, 130,368 and 218,832 tokens external, tier served 4 of 4 |
| bytes to the host tier / back from it | 102.5 GB / 0 | round-trips |
| prefix-cache hits over the run | 0 until the flood, then 693,312 of 6.74M queries inside it (shared chunks within a batch, not reuse across requests) | reuse across requests |

Readings. The answers are right; the cost is that every repeated or evicted prefix is recomputed in full, and revisits of evicted context are the workload the host and disk tiers exist for. The three-token MTP configuration keeps its pool and batched-decode advantage from the ladder but is not a candidate for the served configuration until the engine can name the drafter's KV-cache group on a Mamba hybrid. One observation without an explanation: one of the two 131K needle samples answered with a spurious "BNBN " prefix on both its cold and its evicted ask under MTP; the same sample answered cleanly on the served configuration an hour earlier.
