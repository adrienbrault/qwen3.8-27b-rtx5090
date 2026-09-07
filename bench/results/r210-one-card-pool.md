# R210: the served configuration on one card instead of two — the KV pool falls to 21.5% (2026-09-07 23:14 to 23:53 UTC, results `2026-09-08-r210e-tp1-pool`, [scripts/r210e-tp1-pool.sh](../../scripts/r210e-tp1-pool.sh))

[← all results](../RESULTS.md)

The question was what KV pool the served MTP ns3 configuration would get on a single RTX 5090. The answer is not half.

The arithmetic comes out of the two-card boot log before any measurement. The checkpoint is 21.81 GiB; model loading takes 10.19 GiB **per card** at two-way tensor parallelism, because the embedding table is offloaded to host RAM and everything else is sharded; and a 13.98 GB per-card KV pin yields 1,309,368 tokens. That is 10,677 B/token/card, so **21,354 B/token in total**. One card pays that full per-token cost out of a budget the unsharded weights have already eaten, which predicted 215,000 to 280,000 tokens and raised the possibility that a single card could not hold one full-length request.

Measured, at the largest pin that clears the 384 MiB post-warmup headroom floor:

| | two cards (served) | one card |
|---|---|---|
| KV pin | 13.98 GB per card | 6.0 GB |
| model loading | 10.19 GiB per card | 19.84 GiB |
| embedding offloaded to host | 1.18 GiB per rank | 2.37 GiB |
| KV pool | 1,309,368 tokens | **281,061 tokens** |
| free VRAM after pre-warm | 3,661 MiB | 807 MiB |
| maximum concurrency at 262,144 tokens | 4.99x | **1.07x** |
| attention block | 1,472 | 1,472 |
| mamba block | 16 | 1,472 |
| GPU blocks | 969 | 208 |

**One card serves a single full-length request and nothing beside it.** The pool is 21.5% of the two-card pool, so the second card is worth 4.7x the KV capacity rather than the 2x a naive reading suggests. Tensor parallelism shards both sides of the trade at once: the weights drop to 10.19 GiB per card instead of 19.84 GiB on one, freeing about 10 GiB per card for KV, *and* each card then stores only half of each token's KV.

A 6.5 GB pin also booted and reported 304,032 tokens, but left 235 MiB free against the 384 MiB floor, so it is not a usable setting. It is a useful check all the same: 6.5e9 / 304,032 = 21,379 B/token, **within 0.1% of the 21,354 B/token predicted from the two-card log**. The model of where the memory goes is exact.

The two-way control ran on the same code path in every re-run of this experiment and reported 1,309,368 tokens each time, matching the served configuration exactly.

## What it took to boot one card at all

The launcher had not booted a single card since this became a two-card machine, and its two-card assumptions are written as exact-value asserts rather than as ratios, so five units were needed before an arm measured what it claimed to. Recorded because each is a distinct failure mode a reader could hit:

1. The experiment pool band was 850,000 to 1,100,000 — narrower than the served pool of 1,309,368 — so a control at the *served* configuration was unbootable. An earlier note in the same file had already been bitten by that ceiling and widened the band only for arms using a different speculative method or length, missing the case where the experiment default *is* the daily.
2. The teardown helper ignored its settle argument and always used the 60 s default, so a requested 300 s settle never happened and the heavy two-card teardown transient recurred. It presents as `torch.full((1,), NULL_BLOCK_ID, dtype=torch.int32)` failing with `cudaErrorInvalidValue` inside the Qwen Triton warmup — a one-element int32 allocation, which cannot fail for a legitimate reason and so identifies a sick CUDA context rather than a memory problem.
3. **The tensor-parallel knob was read after the launcher's `unset` list**, which clears it on the experiment path as well as the daily one. `TP=1` silently read as 2 and an entire ladder booted two cards while reporting them as one: pool 608,065, which is exactly 6.5e9 / 10,677, the two-card per-token rate. Knobs that must survive that list now carry an `EXP_` prefix.
4. vLLM refuses `--enable-batch-sharded-sampling` at one-way tensor parallelism ("there is nothing to shard"), and the served configuration forces it on, so every experiment inheriting that shape carried it. The launcher now refuses the combination with a message naming the fix. Sharded sampling changes which tokens are sampled, not the cache layout, so the pool measured here is unaffected.
5. The offload assert expected the literal string `Total CPU offloaded parameters: 1.18` — the per-rank half of the embedding. That assert and the exact 1,472-token attention block are now conditional on two cards and log their observed value otherwise. The attention block turned out to be 1,472 on one card as well; only the offload figure actually changes.

The four superseded units were each chained through the GPU queue — start the successor, then stop the predecessor, so the predecessor's restore helper sees a live registration and skips the daily boot — so five re-runs cost no extra daily down and up cycles.
