# R167, the embedding table moved to pinned host RAM: +9% KV pool for no measurable cost, on the rc1 image only, which turns out to decode 5x slower than v0.28 at 30K context with nvfp4 KV (2026-09-03, `results/2026-09-03-r167-embed`, `scripts/r167-embed-audition.sh`, `patches-v0290/0135-embed-uva-offload-v0290.diff`)

[← all results](../RESULTS.md)

**What was tried.** The model's input embedding table is 2.37 GiB of BF16 that is read once per token. A vLLM pull request (vllm#53981, see THIRD_PARTY.md) makes the existing UVA offloader reach it, so `--offload-backend uva --cpu-offload-gb 1 --cpu-offload-params embed_tokens` keeps each tensor-parallel shard (1.18 GiB) in pinned host memory and gathers rows over PCIe. Three arms on the v0.29.0rc1 chain, one hour, same box: nvfp4 KV without and with the offload, and fp8 KV with it. The launcher fails closed unless the engine log proves the offloader engaged and the offloaded size is the shard.

| | nvfp4 KV, control | nvfp4 KV + offload | fp8 KV + offload |
|---|---|---|---|
| KV pool (tokens) | 888,986 | **971,797 (+9.3%)** | 670,810 (+8.4% over rc1 fp8 without it) |
| needles 9K/131K, cold + warm | 8/8 | 8/8 | 8/8 |
| fidelity vs control (ΔNLL / top-1 agreement) | — | −0.07% / 0.934 | vs v0.28 fp8 daily: +0.04% / 0.936 |
| steady-state decode, code c1 / c8 | 227 / 1,051 | 223 / 1,065 | 250 / 1,075 |
| steady-state decode, prose 30K context | **29** | **26** | 135 |

**The offload is free.** The whole shard comes back as KV pool (+82,811 tokens at 15,466 bytes per token is 1.28 GB), and decode, fidelity and recall do not move. On a pool that is sized by utilization it grows on its own; a pinned pool has to be raised by hand.

**But the rc1 image has a problem of its own.** Both nvfp4 arms decode at 29 tok/s with 30K tokens of context, against 135 for fp8 in the same run and 145–157 for the v0.28 nvfp4 candidate. Short-context decode is normal, so it is the long-context attention decode path (the FlashInfer FA2 fallback that runs with XQA disabled) that got 5x slower between the two FlashInfer/torch generations. The earlier rc1 audition did not measure 30K decode, so this is new. Until it is understood, the offload cannot reach the nvfp4 daily through rc1; the choices are porting the patch back to the v0.28 chain or moving the fp8 daily to rc1 with the offload, where it would gain 2% of pool over today's daily.

**One more thing the run showed.** The second nvfp4 arm booted a fresh container over the tier the first arm had written, and its first request was served from the tier at 131K tokens (129,536 of 131,245 tokens external hits, 7.0 s against a 30.1 s cold prefill), correct 4/4. That is the first exact-match evidence of tier-served nvfp4 blocks, and it contradicts what the v0.28 image did an hour earlier (no first touch ever served). Whether that is the vLLM version or a container restart versus a fresh container is not isolated yet.
