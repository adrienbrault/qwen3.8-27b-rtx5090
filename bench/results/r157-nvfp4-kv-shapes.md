# The nvfp4-KV shapes on the RedHat daily: pool doubles, single-stream decode drops (2026-09-02, `results/2026-09-02-r157-nvfp4-shapes`)

[← all results](../RESULTS.md)

Two TP=2 shapes with the 4-bit KV cache, both on the RedHat checkpoint, tier on, measured with the same instruments as the daily row (llama-benchy T=0.6 pp2048/tg256; `decode_ss` steady-state c8; needles at 9K/20K/131K). The DFlash2-on-nvfp4 shape is the R155 revival config (drafter unsharded, full MNBT, XQA off, which avoids the two known correctness bugs structurally; revival image with patches 0116–0119).

| RedHat, TP=2 | **fp8 KV + DFlash2 ns9 (daily)** | nvfp4 KV + MTP ns4 + XQA | nvfp4 KV + DFlash2 ns7 (draft_tp=1, XQA off) |
|---|---|---|---|
| KV pool @262K | 654,491 | **1,317,869** | 1,030,418 |
| needles | 9/9 | 6/6 | 6/6 |
| top-1 agreement with bf16 (agentic teacher-forced) | 95.95% | 95.57% | 95.57% |
| decode c1, natural | **318.8 t/s** | 188.9 (−41%) | 229.7 (−28%) |
| decode c8 code, steady state (acceptance/draft) | 1,212 (0.29) | **1,304 (0.57)** | 1,189 (0.38) |
| decode c8 prose, steady state | 925 (0.19) | **1,106 (0.44)** | 925 (0.25) |
| decode c1 @30K context | **157** | 139 | 132 |
| prefill pp2048 | 8,741 | 8,328 | 8,328 |

MTP+nvfp4 is the capacity shape, as it was on gittensor: twice the pool and the best batched throughput, because Qwen's MTP head accepts ~0.55 per draft token. It costs −41% single-stream, since it drafts its four tokens with four sequential head passes. DFlash2-on-nvfp4 sits between: c8 parity with the fp8 daily and a higher acceptance than fp8, so its −28% at c1 is forward-pass cost (FA2-over-nvfp4 verify batches plus the unsharded drafter), not draft quality. Both nvfp4 shapes prefill ~5% slower. **Where the −28% goes** (same night, `r157b-levers.sh` and `r157c-xqaverify.sh`): plain decode on 4-bit KV is free, at spec-off single-stream 108.9 t/s on nvfp4 (FA2 path, XQA off) against 105.5 on fp8, and tokens-per-step are equal (fp8 ns9 at 0.29 ≈ 3.6, nvfp4 ns7 at 0.38 ≈ 3.6). The gap is step time. Running the fp8 daily with the drafter unsharded (`draft_tensor_parallel_size=1`, the sharded-drafter bug's workaround) costs 11% at c1 and nothing at c8, and the remaining ~19% is the 8-row verify batches on the FA2-over-nvfp4 prefill path. Routing those through XQA (`VLLM_SM12X_XQA_VERIFY=1`, MNBT 4096 to stay clear of the max-len bug) is correct (needles 6/6) but 40% slower (c1 137, c8 code 852), so that route is out. What remains is kernel work in FlashInfer: the nvfp4 reader for the drafter's 4-heads-per-rank shape (worth ~11 points) and the sm120 nvfp4 multi-row prefill wrapper (worth ~19). The served daily stays fp8 DFlash2 for its interactive single-stream workload; the MTP+nvfp4 launch is one env change away (`TP=2 UTIL=0.90` on `serve-v0280-daily.sh`) for a batched or long-context job. Scripts: `scripts/r157-shapes.sh`.
