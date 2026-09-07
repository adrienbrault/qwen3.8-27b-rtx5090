# R205d: the MTP prefix cache reopened live (2026-09-06 20:10 to 20:47 UTC, results `2026-09-06-r205d-eagleshift-gate`, [scripts/build-r205d-eagle-shift.sh](../../scripts/build-r205d-eagle-shift.sh), [scripts/r205d-eagleshift-gate.sh](../../scripts/r205d-eagleshift-gate.sh), layer `patches-v0290/Dockerfile.mtp-eagle-shift`, patch 0158)

[← all results](../RESULTS.md)

The image of R205 plus patch 0158 (R205c below) on the MTP route (draft length 3, `pcie_ipc` all-reduce with the MTP drafter admitted, 16 sequences, 13.98 GB pin, evaluation tier). Boot: pool 1,309,368 tokens, block 1,472, 3,541 MiB free, the 0158 proof line once ("kv cache groups [0, 1, 2] keep the state one block before the dropped block, retention_interval=0, scheduler_block_size=1472"), 0 engine error lines, 0 preemptions.

| probe | R205 (without 0158) | R205d (with 0158) |
|---|---|---|
| one 6,000-token prompt sent three times (completions, 8 tokens, no gap) | hits 0 / 0 / 4,416 | hits 0 / 4,416 / 4,416; time to first token 0.70 s then 0.20 s |
| warm_equal 6K, greedy cold vs warm, 3 prompts | equal 3/3 with 0 hits | equal 3/3 with 4,416 hits each (cold 0.95 s, warm 0.46 s) |
| warm revisit, 32K prompt | 7.8 s second send, 0 hits | 7.79 s then 0.48 s, 51,520 of 53,456 tokens hit |
| revisit after a 12 × 90K flood | 7.8 s, 0 hits | 0.68 s, 51,520 tokens served through the offloading connector |
| warm_equal after the flood | equal 3/3 with 0 hits | equal 3/3 with hits |
| needle gate, 131K and 220K, cold then two re-asks after the flood | found 4/4; first re-asks recomputed (27 s, 60 s) | found 4/4; every re-ask 0.96 to 1.76 s, served from the GPU (129,536 / 217,856 tokens hit) |
| prefix hits over the unit | 4,416 | 1,473,472 |
| decode, code 8 streams / prose 1 stream | 1,502 / 165.1 | 1,544 / 163.0 |

The needle gate's tier counter stayed at 0 here because the flood is sequential (12 chat prompts of 90K tokens one after another, filler calibrated on the served tokenizer) and the pool is LRU: at each re-ask the pool holds the needle plus 1.08M flood tokens, 1.21M to 1.30M, under the 1.31M pool, so the needle's blocks are never the oldest (logged GPU KV usage peaked at 17.0 %, one prompt at a time; prefix hits rose by 1,386,624 over the gate = exactly the eight re-asks, the flood itself hit nothing). The flood is sized for the served pool of 1,052,277 and has to grow to at least 16 × 90K on this route. The 32K row after the flood is the tier proof on this route (GPU hits 0, connector hits 51,520). The MTP route loses one more block than DFlash on every revisit (1,936 tokens re-prefilled instead of 685) because the FullAttn group drops its last matching block by design.

What this changes. R197 and R198 set the MTP route aside partly because every revisit and every evicted re-ask paid full prefill. With 0158 the route hits the cache and the tiers like the served DFlash configuration, and the rest of the comparison stands as measured: pool 1,309,368 against 1,052,277 (+24 %), request ceiling 74 against 36 from the speculative-state ring (R200b), code at 16 streams +7 %, prose at 1 stream +4 %, code at 8 streams −11 %, fidelity in the same band (R205). Not yet measured on the MTP route since the fix: code at 1 stream, tool-eval, the agentic ruler, boot reliability at the pin on the longer image chain, and a pool-aware draft-length ladder (draft length 3 was chosen while the cache was dead). The served configuration is unchanged (DFlash never drops a block, so 0158 is inert there).
