# R207: the MTP head promoted to the served configuration, gated on the serving port (2026-09-06 23:05 to 23:39 UTC, results `2026-09-06-r207-promote-mtp`, [scripts/r207-promote-mtp.sh](../../scripts/r207-promote-mtp.sh))

[← all results](../RESULTS.md)

The speculative route changed from the DFlash2 drafter at 7 draft tokens to the checkpoint's own MTP head at 3, on the image that carries patch 0148 (the sequential MTP drafter admitted on the `pcie_ipc` all-reduce), the R205 cache fixes and patch 0158 (the eagle block-drop replay boundary). The launcher is [scripts/serve-r207-mtp-daily.sh](../../scripts/serve-r207-mtp-daily.sh); the previous configuration is frozen as a rollback launcher on the host. Sixteen sequences, the same 13.98 GB pin, the same weights, KV dtype, tiers and kernels.

Every gate passed on the first boot at the table pin. The KV pool is 1,309,368 tokens on a 1,472-token attention block, against 1,052,277 on 1,552 for the drafter route, and 3,179 MiB of VRAM was free after pre-warm. The disk tier was wiped by the block-size stamp, as at every block change, so the needle gate ran fully cold.

| gate | reading |
|---|---|
| decode fidelity vs the bf16 model, 20 chunks | median abs delta-logprob 0.00056 at no context and 0.00592 at 30K, inside the 0.0051 to 0.0062 band of the previous route |
| 120K prompt, 5 concurrent | all five resident at 62.5% pool usage, no preemptions (the drafter route read 76.5% on the same probe) |
| needles at 131K and 220K | 4 of 4 answered cold on the wiped tier, then 4 of 4 answered again from the tier after a flood of 16 unrelated 90K prompts, in 1.6 to 2.8 seconds with 128,064 to 217,856 tokens read back |
| tool-eval, 69 scenarios x 4 trials | 91.2 +- 0.5, against 90.8 +- 1.0 for the drafter route |
| engine errors, preemptions | none |

The decode rows below are the first boot of each route on the serving port, same instrument and run counts, tokens per second aggregate.

| streams | drafter route, 7 draft tokens (2026-09-05) | MTP head, 3 draft tokens (2026-09-06) | change |
|---|---:|---:|---:|
| 1, code | 278.9 | 209.1 | -25.0% |
| 1, prose | 176.1 | 154.9 | -12.0% |
| 1, prose at 30K context | 150.8 | 148.6 | -1.5% |
| 8, code | 1,629.0 | 1,539.2 | -5.5% |
| 8, prose | 1,092.6 | 1,231.1 | +12.7% |
| 16, code | 2,361.4 | 2,633.7 | +11.5% |
| 16, prose | 1,700.3 | 2,145.1 | +26.2% |

Draft acceptance per token is 0.65 to 0.68 on code and 0.39 to 0.49 on prose, against 0.38 to 0.42 and 0.17 to 0.24 for the drafter route. The trade is the one the paired R206 run predicted: single-stream coding decode pays about a quarter, everything concurrent gains, and the pool gains 24%. One row disagrees with the paired reading, code at 8 streams, which was +1 to +2% in R206 and -5.5% here.

Two facts a reader reproducing this should know. The MTP route stores 595 blocks in the 16 GiB host tier where the drafter route stores 1,921, because an eagle block also carries the speculative state copies and is about 3.2 times larger; 595 blocks of 1,472 tokens is still 875K tokens of staging and served every needle. And above 40 sequences the launcher must cap `max_cudagraph_capture_size` at 320, because the MTP drafter validates its capture rows against the `pcie_ipc` slab, which holds 320 rows. Measured on the tier after the run, each MTP block file is 28,827,648 bytes for 1,472 tokens, or 19.6 KB per token, against 5.8 KB per token on the drafter route, so the 300 GB disk tier holds about 15M tokens instead of about 52M. Read volume per served token did not rise: the 131K needle came back on 2,529 MB for 129,536 external tokens here, against 2,968 MB for 130,368 on the drafter route.
