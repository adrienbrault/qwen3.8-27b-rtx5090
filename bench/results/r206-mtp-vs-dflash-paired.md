# R206: MTP ns3 (0158 image) against the DFlash ns7 daily, paired (2026-09-06 20:59 to 21:52 UTC, results `2026-09-06-r206-mtp-decision` and `2026-09-06-r206b-prose-conc`)

[← all results](../RESULTS.md)

The decision run R198 could not be: with the prefix cache reopened by 0158, the MTP route is measured against the served daily in one unit, both arms booted the same hour at the 13.98 GB pin (`SEQS 16`, `PCIE_IPC=1`, `BSS=1`, `VLLM_TRITON_FORCE_FIRST_CONFIG=1`). DF = daily image on DFlash ns7. M3 = daily image + 0148 + 0152/0154/0155/0156 + 0158, `SPEC_METHOD=mtp SPEC_NS=3`, `VLLM_SM12X_PCIE_IPC_MTP=1`. Scripts: `scripts/r206-mtp-decision.sh`, `scripts/r206b-prose-conc.sh`.

| row | DF (DFlash ns7) | M3 (MTP ns3) | M3 vs DF |
|---|---:|---:|---:|
| KV pool (tokens) / block | 1,052,277 / 1,552 | 1,309,368 / 1,472 | +24.4 % |
| free VRAM after boot | 1,809 MiB | 4,033 MiB | |
| warm 32K revisit: tokens re-prefilled / ttft | 690 / 0.225 s | 1,934 / 0.485 s | +0.26 s per revisit |
| code c1, 3 runs (tok/s) | 288.6 | 220.7 | −23.5 % |
| prose c1 / prose c1 at 30K | 157.2 / 152.6 | 163.2 / 152.0 | +3.8 % / 0 |
| code c8 (R206 / R206b) | 1,516.8 / 1,549.7 | 1,542.8 / 1,565.0 | +1 to +2 % |
| prose c8 (R206b) | 1,114.3 | 1,252.3 | +12.4 % |
| code c16 | 2,396.5 | 2,668.2 | +11.3 % |
| prose c16 (R206b) | 1,701.6 | 2,099.3 | +23.4 % |
| tool-eval 69×4 | 91 (126/125/126/126) | 90 ± 1.4, CI [89.0, 91.2] (127/123/123/124) | inside the CI |
| needles 131K + 220K after a 16×90K flood | not re-run (R197: 4/4 + 4/4 from the tier) | 4/4 cold, 4/4 re-asks served from the tier in 1.5 to 2.6 s (129,536 to 217,856 tokens external) | |
| error lines / preemptions | 0 / 0 | 0 / 0 | |

The flood was raised from 12 to 16 prompts of 90K because a sequential 12×90K flood (1.08M tokens) cannot evict a needle from a 1.31M LRU pool; at 16×90K (1.44M) all four needles were evicted and came back through the tier, so the tier path is proven at needle depth on the MTP route.

An ns5 arm (the one-layer MTP head applied recursively) booted at pool 1,268,831 (two more GDN state copies per request than ns3) and lost on every row: code c8 1,459.3, code c16 2,409.4, prose c1 145.6, acceptance per draft position 0.49 / 0.53 / 0.27 against 0.65 / 0.70 / 0.43 at ns3. ns3 stays the MTP setting.

Fidelity was not re-run (0158 changes only which block is hashed; R205d's warm answers were token-exact with hits). The MTP route's R205 rulers stand: dense +0.827 % PPL, top-1 92.79 %, agentic +2.672 %, 95.63 %, against the daily's +0.667 % / 92.74 % and +2.748 % / 95.63 % (R188 control, measured a day earlier). That +0.16 pp sits at the edge of the R203 two-boot band, so the dumps were compared directly (`ladder_doc_compare.py`, 2026-09-06, dumps in `2026-09-06-r205-mtp-cache` and `2026-09-06-r203-spec-ladder`): MTP dense corpus PPL 5.3019 against R203 spec-ON 5.3062 (+0.082 %) and spec-OFF 5.3010 (−0.017 %), between the daily's own two boots (band 0.099 %); documents beyond ±2 % 70 and 75 of 693 against 92 for the ON-vs-OFF control; positions beyond 1 nat 1.29 % against 1.31 %. That is parity with no vllm#53488 signature on the MTP route. Carrying the 0158 image on the DFlash daily is unmeasured: 0154–0156 touch scheduler and cache-manager paths DFlash shares, so it needs a bitwise gate before it counts as free. The MTP chain has booted 5 of 5 times at the 13.98 GB pin (R205, R205c, R205d, R206, R206b).

The trade: MTP ns3 buys +24 % pool, +11 % code c16, +12 % prose c8, +23 % prose c16, the same tool-eval and fidelity band, for −24 % single-stream code decode and one more re-prefilled block on every warm revisit. Concurrent or multi-agent use favours MTP; single interactive coding sessions favour DFlash. Not promoted; the decision is the operator's. One oddity carried from R198: the 131K sample-0 needle answers with a spurious "BNBN " prefix on every MTP boot and never on DFlash; the answer is still correct.
