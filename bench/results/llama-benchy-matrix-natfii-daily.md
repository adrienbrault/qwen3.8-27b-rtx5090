# Complete llama-benchy matrix on the promoted daily (2026-07-19, util 0.98)

[← all results](../RESULTS.md)

One measurement pass on the live daily (`:8020`, natfii W4A4 + fp8 KV + MTP ns=4 + vision on), llama-benchy 0.3.8. Raw: `/srv/qwen5090/results/2026-07-19-natfii-daily-bench/`. This supersedes the candidate-phase (util 0.96) numbers where they differ, notably pp8192×c8 sustained, 326 → **466**; the earlier cell was measured mid-campaign against a cold autotune.

**Decode, tg128, 3 runs (aggregate t/s, peak in parens):**

| | c1 | c2 | c4 | c8 | c16 |
|---|---|---|---|---|---|
| pp512 | 116 | 213 | 358 | 706 (933) | 593 (898) |
| pp4096 | 126 | 204 | 280 | 352 (854) | — |

c16 = 2× `max-num-seqs`: it queues cleanly with no instability, active streams cap at 8, and the rest wait. The historical "MTP crashes c≥16" behavior predates PR #42603 and the seqs-8 cap.

**Sustained steady-state, tg512, 2 runs (aggregate t/s, peak in parens):**

| | c1 | c4 | c8 |
|---|---|---|---|
| pp512 | 125 | 422 (512) | 778 (961) |
| pp4096 | 127 | 369 (495) | 605 (950) |
| pp8192 | 114 | 308 (533) | 466 (925) |
| pp30000 | 125 | 164 (481) | 149 (582) |
| pp90000 | — | 39 (263) | — |

The pp30000/pp90000 rows are prefill-lane arithmetic, not decode capability: per-stream decode peaks stay 128–136 t/s at every depth once prefills drain. The sustained aggregate is the cold-prefill shadow (four 90K contexts = 360K tokens through a ~5.3K t/s deep lane, worst TTFT ~56 s). Warm and prefix-cached fleets run at the peaks column.

**Prefill lane (aggregate, flat with concurrency):** pp8192: 13,315 / 13,577 / 13,347 (c1/c4/c8) · pp30000: 10,117 / 10,001 / 9,878 · pp90000: 5,288 (c4).

**Long context c1 (prefill / e2e TTFT / decode, tg128):** 30K: 10,167 / 2.7 s / 136 · 90K: 5,780 / 14.1 s / 140 · 180K: 3,472 / 47.0 s / 138.
