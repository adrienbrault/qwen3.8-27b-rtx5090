# Previous generation: Qwen3.8-27B, NVFP4 KV + LMCache tiers, V2 model runner (daily 2026-08-21→28)

[← all results](../RESULTS.md)

Engine: saka W4A4 NVFP4 + `nvfp4` KV + FlashInfer FA2 + MTP `ns=4` + vision, V2 runner, LMCache 0.5.4rc4 (chunk 2864, mnbt 5727), util 0.93, max-len 200K, seqs 8, T=0.6, reasoning effort medium. Results dir `2026-08-21-qwen38-tiers-nvfp4kv` (R81).

| | c1 | c4 | c8 |
|---|---|---|---|
| decode, pp8192 tg512, aggregate t/s | 140.8 | 338.8 | 352.9 |
| decode, pp30000 tg512 | 142.4 | — | — |
| prefill, t/s | 12,850 @8K · 9,312 @30K | 12,763 @8K (aggregate) | 6,976 @8K (aggregate) |

Pool 309,090. Needles 9K / 20K / 40K / 60K / 100K × 2, cold + warm: 10/10 + 10/10; warm revisits 0.51–0.56 s @9K, 0.72–0.77 @20K, 0.90–0.96 @40K, 1.07–1.08 @60K, 1.20–1.99 @100K (cold 26.6–27.0 s). Restart-proof: store 40K / 60K (6.26 / 11.55 s), `docker restart`, revisit 2.47 / 3.25 s, correct. The "sean gate" (needles under concurrent loaders): 4 × 20K loaders, 32K / 48K / 64K × 5, cold + warm: 15/15 + 15/15. Tool-eval 69×2: 92 ± 1.4 (91 / 93). L2 stored 41 GB / 757 files during the audit.

## fp8 KV tiers on the V2 runner, same day (R79, results dir `2026-08-21-qwen38-tier-v2`)

Same stack with `fp8_e4m3` KV (chunk 1616, mnbt 3231, util 0.95): pool 209,859.

| | c1 | c2 | c4 | c8 | c1 pp30000 | c8 pp30000 |
|---|---|---|---|---|---|---|
| decode aggregate t/s (mean) | 152.2 | 248.8 | 359.9 | 367.1 | 143.0 | 134.3 |
| decode peak t/s | 181 | 349 | 643 | 1033 | 160 | 580 |

Prefill c1 9.7K @8K, 8.9K @30K, 8.5K @32K, 5.0K @100K. Needles 10/10 + 10/10, warm 0.5–1.5 s; restart-proof 40K 6.4 → 3.3 s, 60K 11.3 → 4.7 s; the "killer", a concurrent needle burst, 8×24K 8/8; tool-eval 69×2 91 ± 0.0, then 69×4 on the promoted daily 90.8 ± 0.5 (CI 90.2–91.0). The V1-runner daily measured the same hour: c1 128.3 / c4 309.2 / deep c1 118.9; the previous image on the V2 runner: 69×4 91.2 ± 2.1.

Peak vs mean: benchy sends cold prompts through one chunked prefill lane, so at c8 the eight 8K prefills serialize (about 7 s) and decode phases barely overlap at tg512. The peak column is the aggregate when all streams decode at once, and it scales near-linearly to c8. Per-request decode still drops with concurrency (119 → 94 t/s from c4 to c8) because 48 of 64 layers are GDN, whose decode cost is per sequence.

## Plain nvfp4 KV on the V2 runner (R80, results dir `2026-08-21-qwen38-nvfp4kv-v2`)

No tiers, util 0.93, mnbt 4096: pool 300,000 (338,636 at 0.95, where the FlashInfer autotuner OOM-falls-back). c8 × pp8192 354.9 t/s, c8 × pp4096 431–447, c4 @8K 276, c6 @8K 292, c8 @6K 315, c4 @12K 220. Needles 10/10 + 10/10 for `nvfp4` and `nvfp4_4over6`. Tool-eval 69×2: nvfp4 89 ± 1.4, nvfp4_4over6 88.5 ± 0.7. On the V1 runner (R77, results dir `2026-08-21-qwen38-nvfp4kv`) the same stack had a cliff at ≥50–57K prompt tokens in flight with MTP on (c8 × pp8192 159 t/s, 23 preemptions per 8 requests); it is gone on V2.

## DFlash2: rejected (R78, results dir `2026-08-21-qwen38-dflash2`)

`incoai/Qwen3.8-27B-DFlash2`, `ns=7`, fp8 KV, 60K max-len (62K is the ceiling on 32 GB), util 0.96: c1 164.5 (peak 239), c2 298.6, c4 267.0, c8 @4K 294.9, deep-30K c1 170.4; prefill 13.5–14.4K @8K; acceptance 2.60 per draft; tool-eval 91 ± 1.4; needles 6/6 + 6/6 to 40K. Autoregressive on the same image: c1 73.2 / c2 126.4 / c4 213.4. MTP `ns=4` on the same image and runner: c1 160.4 / c2 251 / c4 380, pool 166K vs DFlash2's 66K. Rejected. [../docs/archive/DFLASH2.md](../../docs/archive/DFLASH2.md).

## Controls on the plain 2026-08-21 nightly, MTP, fp8 KV, 200K / 0.98

V1 runner: pool 221,126, c1 131.6, c4 367.7, deep c1 128.5 (the 2026-08-15 nightly read 207,042 / 123.1 / 314.6 / 122.9). V2 runner: pool 235,211, then OOM inside the `fp4_gemm` autotuner on the first request at util 0.98.
