# Nightly re-platform measurements: decode, quality ladder and DSpark (2026-08-15, vLLM 0.27.2rc1.dev77, saka Qwen3.8, plain, util 0.98/200K)

[← all results](../RESULTS.md)

Decode (pp8192, tg512): c1 **123.1** / c2 207.1 / c4 314.6 / c8 321.7 tok/s aggregate. The 0.23 tier daily measured 84.5 / 134.5 / 215.4 / 255.6 on the same protocol; the delta is the nightly's spec path plus T=0.6, and the earlier 84.5 also ran with a T=1.0 boot. Prefill c1: 12,972 @8K / 9,690 @32K / 5,048 @100K tok/s. Deep-concurrent pp30000×c8: 122.9. Pool 207,042. MTP acceptance 60–65% (accept-len 3.4–3.6).

Quality ladder (69×2 @T0.6): async ON no-align 87 ± 1.4, async OFF no-align 88.5 ± 0.7, **async OFF + `--mamba-cache-mode align` 91 ± 0.0**, align + async ON 90 ± 1.4. Alignment of the GDN state cache with spec decode matters. Async costs about 1pt.

DSpark (vLLM-native, RadixArk draft, block 7, fixed verify): c1 essay 138 / c1 code 174 / c2 229 / c4 442 aggregate, which beats MTP at 20–22% draft acceptance (the draft was trained for the FP8 target). Draft KV limits context to ~64K (pool 84,292 @0.98). Adaptive verification was rejected on the GDN backend. Quick-eval 93.

Caveat: these are 2-trial pairs, not the 4-trial CI protocol. SWE-Bench and TB numbers for this base do not exist yet.
