# Correction (2026-07-19): util 0.98 retired after a serve-time autotune OOM, deep-concurrency numbers re-based

[← all results](../RESULTS.md)

The "Pool vs util" section above promoted util 0.98 on boot-margin and burst evidence. **This is superseded**: the first genuinely new deep batch shape (`pp8192 × c8`) makes the fp4-GEMM/FlashInfer autotuner allocate ~266 MiB of serve-time workspace (mnbt-4096 shapes, and ~486 MiB for 8192 shapes), which OOM-kills the engine at 0.98's ~600 MB margin, 2/2 reproducible and with zero warning in any boot-time probe. The daily became util 0.96, pool 270,422, validated against that deep concurrent shape and the full burst battery. `mnbt 8192` needs ≲0.94.

Deep-concurrency re-base (pp30000, util 0.96, `tg 512`): sustained aggregate c1 122 / c4 76 / c8 67, with **peak 510 (c4) / 604 (c8)** and ~135 t/s per stream during overlap. Sustained throughput is prefill-gated, not decode-gated: a cold 30K prefill takes ≈ 8.6 s of the shared ~3.5K t/s chunk lane and shadows all decoders down to ~1–5 t/s. Warm and prefix-cached fleets run at the peaks. `tg 128` deep cells (19–22 t/s "aggregate") measure only the prefill shadow, so the protocol now requires `tg ≥ 512` for steady state. Raw: `/srv/qwen5090/results/2026-07-18-mnbt-sweep/`.
