# Promotion (2026-07-19): natfii NVFP4 W4A4 is the daily, util 0.98, pool 239,436

[← all results](../RESULTS.md)

All numbers measured on the promoted config (natfii W4A4 + fp8_e4m3 KV + FlashInfer 0.6.15 + MTP ns=4 + vision, `mnbt` 4096, `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=134217728`), llama-benchy 0.3.8, raw output in `/srv/qwen5090/results/2026-07-19-natfii-daily-bench/` and `/srv/qwen5090/results/2026-07-19-natfii-ceiling/`.

**Decode (tg128 total t/s):** pp512 116/213/358/706 (c1/c2/c4/c8, peak 933) · pp4096 126/204/280/352 (peak 854).

**Sustained deep concurrency (tg512 aggregate, c8):** pp512 769 · pp8192 326 · pp30000 148, against the AR daily's 604/225/67 on the identical protocol. The gap is the prefill lane: W4A4 widened it ~3.4×.

**Prefill lane (aggregate, flat with concurrency):** pp8192 13,315 (c1) / 13,577 (c4) / 13,347 (c8) · pp30000 10,117 / 10,001 / 9,878. Per-request throughput divides by N, and the queue drains 3× faster (c8×30K worst-case TTFT ~12.3 ± 6.3 s, previously ~30 ± 19 s).

**Long context c1 (prefill / e2e TTFT / decode):** 30K: 10,167 / 2.7 s / 136 · 90K: 5,780 / 14.1 s / 140 · 180K: 3,472 / 47.0 s / 138. Decode is flat with depth. The prefill advantage narrows with depth, because attention's O(n²) share is not FP4, but it never inverts.

**Quality (tool-eval-bench, full 69×2):** natfii pooled 89.8 over 4 independent trials against AR 87.8 (4 trials), which is parity within noise. The W4A4-activation cost was bounded at ≈1 pt by a chimera A/B (natfii MLPs + NVIDIA fp8 attention, one merged checkpoint: 90.0; NVIDIA W4A16: 91.0). The quick-15 subset has a ±7 noise band (106-sample distribution, median 93), so promotions are scored on the full suite only.

**Util ceiling (this model):** 0.98 = 239,436 tok, boot free ~1.4 GiB, steady-state floor ~130–190 MiB after autotune workspaces allocate. The battery was: needle (60K), pp8192×c8, pp30000×c8, pp512×c8 tg512, 8× distinct ~34K floods, 8× 4-image vision bursts, then two simultaneous combined waves (16 requests + benchy) on a cold engine, with zero crash signatures, plus a 106-cycle overnight soak. 0.96 = 222,535 (validated fallback). The previous daily's 0.98 serve-time OOM does not reproduce here, thanks to smaller margin pressure, a 128 MiB workspace cap and boot pre-warm: the ceiling is model-specific.
