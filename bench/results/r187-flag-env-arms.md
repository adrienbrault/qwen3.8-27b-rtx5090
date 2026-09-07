# R187, flag and environment arms on the served image: none beats the same-configuration band; batch-sharded sampling raises the step rate 3.5% at c16 but changes the sampled tokens; the 30K to 100K decode tax is 6%, all attention (2026-09-05 02:37 to 03:34 UTC, `results/2026-09-05-r187-flags`, `scripts/r187-flags.sh`)

[← all results](../RESULTS.md)

Nine boots of the served configuration minus the pcie_ipc layer (image `...-fi0616`, pin 13.98 GB, 16 sequences, port 8029), each measured with the steady-state decode probe (code c1, prose c1, code c8, code c16; two runs at c1 and c8, one at c16). Two identical BASE boots bracket the battery: code c1 307.7 and 307.8 t/s, prose c1 158.7 and 158.3, code c8 1,407 and 1,399, code c16 1,729 and 1,774, with acceptance per draft token identical to the digit (0.3635 code, 0.1335 prose), so the seeds reproduce. Step rate is tokens/s divided by (1 + 9 × acceptance).

| arm | code c1 t/s | prose c1 | code c8 | code c16 | steps/s c1 / c8 / c16 | note |
|---|---|---|---|---|---|---|
| BASE-a / BASE-b | 307.7 / 307.8 | 158.7 / 158.3 | 1,407 / 1,399 | 1,729 / 1,774 | 72.0 / 354 / 455 | acceptance 0.3635 code, 0.1335 prose |
| OMP_NUM_THREADS=1 | 303.4 | 157.6 | 1,403 | 1,801 | 70.9 / 354 / 455 | same acceptance as BASE |
| OMP_NUM_THREADS=2 | 308.1 | 159.0 | 1,382 | 1,824 | 72.2 / 354 / 454 | same acceptance as BASE |
| cpuset 0-7 | 308.2 | 158.6 | 1,368 | 1,793 | 72.2 / 354 / 456 | same acceptance as BASE |
| batch-sharded sampling | 245.8 (190 to 301) | 170.9 | 1,338 | 1,719 | 70.6 / 359 / 470 | acceptance 0.276 / 0.303 / 0.295 code, 0.156 prose |
| max-num-batched-tokens 4,096 | 289.8 | 171.6 | 1,326 | 1,786 | 69.9 / 353 / 454 | pool 1,027,121, 3,169 MiB free |
| max-num-batched-tokens 12,288 | 261.1 | 165.4 | 1,295 | 1,741 | 71.6 / 351 / 452 | pool 1,014,153, 435 MiB free (below the served floor) |

The three CPU arms leave the GPU work untouched (their acceptance equals the base seeds') and their c16 reads fall inside the single-run spread of R183; CPU contention is not a lever on this host. Batch-sharded sampling is the only arm outside the band in step rate, +1.6% at c8 and +3.5% at c16, which is the size expected from halving the one 19.9 MB logits all-gather per step (R190 audit, NOTES26); it also samples different tokens for the same seeds everywhere, so tokens/s fell 20% at code c1 while the step rate rose. That is a numerics question to settle with a temperature-0 equivalence check and the decode ruler before the step rate counts (R191). Both batched-token arms change the prompt's prefill chunking and therefore the generated text, and neither moves the step rate; 8,192 stays. Clocks held at 2.77 to 2.87 GHz on every arm with only software power-cap throttle reasons on at most 12% of samples.

Deep-context profile (torch profiler, 50 delay and 60 captured iterations, prose c1): at 30,000 tokens of context the decode step is 17.11 ms with 13.00 ms of GPU work (GEMM 6.69, custom all-reduce 1.75, attention 0.90 over 46 calls, drafter 0.83, elementwise 0.65, GDN 0.63, fused Triton 0.57, FP4 activation quant 0.30); at 100,000 tokens it is 18.19 ms with 14.17 ms busy, attention 1.96 ms (107 µs per call), every other bucket within 0.08 ms of the 30K read. The long-context decode tax between 30K and 100K is therefore 1.06 ms per step, 6%, entirely in the attention kernel; the 4.0 to 4.1 ms of no-kernel time per step is the same at both depths. The battery's first attempt (02:27) failed at boot because the launcher's experiment default for the pcie_ipc knob read its own value; fixed in `scripts/serve-r168-daily.sh` and re-run.
