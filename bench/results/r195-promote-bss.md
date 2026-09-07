# R195: batch-sharded sampling promoted, gated on the serving port (2026-09-05 10:59 to 11:22 UTC, results `2026-09-05-r195-promote-bss`, [scripts/r195-promote-bss.sh](../../scripts/r195-promote-bss.sh))

[← all results](../RESULTS.md)

The served launcher now boots the image with patch 0147 and `--enable-batch-sharded-sampling`, and asserts the sampler's log line, the flag on the container and the patch marker at boot. The previous launcher is frozen as [scripts/serve-r189-daily.sh](../../scripts/serve-r189-daily.sh). The compile artifact changed with the image, since patch 0147 changes the configuration hash string: the engine loaded the artifact R193b's unsharded arm had compiled (nothing saved), a different draw of the compile lottery described in R193. Its position against the bf16 decode reference: median |Δlogprob| 0.00044 at ctx 0 (2 of 20 chunks agree in full) and 0.00805 at 30K (3 of 20), inside the day's spread.

Gates on the serving port: first-try boot at the 13.98 GB pin, pool 1,020,596, 599 MiB free; a short request 3.4% of the pool, a 100K prompt 11.7% (served from the kept disk tier in 2.5 s), five 100K contexts 59.6%, 0 preemptions; needle 4 of 4 cold and 4 of 4 tier re-asks; tool-eval 90 ± 0.8 over 69 × 4 (91.2 ± 1.3 and 90.5 ± 2.1 on the two previous days, the instrument's spread); 0 engine error lines.

Decode on the promoted configuration, derived steps/s (tokens per second divided by 1 + 9 × acceptance per draft token) against R189b's rows:

| probe | R189b t/s | R195 t/s | R189b steps/s | R195 steps/s |
|---|---|---|---|---|
| code, 1 stream | 333 @ 0.38 | 313 @ 0.358 | 75.4 | 74.2 |
| prose, 1 stream | 180 @ 0.154 | 171 @ 0.144 | 75.4 | 74.5 |
| prose at 30K | 160 @ 0.132 | 159 @ 0.137 | 73.0 | 71.5 |
| code, 8 streams | 1,308 @ 0.281 | 1,406 @ 0.303 | 371 | 377.6 |
| code, 16 streams | 1,870 @ 0.318 | 1,997 @ 0.329 | 484 | 504.1 |

Single-stream step rates are within the 2% run-to-run band; the single-stream tokens-per-second differences are the acceptance draw. At 8 and 16 streams the gain is the R193b/c/e size. Requests that pass a seed at temperature above 0 draw a different sample stream than the unsharded sampler did.
