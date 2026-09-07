# R189b, the README decode rows on the served configuration (2026-09-05 09:37 to 09:39 UTC, `results/2026-09-05-r189b-readme-decode`, `scripts/r189b-readme-decode.sh`)

[← all results](../RESULTS.md)

Steady-state decode on the serving port of the promoted configuration, same probe and run counts as the 2026-09-04 rows (`probes/decode_ss.py`, 1,024 output tokens, 3 runs at 1 stream for code and prose, 2 for the rest). Code 1 stream 333 t/s at acceptance 0.38 (75.4 steps/s), prose 1 stream 180 t/s at 0.154 (75.4), prose at 30K context 160 t/s at 0.132 (73.0), code 8 streams 1,308 t/s aggregate at 0.281 (371 steps/s), code 16 streams 1,870 t/s aggregate, 117 per stream, at 0.318 (484 steps/s). Zero engine error lines, zero preemptions. The 8-stream tokens/s is lower than R189's 1,385 an hour earlier at the same step rate because that run drew acceptance 0.30; the step rate is the stable quantity and matches R189 (370) and the same-artifact controls of the day (369 to 371).
