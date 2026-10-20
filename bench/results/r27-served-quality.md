# r27 served quality: tool-eval and GSM8K on the served configuration (2026-10-20, results `2026-10-20-172512-r27-served-quality`, `2026-10-20-173445-r27-served-quality`)

[← all results](../RESULTS.md)

2026-10-20 17:25 to 17:37 UTC, raw records in [`2026-10-20-r27-served-quality/`](2026-10-20-r27-served-quality/) (`run1` = results `2026-10-20-172512-r27-served-quality`, `run2` = results `2026-10-20-173445-r27-served-quality`), driver [`scripts/r27-served-quality.sh`](../../scripts/r27-served-quality.sh). Both runs went to the live served daily on port 8020, launcher [`scripts/serve-v031-served-daily.sh`](../../scripts/serve-v031-served-daily.sh): BA4, no block drop and the 15.90 GB KV pin ([r27-served-curves](r27-served-curves.md), [r27-kv-pin](r27-kv-pin.md)). No engine was booted for them, and other requests could share the GPUs during the runs.

The 2026-10-19 task checks in [r27 BA4](r27-ba4.md) were measured on the BA4 launcher before no block drop and the pin change. These runs repeat them on the served configuration.

## Tool-eval

[tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench), 69 scenarios × 4 trials, temperature 0.6, top-p 0.95, top-k 20, 8 in parallel, the invocation of the 2026-10-19 gate.

| run | score (mean ± stddev of 4 trials) | CI 95 | trial points of 138 |
| --- | ---: | --- | --- |
| run1 | 88.8 ± 2.5 | 86.8 to 91.0 | 127, 122, 119, 123 |
| run2 | 90.2 ± 1.0 | 89.5 to 91.0 | 123, 124, 126, 125 |
| both, 8 trials | 89.6 ± 1.8 | | |

The 2026-10-19 gate read 91.0 ± 1.4 on BA4 and 89.8 ± 1.9 on L1a, each one run of 4 trials on its own boot ([r27 BA4](r27-ba4.md)).

## GSM8K

lm-evaluation-harness 0.4.12, `gsm8k`, 5-shot, chat template, temperature 0, 200 problems, thinking off. The 2026-10-19 gate turned thinking off with the server flag `--default-chat-template-kwargs`; the served daily keeps thinking on, so these requests carry `chat_template_kwargs: {"enable_thinking": false, "preserve_thinking": true}` through `--gen_kwargs`. None of the 200 answers contains a reasoning block.

| filter | exact match |
| --- | ---: |
| flexible-extract | 0.895 ± 0.022 |
| strict-match | 0.890 ± 0.022 |

The 2026-10-19 gate read 0.905 on BA4 and 0.885 on L1a with flexible-extract.
