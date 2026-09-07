# R195b, prose decode at 8 and 16 streams on the served configuration (2026-09-05 11:42 to 11:43 UTC, results `2026-09-05-r195b-readme-prose`, [scripts/r195b-readme-prose.sh](../../scripts/r195b-readme-prose.sh))

[← all results](../RESULTS.md)

The README's 8-stream and 16-stream rows carried code prompts only. Same probe as R195 (`probes/decode_ss.py`, 1,024 output tokens, 2 runs), on the serving port of the R195 configuration, prose prompts. The step rates equal the code rows: at these batch sizes the step time does not depend on the content, and the drafter's acceptance alone sets the tokens per step. 0 engine error lines, 0 preemptions.

| probe | t/s aggregate | per stream | acceptance | steps/s |
|---|---|---|---|---|
| code, 8 streams (R195) | 1,406 | 176 | 0.303 | 378 |
| prose, 8 streams | 971 | 121 | 0.178 | 374 |
| code, 16 streams (R195) | 1,997 | 125 | 0.329 | 504 |
| prose, 16 streams | 1,319 | 82 | 0.180 | 504 |
