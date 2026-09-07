# Tool-eval cross-trial statistics: 69×4 on the tier daily (2026-07-22)

[← all results](../RESULTS.md)

The daily's standing quality number, re-measured with four full trials for error bars. Protocol: tool-eval (tool-calling benchmark, tool-eval-bench) v2.1.0, all 69 scenarios, temp 0.6 / top-p 0.95 / top-k 20, thinking on, parallel 8, which is the promotion-run protocol with the trial count doubled.

| metric | value |
|---|---|
| final score, per trial | 88 / 91 / 88 / 89 |
| **mean ± σ** | **89.0 ± 1.4** (95% CI 88.0–90.2) |
| Pass@4 (capability ceiling) | 85.5% |
| Pass⁴ (reliability floor: passes *every* trial) | 76.8% |
| reliability gap | 8.7 pp |
| deployability (α=0.7) | 82 — quality 89, responsiveness 64, median turn 2.0 s under parallel-8 load |

The tier daily holds **89.0 ± 1.4**. The 89.8 pooled plain-profile baseline sits inside the CI, so the six-patch tier stack still costs nothing measurable on quality. The 8.7 pp gap between Pass@4 and Pass⁴ is ordinary temp-0.6 flakiness spread over a handful of scenarios. The one systematic failure is TC-60, covered in the next section. The responsiveness 64 is a load artifact of the parallel-8 protocol: an earlier serial-protocol run scored 80 at 1.2 s median turn.
