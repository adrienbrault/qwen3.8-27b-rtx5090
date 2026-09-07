# R193e: the pin on three boots, and batch-sharded sampling under the protocol (2026-09-05, results `2026-09-05-r193e-pin-bss`)

[← all results](../RESULTS.md)

Two more engines on R193d's compile artifact (all six AOT entries loaded, none saved) with `VLLM_TRITON_FORCE_FIRST_CONFIG=1`: P3 with sharded sampling off, B3 with `--enable-batch-sharded-sampling`. P3 against R193d's P1 agrees on 20 of 20 chunks with median 0 at ctx 0 and at 30K, so the pin now holds across three boots on one artifact. B3 against P3, and B3 against P1, agree on 20 of 20 with median 0 at both contexts: the sharded sampler is numerically inert under the protocol, the third independent bitwise pair across modes (after R193 and R193c). Both arms sit where P1 sat against bf16 (ctx 0 median 0.00062, 30K 0.00818), the far end of the day's 30K spread, so the knob is a measuring tool, not a fidelity setting.

Derived steps/s (tokens per second divided by 1 + 9 × acceptance per draft token, two runs each, code):

| arm | c1 | c8 | c16 |
|---|---|---|---|
| P3 (sharded off) | 72.6 | 370.0 | 484.2 |
| B3 (sharded on) | 73.7 | 379.6 | 506.0 |
| change | +1.5 % | +2.6 % | +4.5 % |

Same size as R193b and R193c. Sharded sampling is not in the served configuration; adopting it needs patch 0147 in the served image and the capacity, needle and tool-call gates on the served port.
