# Native XQA improves deep steady decode; the six-agent replay remains tied

Measured 2026-10-21, results [`2026-10-21-v032-gpu`](2026-10-21-v032-gpu/), on the two RTX 5090s. The served 0.31 FA2/L1a/BA4 configuration is the control; the candidate uses native 0.32 NVFP4 writing/XQA with BA4 disabled. This compares whole configurations. The checkpoint, TP2 MTP3, bf16 recurrent state, no block drop, 15.90 GB KV pin, 16-sequence limit and stock power limits are matched. The candidate has a fresh KV/compilation directory. Both report a 1,490,437-token pool.

The existing steady-state decode probe runs code prompts at nominal contexts 0 and 60K, two runs per cell with 1,024 generated tokens. Reported rates cover the overlap where all requested streams are decoding; they exclude most prefill and ramp-up. The small sample and short overlap windows limit precision, and the candidate starts from a fresh compilation cache. The [raw records](2026-10-21-v032-gpu/) retain the per-run ranges.

| Nominal context | Streams | 0.31 aggregate tok/s | 0.32 aggregate tok/s | Change |
|---|---:|---:|---:|---:|
| Short | 1 | 284.2 | 242.3 | −14.8% |
| Short | 4 | 938.2 | 899.4 | −4.1% |
| Short | 16 | 2746.9 | 2794.0 | +1.7% |
| 60K | 1 | 268.0 | 249.6 | −6.8% |
| 60K | 4 | 744.0 | 867.3 | +16.6% |
| 60K | 6 | 945.4 | 1131.1 | +19.6% |

The six-agent replay uses fixed base contexts 30K/40K/50K/60K/80K/100K, three growing turns each, synthetic tool results, staggered arrivals and matched random seeds. Actual control prompts span 29,974–105,442 tokens. The generation budgets are fixed by the seed; generated text and subsequent conversation lengths can differ between engines. One replay per configuration completed all 18 requests without failures. Wall output rate includes prefill, tool delays and ramp-up/down; it is distinct from steady-state aggregate decode.

| Replay measurement | 0.31 | 0.32 |
|---|---:|---:|
| Request-span wall time | 81.23 s | 81.78 s |
| Wall output rate | 171.9 tok/s | 170.8 tok/s |
| Cold-turn median time to first token | 12.22 s | 12.01 s |
| Cold-turn median client decode rate | 16.7 tok/s | 18.2 tok/s |
| Longest streamed-response gap | 1.715 s | 1.710 s |
| Returning-turn median time to first token | 1.34 s | 0.98 s |

Response gaps are intervals between nonempty SSE frames; MTP can emit multiple tokens in one frame. During cold prefills the control's generation throughput fell to single digits with low KV usage and no preemptions. Together with the replay timings, this points to prefill interference as a cause of the interactive slowdown. Faster steady decode does not remove those pauses in this screen. A separate smaller-prefill-budget screen tests that mechanism.

The native candidate boots with XQA and PCIe collectives active, and passes arithmetic, automatic tool parsing and finite decode-logprob smoke checks. The first boot found a missing `functools.cache` import in the retained split-KV helper; it was fixed and a decorator-initialization regression case brings the Linux CPU suite to nine passing cases. The explicit 0.32 attention-backend CLI flag is also checked. Earlier CPU-only evidence describes the [earlier package snapshot](https://github.com/adrienbrault/qwen3.8-27b-rtx5090/tree/ada8b92/patches-v0320).

Greedy generations diverge in 9 of 12 probe chunks across the engines, including early differences. These are not established as harmless near ties. Native writing and prefill fusion change numerical execution; the smoke and throughput screen do not replace standard numerical qualification. BA4 remains disabled pending fresh admission, image-input compatibility is untested here, and the served daily remains 0.31. No promotion is justified by this screen alone.
