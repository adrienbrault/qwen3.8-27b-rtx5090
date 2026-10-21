# Live concurrent agents, 2026-10-21

Results directory: [`2026-10-21-r27-live-agents`](2026-10-21-r27-live-agents/). These are passive observations of an existing workload, not a controlled benchmark. The sampler sends only GET requests to `/metrics` and reads GPU counters; it adds no inference requests. The served image was `vllm-qwen38:v0310-r27c-A_S-BA4-L1a-ubplan`, with MTP3, TP2, NVFP4 KV at the 15.90 GB pin, 16 sequences and a 262144-token context limit. Both cards were capped at 400 W. No experimental server was competing for the GPUs.

The raw [`passive.jsonl`](2026-10-21-r27-live-agents/passive.jsonl) contains 61 observations over 08:12:15–08:13:15 UTC on 2026-10-21. It contains no prompts, responses or client identifiers. The sampler is [`scripts/passive_serving.py`](../../scripts/passive_serving.py).

| Counter over the sampled minute | Observation |
|---|---:|
| Generated tokens | 29945 |
| Server-wide generation throughput | 499.1 tok/s |
| Running requests, minimum / mean / maximum | 6 / 6.95 / 7 |
| Waiting requests | 0 throughout |
| Preemptions | 0 |
| GPU KV occupancy, minimum / maximum | 63.2% / 75.7% |
| Prefix query / hit tokens | 933219 / 899392 |
| Token-weighted prefix hit rate | 96.4% |
| Drafted / accepted speculative tokens | 28077 / 20581 |
| Draft acceptance | 73.3% |
| GPU utilization | 99–100% |

These are server-wide counters across changing requests. They do not reconstruct individual stream rates. Prompt counters include cached tokens and do not advance at every prefill chunk, so counter windows cannot establish whether an engine step mixed prefill and decode. The observation shows no sustained admission queueing or KV preemption in this minute; it does not identify the responsible kernel or quantify prefill interference or power-cap effects.

The [2026-10-20 served curves](r27-served-curves.md), results `2026-10-20-141816-r27-served-curves`, measure concurrent short prompts and single-stream deep contexts. They do not measure several deep histories repeatedly receiving tool results. The live observation also uses lower power limits than those curves. A comparison between their rates is a workload-coverage warning, not a measured regression ratio. The next comparison needs deep concurrent decode and staggered tool-result prefills, with per-request TTFT, inter-token gaps, completion time and output/reasoning token counts.
