# Dynamic MTP 3/2/1 reduces concurrent throughput

Measured 2026-10-21, results [`2026-10-21-v032-dynamic-mtp`](2026-10-21-v032-dynamic-mtp/). The tested schedule is rejected. Fixed MTP depth 3 remains the default; the served 0.31 configuration is unchanged.

The candidate uses the same native vLLM 0.32 image as the [earlier screen](v032-gpu-screen.md), image ID `sha256:0ae66d77cb49ad12ac134dd2b5d8ae63b16ed35ca933a48500d5912e4e4b2c93`. Both arms use TP2, BA4 off, prefill budget 8192, the same checkpoint, 15.90 GB KV pin per card, 16 sequences and the same power limits. The completed fixed-K3 control from that screen is reused. The only candidate configuration change is `num_speculative_tokens_per_batch_size=[[1,2,3],[3,4,2],[5,16,1]]`: K3 at scheduled batch sizes 1–2, K2 at 3–4, K1 at 5–16. The runtime-K handling credited in [THIRD_PARTY.md](../../THIRD_PARTY.md) makes this possible.

## Steady decode falls 21–33% at concurrent loads

`decode_ss.py` measures aggregate steady output tokens/s during two 1024-token code generations per cell. Short and nominal 60K prompts use the same seeds as the control. Runtime proposal counters verify that the requested draft depths are actually used.

| Context | Streams | Fixed K3 tok/s | Dynamic tok/s | Change | Observed dynamic K |
|---|---:|---:|---:|---:|---:|
| Short | 1 | 242.2 | 238.3 | -1.6% | 3 |
| Short | 4 | 899.4 | 635.4 | -29.4% | 2 |
| Short | 16 | 2793.9 | 2199.1 | -21.3% | 1 |
| 60K | 1 | 249.6 | 233.9 | -6.3% | 3 |
| 60K | 4 | 867.3 | 635.4 | -26.7% | 2 |
| 60K | 6 | 1131.1 | 760.9 | -32.7% | 1 |

The accepted fraction of proposed tokens increases with shorter drafts, but throughput falls. This supports keeping the longer draft on this workload: reduced drafting work does not compensate for fewer accepted tokens per verification. The single-stream cells retain K3 and provide a boot/run variation check. This is one candidate boot with two runs per cell, sufficient to reject the measured policy; it does not establish that every dynamic schedule is slower.

## Agent replay takes 5.0% longer

All 18 requests complete without failures in both arms of `agent_replay.py`, using six agents with staggered tool returns and roughly 30K–106K prompts. The request-span time rises from 81.78 to 85.86 seconds. Cold median time to first token is 12.014 versus 12.023 seconds; returning median is 0.983 seconds in both. The longest cold SSE frame gap is 1.710 versus 1.707 seconds. The median of cold requests' p95 frame gaps falls from 1.054 to 0.860 seconds, but the longest stalls remain and completion slows. SSE gaps are frame gaps; an MTP frame can contain several tokens.

## Qualification remains incomplete

Arithmetic, automatic-tool parsing and finite-logprob smoke pass, with the unchanged 1,490,437-token pool. Ten Linux CPU cases and the actual CLI parser check pass; their logs are in the results directory. The pinned scheduler lookup is checked for every batch size from 1 to 16.

Greedy text differs in 9/12 chunks between these boots. The greedy probes run one request at a time, where this schedule still uses K3; the preceding native screen already showed variation between fresh compile/autotune boots. These differences cannot be attributed to dynamic depth from this comparison and are not established as harmless. The candidate has no numerical qualification or promotion approval.
