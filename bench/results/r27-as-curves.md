# r27-as-curves: decode and prefill curves of the served v0.31.0 configuration, one boot

2026-10-16 20:32 to 20:49 UTC, results `2026-10-16-201838-r27-as-curves` (raw records in [`2026-10-16-r27-as-curves/`](2026-10-16-r27-as-curves/)), driver [`scripts/r27-as-curves.sh`](../../scripts/r27-as-curves.sh), figures drawn by [`bench/plot.py`](../plot.py).

The served launcher [`scripts/serve-v031-splitkv-daily.sh`](../../scripts/serve-v031-splitkv-daily.sh) (image `vllm-qwen38:v0310-r27c-A_S-splitkv`: vLLM v0.31.0, [patches-v0310/](../../patches-v0310/), FlashInfer 0.7.0.post1, nvidia/Qwen3.8-27B-NVFP4, NVFP4 KV at the 14.86 GB pin, MTP at 3 draft tokens, `pcie_ipc` all-reduce, batch-sharded sampling, split-KV override, V2 model runner), booted once on port 8020 at stock power limits (600 W / 575 W). The first boot attempt came up with a 1,391,795-token pool and 3,609 MiB free per card. Engine errors 0, preemptions 0, no GPU Xid. The memory clock offset was not recorded.

The instruments, shapes, run counts and seed scheme are R675's ([r675-27b-curves.md](r675-27b-curves.md)), so each point has an R675 counterpart.

## Decode against concurrency

`probes/decode_ss.py`, 1,024 forced tokens per stream, three runs per shape, the rate taken over the samples where every stream was decoding; median of the runs.

| streams | code, all streams | code, per stream | prose, all streams | prose, per stream |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 229.5 | 229.5 | 189.6 | 189.6 |
| 2 | 446.8 | 223.4 | 360.6 | 180.3 |
| 4 | 888.3 | 222.1 | 712.4 | 178.1 |
| 6 | 1,214.8 | 202.5 | 1,006.0 | 167.7 |
| 8 | 1,609.7 | 201.2 | 1,362.2 | 170.3 |
| 12 | 2,172.6 | 181.0 | 1,813.6 | 151.1 |
| 16 | 2,633.4 | 164.6 | 2,205.8 | 137.9 |

MTP accepted 0.63 to 0.67 of its draft tokens on code and 0.46 to 0.49 on prose at every concurrency (accepted over drafted tokens, from the server's counters). The spread of the three runs is widest at one and two streams on code (205.5 to 239.7 at one stream) and within ±3 % from 12 streams up.

Against R675 (2026-09-23, the same configuration on vLLM v0.29.0rc2) these read 8 % higher on code and 12 % higher on prose at one stream and 8 % and 6 % higher at 16 streams. The two runs are on different days with different seed prefixes and an unrecorded memory clock offset; the paired same-day screen of the two launchers ([r27-v031-splitkv.md](r27-v031-splitkv.md), results `2026-10-16-r27-audition-screen-065122-1708414`) reads the 4- and 8-stream aggregates within −0.7 % to +2.4 %, and is the comparison that holds.

## Cold prefill against prompt length

`probes/kv_capacity_probe.py --conc 1 --tokens 1`: one request at a time, one output token, prompt tokens as counted by the server, three prompts per length, each with a fresh seed so that neither the GPU prefix cache nor the CPU and disk tiers hold it.

| prompt tokens (mean of 3) | time to first token | prefill rate |
| ---: | ---: | ---: |
| 6,680 | 0.70 s | 9,542 t/s |
| 24,984 | 2.90 s | 8,615 t/s |
| 50,041 | 6.80 s | 7,359 t/s |
| 100,062 | 17.40 s | 5,751 t/s |
| 166,660 | 37.47 s | 4,448 t/s |
| 200,047 | 50.13 s | 3,990 t/s |

The three prompts at each length agree to 0.2 s. The rate falls with length because every full-attention layer reads the whole prefix for each chunk.

## Decode at depth, one stream

`decode_ss.py --conc 1 --ctx N` puts N/1.3 filler words in front of the prompt. The probe does not record the prompt's token count, so the figure plots these points against N, not against prompt tokens. Two runs per point. The 0 point is the one-stream row above.

| `--ctx` | code | prose |
| ---: | ---: | ---: |
| 0 | 229.5 | 189.6 |
| 30,000 | 226.6 | 181.1 |
| 60,000 | 227.9 | 170.9 |
| 120,000 | 211.1 | 165.8 |
| 200,000 | 193.3 | 151.2 |

Code decodes at the same rate to 60,000 and 16 % slower at 200,000; prose is 20 % slower at 200,000. Acceptance at depth reads 0.69 to 0.73 on code and 0.47 to 0.50 on prose.
