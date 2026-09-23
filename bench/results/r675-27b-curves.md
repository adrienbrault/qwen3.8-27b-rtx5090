# R675: decode and prefill curves of the served configuration, one boot

2026-09-23 11:33 to 11:51 UTC, results `2026-09-23-r675-27b-curves` (raw records in [`2026-09-23-r675-27b-curves/`](2026-09-23-r675-27b-curves/)), driver [`scripts/r675-27b-curves.sh`](../../scripts/r675-27b-curves.sh), figures drawn by [`bench/plot.py`](../plot.py).

The served launcher (R231/R234 configuration: nvidia/Qwen3.8-27B-NVFP4, NVFP4 KV at the 14.86 GB pin, MTP at 3 draft tokens, `pcie_ipc` all-reduce, batch-sharded sampling), image `vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc-bsshash-mtppcie-mtpcache-eagleshift`, booted once on port 8020 at stock power limits (600 W / 575 W). The first boot attempt hit the known warmup flake (`CUDA error: invalid argument`, R233) and the second came up with a 1,391,795-token pool and 3,351 MiB free per card. Engine errors 0, preemptions 0.

## Decode against concurrency

`probes/decode_ss.py`, 1,024 forced tokens per stream, three runs per shape, the rate taken over the samples where every stream was decoding; median of the runs.

| streams | code, all streams | code, per stream | prose, all streams | prose, per stream |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 212.5 | 212.5 | 169.1 | 169.1 |
| 2 | 397.7 | 198.8 | 337.1 | 168.6 |
| 4 | 851.8 | 212.9 | 718.2 | 179.6 |
| 6 | 1,121.1 | 186.8 | 950.2 | 158.4 |
| 8 | 1,496.1 | 187.0 | 1,229.9 | 153.7 |
| 12 | 2,097.6 | 174.8 | 1,729.5 | 144.1 |
| 16 | 2,442.5 | 152.7 | 2,085.6 | 130.3 |

MTP accepted 0.61 to 0.68 drafts per verify on code and 0.46 to 0.49 on prose at every concurrency.

R234 measured 1, 8 and 16 streams on the same configuration on 2026-09-09 with two runs per shape: code 216 / 1,476 / 2,596 and prose 164 / 1,267 / 2,183 tokens per second. The 16-stream code figure is 5.9 % lower here, and R675's spread at that shape is 2,403.5 to 2,501.2.

## Cold prefill against prompt length

`probes/kv_capacity_probe.py --conc 1 --tokens 1`: one request at a time, one output token, prompt tokens as counted by the server, three prompts per length, each with a fresh seed so that neither the GPU prefix cache nor the CPU and disk tiers hold it.

| prompt tokens (mean of 3) | time to first token | prefill rate |
| ---: | ---: | ---: |
| 6,692 | 0.80 s | 8,365 t/s |
| 25,035 | 3.03 s | 8,255 t/s |
| 50,051 | 6.83 s | 7,325 t/s |
| 100,067 | 17.50 s | 5,718 t/s |
| 166,768 | 37.83 s | 4,408 t/s |
| 200,117 | 50.57 s | 3,958 t/s |

The three prompts at each length agree to 0.1 s. The rate falls with length because every full-attention layer reads the whole prefix for each chunk.

## Decode at depth, one stream

`decode_ss.py --conc 1 --ctx N` puts N/1.3 filler words in front of the prompt. The probe does not record the prompt's token count, so the figure plots these points against N, not against prompt tokens. Two runs per point. The 0 point is the one-stream row above.

| `--ctx` | code | prose |
| ---: | ---: | ---: |
| 0 | 212.5 | 169.1 |
| 30,000 | 214.1 | 178.7 |
| 60,000 | 214.8 | 167.8 |
| 120,000 | 196.0 | 158.7 |
| 200,000 | 184.6 | 150.6 |

Code decodes at the same rate to 60,000 and 13 % slower at 200,000. R234's prose row at `--ctx 30000` read 157.6 with an acceptance of 0.41 drafts per verify, and R675's reads 178.7 at 0.55. The prompts differ between the two runs (per-run seed prefix), so the prose rate at depth moves with the drafts the text admits, not only with depth.
