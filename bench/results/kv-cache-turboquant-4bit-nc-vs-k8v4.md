# KV cache (prior daily): turboquant_4bit_nc vs turboquant_k8v4

[← all results](../RESULTS.md)

Both configs, same box, same session (2026-07-15), identical invocation: [llama-benchy](https://github.com/eugr/llama-benchy) 0.3.8, `--pp 512 4096 --tg 128 --concurrency 1 2 4 8 --runs 3`, util 0.94, both with `--no-async-scheduling`.

| | turboquant_k8v4 | **turboquant_4bit_nc** (prior daily) |
|---|---|---|
| KV pool | 165,274 tok | **~235,000 tok** (+42%) |
| KV memory | 4.89 GiB | ~4.89 GiB |
| KV density | 33.8K tok/GiB | **~48K tok/GiB** |
| max-model-len | 160K | **200K** (+25%) |
| decode c1 @512 (tg mean) | **137** | 133 |
| decode c2 @512 | **250** | 211 |
| decode c4 @512 | 426 | **432** |
| decode c8 @512 | **467** | 435 |
| decode c1 @4096 | **145** | 126 |
| decode c2 @4096 | 179 | 179 |
| decode c4 @4096 | 230 | 230 |
| decode c8 @4096 | **216** | 214 |
| MTP acceptance length (ns=3) | ~3.2 | ~3.2 |
| tool-eval-bench v2.1.0 | 89 | 89 |

The split: `4bit_nc` costs a small decode tax of −3% c1 / −7% c8 at short context (c2 is the noisiest at −16%; c4 is at parity), and its worst case is deep single-stream, −13% (c1@4096: 126 vs 145). The cause is the 4-bit-key dequant, a Lloyd-Max codebook plus per-GQA-head norm-correction, with the inverse Hadamard hoisted to one per-query GEMM rather than per key: that is more ALU work than k8v4's cheap FP8-cast keys. From c2 up at deep context the two are within noise. In exchange `4bit_nc` carries **+42% pool / +25% usable context**, with equal retrieval and equal MTP acceptance, a pool-for-decode trade that fits interactive coding (low concurrency, deep context).

> Older k8v4 numbers are retired. Earlier revisions quoted `turboquant_k8v4` at decode c1 164 @512 (from a standalone `k8v4-bench.json`) and a k8v4-vs-fp8 table built on it. A fresh same-session re-measurement did not reproduce the 164: fresh k8v4 measured ~137 c1 @512. The table above uses the reproduced same-session figures, and the 164 outlier and the derived fp8 head-to-head are dropped. fp8's own earlier same-session decode, for reference, was not re-run under `--no-async-scheduling`: @512 c1 130 / c2 251 / c4 482 / c8 478 (peak c8 832); @4096 it leads from c2 up (c4@4096 461). The one regime fp8 still wins is deep context at high concurrency.

## Retrieval quality (needle-in-haystack)

Plant 5-digit codes in coherent filler and check for exact matches. This test appeared to expose `turboquant_4bit_nc`, until the 0/8 turned out to be async×spec KV corruption rather than the 4-bit keys.

| KV cache | 9K | 20K | 40K |
|---|---|---|---|
| `turboquant_4bit_nc` — *async scheduling ON* | **0/8 across depths (async×spec corruption, not the keys)** | | |
| **turboquant_4bit_nc** — *`--no-async-scheduling`* (prior daily) | **8/8** | **8/8** | **8/8** |
| fp8_e4m3 | 6/6 | 8/8 | — |
| turboquant_k8v4 | 8/8 | 8/8 | 6/6 |

`turboquant_4bit_nc` with `--no-async-scheduling` also passes **high-pressure concurrency: 90/90** (3 rounds × 30 needles, 6 background loaders), the exact test that the "all 4-bit-KV corrupts under concurrency" expectation predicted it would fail.

Pool ≠ usable context. TurboQuant's continuation-prefill materializes the whole cached prefix in bf16 (~4 KB/token transient), which OOM-kills the engine on a single prompt far past the cap. The shipped config caps max-len at 200K against the ~235K-token pool, and the pool beyond the cap buys concurrent-sequence headroom only.
