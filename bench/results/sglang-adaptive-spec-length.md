# SGLang on the same card, adaptive speculative length: it matches the best static setting (2026-08-27, `results/2026-08-27-sglang-adaptive`)

[← all results](../RESULTS.md)

SGLang ships the acceptance-adaptive draft length that vLLM lacks (`--speculative-adaptive`). vLLM's dynamic SD is a static batch-size table, and its adaptive-verification track is unmerged and blocked on GDN ragged-K kernels. Measured on the same 5090 with SGLang's own RadixArk NVFP4 checkpoint and the official cookbook recipe (fp8 KV; note that `mem-fraction-static` contains the hybrid GDN state cache, inverted semantics against vLLM):

| SGLang arm | prose c1 | code c1 |
|---|---|---|
| MTP ns3/draft4 fixed | 124.1 | 138.1 |
| MTP ns7/draft8 fixed | 112.5 | 139.6 |
| MTP + adaptive (ladder [1,3,7], oscillation verified live) | 122.8 | 136.1 |
| DFlash2 draft8 fixed (nightly image) | 139.1 | 174.7 |

The adaptive controller **equals the best static point** and recovers the mistuned one (+9% over static ns7 on prose). It picks the right spot on the depth curve rather than exceeding it, consistent with the depth-sweep frontier above. DFlash2+adaptive is refused ("only EAGLE/EAGLE3"). The cross-engine comparison is confounded by checkpoint recipe (about 8%) and cache systems: SGLang matches vLLM on prose, trails about 20% on code, and its capacity on this recipe (about 12K-token KV pool, hard 2-concurrent GDN-state cap) is not in the same class as the vLLM daily's 388K plus tiered cache. Two upstream sharp edges: `--speculative-adaptive` crashes at boot unless `speculative-num-draft-tokens` covers the candidate ladder's maximum ("shared logits buffer holds N rows but caller needs 2N"), and the flag silently no-ops for non-EAGLE algorithms.
