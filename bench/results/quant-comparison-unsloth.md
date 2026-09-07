# Quant comparison, prior daily: Unsloth selected on quality (NVFP4 model selection, same flags)

[← all results](../RESULTS.md)

| | Unsloth | natfii (modelopt) | NVIDIA official |
|---|---|---|---|
| decode c1 / c8 | 131 / 894 | 126 / 881 | 93 / 757 |
| prefill @4K | 9,592 | **13,348** | 4,921 |
| max ctx @ util 0.95 | 144K | **200K** | OOM (→150K @0.92) |
| **Terminal-Bench 2.1** (8 tasks ×2) | **15/16, 8/8 pass@2** | 12/16, 7/8 | — |

natfii is faster, but **Unsloth scores higher on quality**, and quality decided the selection. NVIDIA's official quant loses on every axis, and its slow prefill path is decisive.
