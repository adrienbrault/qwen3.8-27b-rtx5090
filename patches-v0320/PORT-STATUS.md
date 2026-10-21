# Port status from the served 0.31 chain

Source: upstream vLLM v0.32.0 commit `5df63a7203cdb9328d824a27b682e77c7b9928be`. This is a native-XQA, MRV2, MTP-only candidate. DFlash and MRV1 parity are outside this candidate. Each retained change is in the ordered `series`; exact source and patch hashes are in `manifest.json`. CPU tests and image builds do not qualify GPU execution.

| Previous patch | Candidate disposition |
|---|---|
| 0101 | Native NVFP4 routing and XQA graph support replace the local FA2 wrapper pools. |
| 0102 | Native FlashInfer slot-mapping writer replaces the compiled writer overlay. |
| 0103 | Native SM120 XQA; old opt-in routing removed. |
| 0104 | MRV1-only draft graph edits omitted. |
| 0105 | DFlash-only noncausal FA2 path omitted. |
| 0106 | DFlash selector/RNG edits omitted. |
| 0108 | Accepted-index masks and reusable indices retained in 0001; native zero-draft routing and pure-spec slicing preserved. The old runtime convolution-width override is omitted: scheduled draft counts do not establish the packed query/state width on the new route. Native configured-width handling remains unchanged. |
| 0109 | DFlash wrapper pool omitted. |
| 0112 | Native masked XQA verification replaces local routing. |
| 0113 | DFlash graph setup omitted. |
| 0116 | DFlash/TP1 embedding edits omitted; candidate is TP2 MTP. |
| 0117 | DFlash warmup/persistent input edits omitted. |
| 0118b | DFlash eager escape omitted. |
| 0119 | Native XQA descriptors replace local FA2 verify-width pools. |
| 0129 | DFlash NVFP4 graph opt-in omitted. |
| 0131 | Pooled FA2 integer workspaces omitted with their wrapper pools. |
| 0132 | Native masked verification; old isolated-stream workaround omitted and remains a GPU-validation risk to check. |
| 0133 | Packed GDN BV selector retained in 0001/0004. |
| 0134 | DFlash method changes omitted; MTP explicitly keeps `disable_eagle_block_drop:true`. |
| 0135 | Embedding UVA and loader integration retained in 0002. |
| 0136 | Opt-in split-KV shim adapted to native FA2 prefill in 0003; variadic private-guard signature retained. |
| 0137 | Filesystem cap, LRU and request/job pins retained in 0002. The test fixture follows the new transfer-job API. |
| 0138 | Fixed-tactic TP2 PCIe lifecycle, dispatch and target capture retained in 0004; competing optional backends are refused. DFlash hooks omitted. |
| 0147 | BSS excluded from the parallel compile hash in 0003. |
| 0148 | MTP admission/capture moved to `target_dependent_ar` in 0004; removed pass fields no longer accessed, new specialization arguments preserved. |
| 0154 | Mamba groups exempted from fallback EAGLE tagging in 0002; new native boundary logic preserved. |
| 0160 | PP1 target vocabulary sharing retained in 0003 and combined with the native early checkpoint filter. PP>1 allocation path retained. |
| 0162 | Reload retained in 0003 using each rank's new cache filename and dirty/save policy. |
| L1a | Draft FA2 host planning/events omitted for native XQA. This candidate does not provide the old FA2 fallback optimization. |
| BA4 | Integration retained in 0005, off by default; fresh control-image binding and `V032_BA4-serving-profile-v1` reject old serving evidence. |

The fixed PCIe implementation remains restricted to RTX 5090, SM120, CUDA 13.0, bf16 hidden size 5120, same-node TP2 and sequential MTP. The capture row cap is 320; the proposed configuration sets that maximum explicitly. Other shapes or parallel layouts must use another configuration rather than bypass these guards.
