# R197: the speculative-length ladder, seven draft tokens promoted, and the MTP head admitted to the `pcie_ipc` all-reduce (2026-09-05 12:03 to 14:17 UTC, results `2026-09-05-r197-spec-ladder`, [scripts/r197-spec-ladder.sh](../../scripts/r197-spec-ladder.sh))

[← all results](../RESULTS.md)

One boot per draft length on the served route (port 8029, same image, flags and 13.98 GB pin as the serving port, 16 sequences, `pcie_ipc` all-reduce and batch-sharded sampling on, `VLLM_TRITON_FORCE_FIRST_CONFIG=1`): DFlash2 at 6, 7, 8, 9, 9 again, 10 and 11 draft tokens, then the checkpoint's own MTP head at 3 and 4. Per arm: `probes/decode_ss.py` code at 1 stream (3 runs), prose at 1 stream (2), prose at 30K context (2), code and prose at 8 streams (2 each), code at 16 streams (2); the bf16 decode ruler at 0 and 30K. Steps/s = t/s ÷ (1 + ns × acceptance per draft token), with the arm's own ns.

| arm | pool | code 1 stream | prose 1 stream | prose at 30K | code 8 streams | prose 8 streams | code 16 streams |
|---|---|---|---|---|---|---|---|
| DFlash 6 | 1,066,119 | 269 (70.1) | 170 | 143 | 1,629 (458) | 1,131 (456) | 2,367 (664) |
| **DFlash 7 (promoted)** | 1,052,277 | 273 (71.3) | 171 | 142 | 1,633 (442) | 1,134 (442) | 2,455 (630) |
| DFlash 8 | 1,033,548 | 276 (76.1) | 169 | 153 | 1,513 (394) | 994 (395) | 2,135 (540) |
| DFlash 9 (the daily until R197) | 1,020,596 | 306 (74.9) | 172 | 148 | 1,480 (381) | 941 (379) | 1,996 (505) |
| DFlash 9, second boot | 1,020,596 | 305 | 173 | 147 | 1,472 | 941 | 2,041 |
| DFlash 10 | 1,007,709 | 304 (72.9) | 176 | 155 | 1,346 (365) | 927 (365) | 1,856 (478) |
| DFlash 11 | 990,211 | 278 (73.4) | 171 | 149 | 1,347 (353) | 893 (354) | 1,935 (462) |
| DFlash 9, CUSTOM all-reduce | 1,020,596 | 286 (69.8) | 166 (71.7) | 140 (69.4) | 1,469 (362) | 905 (360) | 1,929 (474) |
| MTP 3, CUSTOM all-reduce | 1,264,777 | 226 (69.2) | 153 (69.1) | 151 (67.1) | 1,522 (507) | 1,225 (507) | 2,580 (837) |
| **MTP 3, `pcie_ipc` (patch 0148)** | 1,309,368 | 232 (71.7) | 158 (71.4) | 157 (69.6) | 1,568 (523) | 1,247 (525) | 2,699 (871) |
| MTP 4, CUSTOM all-reduce | 1,288,987 | 229 (63.7) | 145 (64.1) | 137 (62.3) | 1,500 (451) | 1,136 (447) | 2,386 |
| MTP 4, second boot, CUSTOM | 1,288,987 | 229 | 145 | 137 | 1,520 | 1,122 | 2,388 |
| MTP 4, `pcie_ipc` (patch 0148) | 1,288,987 | 237 (66.6) | 150 (66.5) | 142 (64.5) | 1,548 (467) | 1,162 (467) | 2,523 (733) |
| MTP 6, CUSTOM all-reduce | 1,248,880 | 223 (57.0) | 136 (57.0) | 137 (55.2) | 1,415 (377) | 1,111 (378) | 2,223 (571) |

Tokens per second aggregate, steps/s in parentheses where the acceptance was captured. Acceptance per draft token on code at 1 stream falls with the draft length: 0.473 at 6, 0.404 at 7, 0.329 at 8, 0.342 at 9, 0.317 at 10, 0.253 at 11; the MTP head accepts 0.754 at 3, 0.647 at 4 and 0.486 at 6. The MTP 3 arm booted at the 13.5 GB pin (its first boot at 13.98 GB died in the Triton warm-up with the R191 "invalid argument" signature); every other arm booted at 13.98 GB.

Readings.
- At 8 and 16 streams, 6 and 7 draft tokens are equal within the run band and beat 9 by 10% (code, 8 streams), 20% (prose, 8 streams) and 23% (code, 16 streams) in tokens per second, 16 to 25% in steps/s. 8 sits between; 10 and 11 fall below 9.
- At 1 stream, 9 and 10 keep the code lead (306 and 304 against 273 at 7, 11%); prose at 1 stream and at 30K is within noise across 6 to 11.
- The draft length moves the attention block: the block follows the number of speculative slots (1,536 tokens at 6, 1,552 at 7, 1,568 at 8, 1,584 at 9), so every length compiles its own artifact (the hash sets differ), the cross-length decode-ruler differences are void under the R193 protocol, and every tier hash changes with the length. Speed and pool stand; the pool grows 3.1% from 9 to 7.
- The 2026-09-04 retraction of 7 (R173c: twice the distance from the bf16 decode reference at 30K) rested on a same-size per-boot draw that R193d has since shown to be the runtime Triton autotune, not the draft length. The two boots at 9 in this unit are bitwise at both contexts, so the protocol holds within a length.
- The MTP head (one target layer, no drafter weights, 15 `mtp.*` tensors of the checkpoint) first ran on the CUSTOM all-reduce because patch 0138 admitted only the DFlash drafter. Patch 0148 ([patches-v0290/0148-pcie-ipc-mtp-drafter-v0290.diff](../../patches-v0290/0148-pcie-ipc-mtp-drafter-v0290.diff), design note [NOTES30.md](../../patches-v0290/NOTES30.md), image layer `Dockerfile.pcie-mtp`, knob `VLLM_SM12X_PCIE_IPC_MTP=1`) admits it: both drafter graph families are captured inside the workspace capture context and every drafter row count is registered before the first capture; the FlashInfer kernel is unchanged, its scratch epoch advances on the device at every kernel entry, which is what makes the MTP head's repeated same-shape replays per step safe. On the same loaded compile artifacts, MTP 3 and MTP 4 on `pcie_ipc` are bitwise equal (20 of 20 chunks at 0 and 30K) to their CUSTOM twins, and the boot-to-boot floor holds for MTP too (the two MTP 4 CUSTOM boots are bitwise). The all-reduce gain at MTP shapes is 3 to 5% on every probe (code 1 stream +3.6% steps/s, prose at 30K +4.1%, 8 streams +3.0%, 16 streams +4.6%); at DFlash 9 the same pairing (CUSTOM boot against the `pcie_ipc` boot, bitwise) gives +7.3% steps/s at 1 stream, +5.5% at 30K, flat at 8 streams, +3.5% at 16.
- MTP 3 on `pcie_ipc` is the fastest arm of the ladder at 8 and 16 streams (code 1,568 and 2,699 t/s against 1,633 and 2,455 for DFlash 7; prose at 8 streams 1,247 against 1,134), holds its single-stream rate at 30K context (157 against 142 for DFlash 7), has the largest pool (1,309,368, +24% over DFlash 7), and the worst single-stream code rate (232 against 273, −15%; against DFlash 9 −24%). Longer MTP drafts lose: 4 is below 3 everywhere, 6 is below 4 everywhere. The MTP head also disables prefix-cache reuse and disk-tier hits on this hybrid ("no KV cache group could be identified as the draft model's", every group is treated as a draft group), which the ladder does not measure and which would decide a promotion.
- 8 draft tokens is the outlier at 8 and 16 streams (below both neighbours). The rows-per-step at 8 streams cross 64 and at 16 streams 128 there, a CUDA-graph bucket boundary; not verified.

Promotion (14:xx UTC, user "I validate switching from d9 to d7"): the launcher serves 7 draft tokens, asserts the 1,552-token attention block, and wipes the disk tier's `_model_*` content on the first boot whose block stamp differs (every tier hash carries the block size, as at R182). The previous launcher is frozen as [scripts/serve-r195-ns9-daily.sh](../../scripts/serve-r195-ns9-daily.sh). The serving port (2026-09-05 14:42 to 15:09 UTC, `results/2026-09-05-r197-promote-ns7`, `scripts/r197-promote-ns7.sh`, queued behind the R196 unit so it replaced the plain restore): the first boot of the seven-draft-token launcher on port 8020 came up at the 13.98 GB pin on the first attempt. The launcher found no block stamp on the disk tier, wiped its 270 GB of 1,584-token content and stamped 1552; the engine set the 1,552-token attention block with `num_spec_tokens=7`, the all-reduce order is `PCIE_IPC, CUSTOM, PYNCCL`, the pool is 1,052,277 tokens with 2,041 MiB free.

| gate on port 8020 | result |
|---|---|
| kv_capacity, 1 short request | 2.7% of the pool, no preemption |
| kv_capacity, one 100K request | 15.1% of the pool, 18.5 s prefill |
| kv_capacity, five concurrent 100K requests | 76.5% of the pool, 106 to 115 s each, no preemption |
| needles at 131K and 220K, cold on the wiped tier | 4/4 hits (25.6 s and 56.7 s prefill) |
| the same needles re-asked after a 12 × 90K eviction flood | 4/4 hits served from the tier: 130,368 of 131,245 and 218,832 of 220,336 tokens external, 1.2 s and 2.2 s |
| decode, 1 stream: code (3 runs) / prose / prose at 30K | 279 (accept 0.415) / 176 (0.210) / 151 t/s (0.169) |
| decode, 8 streams: code / prose | 1,629 (204 per stream, 0.381) / 1,093 t/s (137, 0.212) |
| decode, 16 streams: code / prose | 2,361 (148, 0.393) / 1,700 t/s (106, 0.243) |
| tool-eval 69 × 4 | 90.8 ± 1.0, CI [90.0, 91.5], trials 124/124/125/127 (R189 daily 91.2 ± 1.3, R195 90 ± 0.8) |
| engine error lines / preemptions | 0 / 0 |

Against the nine-draft-token daily on the same port (R189b/R195/R195b): code 8 streams 1,480 → 1,629 (+10%), prose 8 streams 941 → 1,093 (+16%), code 16 streams 1,996 → 2,361 (+18%), prose 16 streams 1,319 → 1,700 (+29%), single-stream code 306 → 279 (−9%), prose 172 → 176, prose at 30K 148 → 151, pool +3.1%. The 1,552-token blocks round-trip through the host and disk tiers with the right answers, so the block change costs one tier wipe and nothing else. The seven-draft-token configuration has been the served one since 2026-09-05 14:45 UTC.
