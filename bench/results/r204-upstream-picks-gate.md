# R204: four upstream bug fixes gated on the served route, and the FlashInfer SM120 GDN prefill kernel measured and not adopted (2026-09-06 09:35 to 10:52 UTC, results `2026-09-06-r204-picks-gate`, [scripts/build-r204-picks.sh](../../scripts/build-r204-picks.sh), [scripts/r204-picks-gate.sh](../../scripts/r204-picks-gate.sh), layers `patches-v0290/Dockerfile.r202-picks` and `Dockerfile.gdn-fi-prefill`, credits in THIRD_PARTY.md)

[← all results](../RESULTS.md)

Two images were built on the served image. The first adds four upstream fixes that apply cleanly on the served tree and on the pristine v0.29.0rc2 wheel: vllm#55507 (0149, mamba align state index seeded with the mamba block size), vllm#54275 (0150, kv-cache-memory suggestion clamped by memory other processes took), vllm#54972 (0151, the UVA offloader releases freed accelerator blocks per module) and vllm#52807 (0152, a recurrent group's unhashed block must not truncate the offload load boundary). vllm#54165 was not ported: three of its four files are already the vllm#54163 port (0134) on this tree, and the fourth is a sliding-window fallback this model has no layer for. The second image carries vllm#50862 alone (0153), which opens the FlashInfer GDN prefill selector on SM12x; FlashInfer 0.6.16.post3 ships the SM120 CuTe-DSL kernel it selects, and vLLM casts the SSM state to float32 at the call site, so the bf16 SSM cache is unaffected.

The four-fix image on the served route (16 sequences, `pcie_ipc` all-reduce, batch-sharded sampling, DFlash draft length 7, evaluation tier):

| gate | result |
|---|---|
| boots at the 13.98 GB pin | 5 of 5 (free after boot 1,081 / 2,047 / 1,443 / 1,453 / 2,047 MiB); consistent with the 3 of 3 R193e measured on the served image at this pin; no boot-reliability change expected (vllm#54275 is inert under a pinned KV budget) or observed |
| pool, block, compile artifacts | 1,052,277 tokens, 1,552, the served image's three artifacts loaded on every boot |
| capacity | 100K context 15.1 % of the pool in 18.4 s; five concurrent 100K contexts at 76 %, five running, no preemption |
| warm revisit, 32K prompt | 7.49 s cold, 0.225 s second send, 52,768 of 53,453 tokens from the prefix cache |
| tier revisit, 32K prompt after a 12 × 90K flood | 0.59 s, 52,768 of 53,452 tokens served through the offloading connector (148 tier chunks, 1.26 GB read) |
| needle gate | 4 of 4 cold at 131K and 220K, 4 of 4 evicted re-asks served by the tiers |
| decode, steady state | code c8 1,680 tok/s (210 per stream), prose c1 158.3, code c16 2,489 (155.6 per stream) |
| dense ladder vs bf16 | +0.879 % PPL, top-1 92.805 %, KL 0.014073 |
| agentic ladder vs bf16 | +2.558 % PPL, top-1 95.555 % |
| per document vs the R203 spec-on dump (same artifact) | 0 of 693 dense documents beyond ±2 %, 70 of 724,781 positions moved by more than 1 nat; agentic 0 of 72 documents, 0 positions |
| engine error lines, preemptions | 0, 0 |

The four fixes are numerically inert on this route. Of the four, only 0152 has a live code path here (tier-served loads on a hybrid), and it shows no regression; its benefit was not isolated, because the served image already passed the same tier probes. 0150 never runs: with `--kv-cache-memory-bytes` given, vLLM skips memory profiling, so the 5-of-5 boot tally is the served image's own variance. 0149 is dormant until a block size is pinned explicitly. 0151 made no visible difference to free VRAM after boot.

The vllm#50862 image: the boot log reports the FlashInfer GDN prefill kernel, one of the three compile artifacts is rebuilt (the GDN prefill op changes the graph), and prefill throughput is unchanged at every depth measured, cold rows: 2K 4,604 and 4,850 tok/s (Triton/FLA 4,709 and 4,850), 4K 6,425 and 6,571 (6,425 and 6,366), 8K 7,354 and 7,347 (same), 32K 7,587 and 7,561 (7,569 and 7,543), 131K 5,206 and 5,193; the 100K capacity row takes 18.4 s on both. The GDN prefill kernel is not where prefill time goes on this machine. Fidelity: dense +0.879 % PPL vs bf16, top-1 92.773 %, agentic +2.620 %, top-1 95.610 %; per document against the served image's dump 105 of 693 dense documents beyond ±2 % and 1.51 % of positions moved, at the upper edge of the cross-artifact band established in R203 (92 to 104 documents, 1.42 to 1.44 % of positions). Decode is unchanged (code c8 1,681.5, prose c1 165.7). Not adopted: a speed-neutral change of numerics class. The same-artifact pair above also refines the R203 calibration: two boots on one artifact under the R193d protocol agree to 0.01 % of positions, so the ±2.4 % per-document band is the cross-artifact draw.
