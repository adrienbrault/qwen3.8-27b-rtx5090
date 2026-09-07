# R205b: the GDN speculative-decode metadata computed once per step instead of once per group (2026-09-06 19:36 to 20:05 UTC, results `2026-09-06-r205b-gdncm-gate`, [scripts/build-r205b-gdn-common-metadata.sh](../../scripts/build-r205b-gdn-common-metadata.sh), [scripts/r205b-gdncm-gate.sh](../../scripts/r205b-gdncm-gate.sh), layer `patches-v0290/Dockerfile.gdn-common-metadata`, patch 0157 = vllm#52297 ported)

[← all results](../RESULTS.md)

vllm#52297 (xyang16) hoists the batch-level speculative-decode metadata that `GDNAttentionMetadataBuilder.build()` recomputed for every GDN KV-cache group (the speculative sequence masks, the decode/prefill/spec split, the token index permutations, the query-start cumsums, the accepted-token gather) into one call per step from `MambaHybridModelState.prepare_attn()`; each group's build keeps only its block-table gathers. The PR's `build()` hunks do not apply on this tree (0108, 0111 and 0133 changed the region), so 0157 is a hand port: the helper is the PR's, the ReplaySSM branches are untouched, callers that pass nothing recompute locally. No knob; a refactor must be bitwise.

Gate: served image (OFF) against served image plus 0157 (ON) on the served route, `VLLM_TRITON_FORCE_FIRST_CONFIG=1` on both arms, one compile artifact (ON loaded the three AOT artifacts OFF had saved, nothing recompiled). Proof line once: "SM12X GDN common metadata hoisted: 10 GDN groups share one spec-metadata build per step" (10 = every attention group whose first metadata builder is a GDN builder, counted on the non-capture `prepare_attn` path; the cudagraph-capture path is excluded and computes its own tuple; why 10 rather than the 3 kv-cache groups was not chased).

| | OFF | ON |
|---|---|---|
| decode ruler ON vs OFF, ctx 0 and 30K, 20 chunks each | | 20/20 and 20/20 fully agreeing, median 0.0 |
| code, 1 stream (3 runs, range) | 273.0 (252 to 299) | 278.5 (258 to 297) |
| prose, 1 stream | 168.7 (159 to 178) | 175.7 (167 to 184) |
| code, 8 streams | 1,643 (1,503 to 1,759) | 1,720 (1,529 to 1,742) |
| code, 16 streams | 2,445 | 2,416 |

Every delta sits inside its own run-to-run range. The direction at 1 and 8 streams is consistent with removing launches from a launch-bound step (R194), but three runs cannot resolve a +2 to +5 % claim. Verdict: numerically inert and free; carried in the next image chain, not worth a served-engine restart on its own. Upstream reports build() 900 → 300 µs and +61 % single-stream throughput on an H200 with DFlash, which this box does not see.
