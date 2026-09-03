# BRIEF 16 — rebase the sm120 NVFP4-KV / DFlash2 patch chain from vLLM v0.28.0 onto v0.29.0rc1

Scope: a rebased patch set that applies with `patch -p1 --fuzz=0` to the v0.29.0rc1 tree in the order below, with a per-hunk rationale and an explicit "absorbed upstream / no longer needed / semantics changed" verdict per patch. Verify ONLY with `patch --dry-run --fuzz=0` (in order, on a scratch copy) and `python3 -m py_compile` of every touched file. No GPU, no model execution, no tests, one pass. Neutral engineering vocabulary. Reasoning effort: medium.

## Why

The v0280 image chain (vLLM v0.28.0 + `patches-v0280/` 0101–0113 + `dflash-nvfp4-revival/` 0116–0119, 0129, 0131) is the daily and the nvfp4 candidate on the RTX 5090 box. v0.29.0rc1 (commit `33898f83`, 598 commits later) carries upstream work that overlaps ours (CUDA-graph memory reserve in the V2 runner, DFlash draft RoPE layout, FlashInfer XQA head_dim fallback, DFlash2 conv/selector architecture, Mamba prefill checkpoints). Applied for real the chain fails: 0101 loses 3/24 hunks in flashinfer.py, then 0103/0105/0109/0112 (same file), 0106 (dflash2/speculator.py), 0107 (qwen3_dflash.py), 0108 + 0111 (gdn_attn.py, mamba/abstract.py, config/cache.py), 0116 (vocab_parallel_embedding.py + flashinfer.py). The same sequence applies clean to v0.28.0 (control). Note that the rejects of a patch stacked on a partially-applied 0101 are partly consequences, not independent conflicts.

## Inputs (all local, read-only; paths under `$BUNDLE` = `/private/tmp/claude-501/-Volumes-Developer-smaft-kubernetes-home/23ee8117-a496-4853-be1a-d87476413a5c/scratchpad`)

- `$BUNDLE/v028src/` — pristine vLLM v0.28.0 source tarball. `$BUNDLE/v029src/` — pristine v0.29.0rc1 tarball. (Package root = `<tree>/vllm`.)
- `$BUNDLE/v028p/` — v0.28.0 with the WHOLE chain applied = the intended end state. `$BUNDLE/rebase-v0290/intended/*.diff` = `diff -u v028src v028p` per touched file (flashinfer.py 869 lines, gdn_attn.py 354, llm_base_proposer.py 192, qwen_gdn_linear_attn.py 170, dflash2/speculator.py 116, …). This is the ground truth of what the chain does.
- `$BUNDLE/rebase-v0290/patches/` — the 17 diffs in apply order (0101 0103 0104 0105 0106 0107 0108 0109 0111 0112 0113 are package-root relative, `-p1 -d <tree>/vllm`; 0116 0117 0118b 0119 0129 0131 are tree-root relative, `-p1 -d <tree>`). 0102 (csrc writer) and 0101a are NOT in scope: `csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu` is byte-identical between the two tags.
- `$BUNDLE/rebase-v0290/apply-log.txt` + `rejects/<patch>/<file>.rej` — what failed where on rc1. `$BUNDLE/v029q/` — rc1 after that partial apply (do not trust it as a base; start from `v029src`).
- `$BUNDLE/rebase-v0290/upstream-diffs/*.diff` — `diff -u v028src v029src` of every file the chain touches (what upstream changed underneath us).
- Per-hunk rationale of the original patches: `flan/patches-v0280/README-*.md` (`README-sm120-nvfp4.md` for 0101/0103, `README-r116-dflash.md`, `README-r114-gdn.md`, `README-r123-replayssm.md`, `README-r124-xqa-verify.md`, `README-r126-spec-graphs.md`), `flan/dflash-nvfp4-revival/NOTES-REVIVAL.md`, `NOTES14.md` (0129), `NOTES15.md` (0130/0131 ledger + mechanism).

## Upstream changes to reconcile explicitly (rc1 vs v0.28.0; PR numbers are vllm-project/vllm)

| Upstream | Ours | Question to answer |
|---|---|---|
| #53306 reserve CUDA-graph memory in the V2 runner's KV sizing; #53955 free profiling memory before KV allocation; #53682 profile in a throwaway graph pool | 0131 (pooled wrapper int workspace 8→1 MiB), 0129 (drafter FULL graphs), 0119 (uniform width) | Does the reserve now see our pooled wrappers' memory? Keep 0131 (it still shrinks real memory) but say whether the ledger's "sized before capture with zero reserve" premise is gone. |
| #50488 capture the widest uniform decode batch by default; #54418 keep default capture sizes memory-safe | 0101's graph-bound prefill wrappers pooled per batch size, 0119, `_prefill_cudagraph_max_bs` | Do the default capture sizes / the uniform-batch dispatch change which shapes we pool? |
| #53111 fall back to native FlashInfer decode when XQA cannot serve a KV group's head_dim | 0103 (XQA-NVFP4 decode routing, `VLLM_SM12X_NVFP4_XQA`), 0112 (xqa-verify) | Conflicts in the decode-wrapper selection; keep our knob semantics. |
| #54373 DFlash draft RoPE layout from its own config (a release-branch cherry-pick) | 0107 (quantized draft loader in qwen3_dflash.py) | 0107's failed hunk is in this loader; keep upstream's is_neox_style handling AND our ModelOpt/W4A16 loading. |
| #52816 `DFlash2DraftModel` local convolution + candidate selector; #53797 speculators-format loading; #53435 load fix | 0106 (selector sampling guard), 0113 (speculator graphs), 0117/0118b (speculator warmup/eager escape) | Our drafter is `dflash2-qwen38-syvai-w4a16` (DFlash2 class). Which of 0106's guards did upstream absorb? |
| #53183 V2 runner default for all models | 0104 (V1 runner drafter graphs, gpu_model_runner.py / llm_base_proposer.py) | 0104 applies clean; state whether it is dead code on the default runner now (we run V2). |
| #52789 internal prefill checkpoints for Mamba prefix caching; #53877 packed GDN decode beta in FP32; #52539 Qwen head ratios in fused GDN MTP; #53077 reset spec count on empty draft schedule | 0108 (GDN kernel hardening), 0111 (replayssm spec decode; touches config/cache.py, mamba/abstract.py, gdn_attn.py) | 0111 is the largest and most invasive; if upstream's checkpointing reshaped the metadata builder, say what survives. Model geometry: H=16 key heads, HV=48 value heads, K=V=128, conv 4. |
| #53002/#53336 group geometry for FlashAttention metadata; #52796 normalize FlashInfer prefill LSE before merging | 0101/0105/0109 non-causal FA2 route | LSE normalisation touches the merge path our non-causal route uses. |
| #53559 deprecated-parameter cleanup; #52557 dead `use_prefill_decode_attention` flag removed | 0116 (vocab_parallel_embedding tp1 mask), gate widening in 0101 | 0116's VPE hunk failed: see `upstream-diffs/model_executor_layers_vocab_parallel_embedding.py.diff` (59 lines). |

## Deliverables (write into `$BUNDLE/rebase-v0290/out/`)

1. `patches-v0290/` — the rebased diffs with the SAME file names as the inputs (a diff that needed no change is copied verbatim; say so in the notes). Keep each family's path convention (0101–0113 package-root, 0116+ tree-root). Keep every env knob and log line the launcher asserts on: `linear-V-scale store overlay ACTIVE`, `use_fa2_nvfp4_kv`, `VLLM_SM12X_NVFP4_XQA`, `VLLM_SM12X_DFLASH_GRAPHS`, `VLLM_SM12X_POOLED_INT_WS_MIB`, `_shrink_pooled_int_workspace`, `SM12x pooled FlashInfer wrappers`.
2. `NOTES16.md` — per patch: verdict (rebased / verbatim / dropped-absorbed-upstream / semantics-changed-needs-metal-check), per failed hunk what moved and what you did, and a short "risk on metal" list for the operator's audition (what to look at first in the boot log and which probe would catch a regression). Include the exact verification transcript (the dry-run + py_compile commands and their output).
3. `verify.sh` — the dry-run + py_compile sequence against a fresh copy of `v029src`, exit non-zero on any failure, so the operator can re-run it.

## Phase 2 (only after phase 1 verifies; separate diffs on top of the rebased chain, same verification)

- `0132-masked-nvfp4-xqa-sm120-v0290.diff` — port of vllm PR #53543 (`$BUNDLE/rebase-v0290/53543.diff`, base 234 commits behind rc1): masked NVFP4 XQA for speculative verification on SM120, model-dtype queries/outputs, packed K/V + scale caches passed to the masked call, opt-in isolated stream (`VLLM_FLASHINFER_XQA_USE_ISOLATED_STREAM`). It lands in the same functions 0103/0112 patch. Our runtime: XQA decode is ON for the fp8-KV daily but OFF (`VLLM_SM12X_NVFP4_XQA=0`) on the nvfp4 candidate because verification through XQA was wrong (R155, NOTES-REVIVAL) — state clearly whether #53543 addresses that failure mode and what the A/B cell is.
- `0133-gdn-packed-decode-bv16-v0290.diff` — port of PR #54181 (`54181.diff`): SM120 packed GDN decode launch `BV=16` for B≤24. Its gate is `H == HV == 16`; our model is H=16, HV=48, so the PR as written never fires here. Deliver the PR's kernel-side change plus an env knob `VLLM_SM12X_GDN_PACKED_BV` (unset = upstream selector; `16`/`32` = force for any H/HV with K=V=128 on capability 12.x) so the operator can A/B on this geometry. 0108 touches the same `fused_recurrent.py`; rebase on top of it. Skip the PR's benchmark/test files.

## Constraints

- Never touch the host `10.76.10.5` (no ssh, no docker). Everything is local; the operator builds and runs.
- Do not reformat or reflow untouched code; hunks minimal; `--fuzz=0` must hold.
- If a patch cannot be rebased faithfully without metal evidence, deliver the best mechanical rebase and list the open question in NOTES16 rather than inventing behaviour.
