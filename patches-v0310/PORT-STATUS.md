# Phase A port status — 2026-10-12

Source-only completion; no image build, CUDA compilation, GPU test, SSH, network request or promotion. `patches-v0310/series` is authoritative. All paths below are relative to the supplied pristine `vllm-v0.31.0` tree unless marked daily. The port keeps the served route: ModelOpt mixed NVFP4 Qwen3.8-27B, TP2, NVFP4 KV, MTP `qwen3_5_mtp` normalized to `mtp` ns3, BF16 SSM, prefix caching, UVA embeddings, CPU/filesystem KV tier, full target/draft graphs, PCIe IPC and BSS.

## Ground truth and limits

The served package, rather than the old port notes, decides which knobs are on. `daily-src/vllm/config/cache.py:208-210` enables legacy ReplaySSM spec only from `VLLM_USE_REPLAYSSM_SPEC`; the launcher never sets it. The daily coordinator contains the non-Mamba fallback at :120 and the original 0158 retention log at :188. The daily offload scheduler contains the non-draft fallback at :244. The launcher forces MTP ns3, BF16 SSM, XQA off, PCIe IPC/BSS on, 14,860,000,000 KV bytes/rank, 16 GiB CPU tier, MNBT 8192. The served image does **not** include the later 0160–0163 picks.

A pristine v0.29.0rc2 tree and the original v0290 diffs are not supplied. Therefore a complete byte diff against that pristine release cannot be manufactured. Removed/context lines in the v0300 diffs document the original behavior, and each served implementation was checked directly against daily source. This reconciles the named chain, not an unsupported claim that every unrelated daily/upstream difference has been enumerated. The missing audit probe is: on flan, extract the untouched v0.29.0rc2 wheel/image package separately, then `diff -ruN` it against the supplied daily package; retain unaccounted differences for review. No input trees were modified.

## Every chain member

| ID | Decision | Source/intent and v0310 resolution |
|---|---|---|
| 0101 | regenerated | SM12x NVFP4 admission, HND layout, causal FA2 prefill/spec wrapper pool, linear writer hook in `v1/attention/backends/flashinfer.py`; GPU worker overlay preload. Upstream still does not provide this complete SM120 KV route. |
| 0101a | dropped | Conditional DFlash non-causal alternative, not applied in the served default (`WITH_DFLASH_NC=0` in the supplied prior plan). |
| 0102 | regenerated | `csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu`: SM12x linear V scale writer, `swizzle_v_sf`; overlay retained. Stable ABI target is still 2.11 (`CMakeLists.txt:1216`), matching the overlay. |
| 0103 | regenerated, off | XQA NVFP4 decode opt-in; preserve `VLLM_SM12X_NVFP4_XQA=0`. The dedicated/paged API split remains in the source. |
| 0104 | regenerated, MRV1 dormant | MRV1 proposer/runner full-graph escape. Daily BSS selects MRV2; MRV2 graph support is native and 0148 handles draft collectives. Do not cite 0104 as proof of MRV2 capture. |
| 0105 | regenerated, dormant | Non-causal NVFP4 FA2 DFlash route, FlashInfer backend. |
| 0106 | regenerated, dormant | Selector GPU RNG. v0310 moved the launch into `CandidateSampler`; remove the obsolete `SAMPLE_PROBABILISTIC` argument at that new call site. |
| 0107 | already upstream by replacement | `models/qwen3_dflash.py:_build_context_kv_buffers` detects unquantized projections; quantized projections use their quant method instead of `F.linear` (`_project_context_kv`). Upstream now handles packed weights directly. The old dense-eye materialization/AttributeError suppression is unnecessary and was dropped, preserving this newer behavior. |
| 0108 | regenerated | GDN runtime spec query width + accepted-index bounds in `gdn_attn.py`, `qwen_gdn_linear_attn.py`, FLA recurrent/sigmoid kernels. CUDA fused GDN MTP is additionally present upstream; these Triton edits remain relevant to B and the Triton control. |
| 0109 | regenerated, dormant/pooled | Per-width non-causal graph wrapper pool, used alongside causal target verification pools. |
| 0111 | dropped from A, replaced in B | Old environment-gated GDN spec experiment is off in daily, requires `none`, rejects KV connectors, forces FP32 checkpoints and uses the obsolete `layers/fla` path. B supplies explicit upstream CLI flags and a BF16-compatible align/offload integration. Dropping it changes no served flag-off path. |
| 0112 | regenerated, off | XQA verifier routing opt-in; `VLLM_SM12X_XQA_VERIFY=0`. |
| 0113 | regenerated, dormant | DFlash2 graph builder view. |
| 0116 | regenerated, dormant | DFlash NVFP4 non-causal revival and TP1 embedding guard. |
| 0117 | regenerated, dormant | Sentinel/codebook clamps and persistent draft embeddings. |
| 0118b | regenerated, dormant | Draft eager escape. |
| 0119 | regenerated, dormant | DFlash parallel-width exception. |
| 0129 | regenerated, dormant | DFlash graph opt-in; unchanged MTP behavior. |
| 0131 | regenerated | Pooled wrapper int workspace shrink, default 1 MiB, before plan/capture. |
| 0132 | regenerated, off | Masked XQA isolated stream; v0310 moved layout choice to local `kv_cache_layout`; retain that choice instead of reviving the old attribute. |
| 0133 | regenerated, off | Packed GDN BV opt-in, retains SM12x gate in envs/FLA. v0310 default fused CUDA decode may bypass this Triton kernel. |
| 0134 | regenerated, dormant exclusion | `use_eagle_block_drop` exists upstream; narrow it to target-KV-preserving methods. MTP remains enabled, DFlash excluded. |
| 0135 | regenerated | UVA embedding offload in loader/offloader/LLM API. |
| 0136 | regenerated, off; 2026-10-16 lambda fixed for the FI 0.7 two-arg guard (old port raised TypeError at first plan()) | Split-KV re-enable shim; keep `SPLIT_KV=0` except arm A_S. FI 0.7 source was supplied in r27c; this opt-in remains off and its deployed behavior still requires a separate GPU audit. |
| 0137 | regenerated | FS LRU, pinning and cap metrics/tests. Keep v0310's new `backpressure_detector` argument and superclass call; add eviction state around it. |
| 0138 | regenerated + vendor | PCIe IPC wrapper/capture/teardown, vendored `pcie_ipc_ar21`. Preserve new DFlash `precompute_context_kv` capture argument. v0310 also has native FI IPC; force `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=0` with the vendored path. |
| 0147 | regenerated | BSS removed from parallel compile-hash factors. |
| 0148 | regenerated | MRV2 autoregressive MTP PCIe IPC admission/capture, max rows 320. TP2 drafter remains TP2. |
| 0152 | already upstream | Offload scheduler `resolve_mamba_align_size` (:156), aligned chunk hit boundaries (:773+) and recurrent COW source handling (:289+) retain load boundaries consistent across hybrid groups. |
| 0154 | regenerated | Coordinator fallback marks only non-Mamba groups EAGLE; update warning text. Required for shifted replay retention while the offload scheduler independently treats unannotated shared MTP as non-draft. |
| 0155 | already upstream | Offloading scheduler :236+: unannotated groups treated as non-draft; :473: active decoding tail lift excludes finished requests; :807-837: `is_eagle_unverified` widens all group kinds, not only SWA. |
| 0156 | already upstream | Same scheduler :473 `is_decoding and not self.req.is_finished()` permits finished final chunks to be stored. |
| 0158 | already upstream by replacement | Coordinator `get_replay_boundaries` (:319-340) returns resend/extension minus one scheduler block; Mamba `reachable_block_mask` and `drop_eagle_checkpoint_block` retain the shifted checkpoint. With 0154's non-Mamba fallback, the hybrid uses this branch. No original log line is synthesized. |
| 0160 | regenerated, new pick | Avoid temporary draft vocabulary allocations at PP1; shared target embed/lm_head remains the runtime object. Validate lower load peak on flan. |
| 0161 | dropped/deferred | Unpromoted, default-off CPU seq-bound planning optimization; v0310 lacks its lower-bound plumbing, and FI 0.7 source is now supplied in r27c, but this optional private-wrapper adaptation remains deferred outside the fixing scope. Not required for daily parity. Probe the pin's private buffers/API before a separate opt-in port. |
| 0162 | regenerated, new pick | Post-save autotune barrier and `load_configs`, still absent after `tuner.save_configs` in pristine `warmup/kernel_warmup.py:482`. |
| 0163 | adapted external guard | `guard_flashinfer_workspace.py` audits FI 0.7's actual installed `execute_cudnn_gemm*` functions and replaces any shared `resize_` with zeroed call-local storage. Build report records functions, replacements, source hashes, remaining resizes. Original diff is against older FI, so it is not falsely labeled a clean 0.7 source port. |

## R679b, R685 and changed upstream behavior

v0310 pins **FlashInfer 0.7.0.post1** (`requirements/cuda.txt:17-18`), torch 2.13.0 and cuDNN frontend >=1.19.1. The Dockerfile retains this pin, instead of the prior fiswap to 0.6.16.post3. It introduces source-dependent behavior that needs renewed dense/needle/perf gates.

R679b's c16 loss was associated with unreachable tuned cuDNN FP8 configurations; R685 measured the improvement after 0162. Pristine v0310 still uses tuning-bucket overrides and only loads the disk config at warmup entry, then saves at exit. Its additional `sync_flashinfer_autotune_cache` helper broadcasts configs between ranks; that does not reload the leader's newly saved configs in the warmup function. Carry 0162; inspect FI DEBUG cache-hit/fallback messages on fresh-cache boots. Neither the newer FI pin nor a source port proves a c16 improvement.

`gdn/qwen_gdn_linear_attn.py:_resolve_gdn_prefill_backend` explicitly admits SM120, head K=128, CUDA >=13 to FlashInfer. `fi_chunk_gated_delta_rule` now calls `flashinfer.gdn_prefill.chunk_gated_delta_rule(..., backend="flashinfer")`, casts state/gates/beta to FP32, and converts cu_seqlens to int64. The in-tree CuteDSL prefill option remains **SM100 only**. Thus the R685 family of FlashInfer prefill remains reachable on flan. R27c now supplies FlashInfer 0.7 source: CP auto dispatch and the SM120 T/MN/fixup/prefill path remain reachable. The native `_gemm_workspace_at_least` helper prevents shared workspace relocation, uses call-local allocation outside capture and rejects insufficient workspace during capture. All eight cuDNN execution functions use it; existing 0163 makes zero changes to that snapshot. This mitigates the diagnosis's leading mechanism; it does not prove the historical illegal address fixed. See `../R685-r27c.md` and `../FI-r27c-audit.json`. Native CUDA fused GDN decode/MTP has also been added upstream; A may take it, B explicitly takes Triton.

No vLLM-side pre-grow of FI cuDNN workspace was found on this route. FI's GEMM internals are audited from the supplied tag and separately from the installed candidate pin during build. 0163's guard addresses shared-address relocation, not every possible prefill defect or semaphore fault. Run the trace/core/FlashInfer-vs-Triton-prefill probe in `HOW-TO-VERIFY.md`; retain a complete dense corpus, never score the partial dump after a crash.

R694 is **TabbyAPI/exllamav3** (`draft_gpu_split: [0,32]`), not this vLLM TP2 MTP route. No unsupported cuda:1 placement knob was transplanted. Measure TP2 draft placement/collectives separately if desired.

## Final static self-check

`../self-check.json` and `../self-check.log` record a fresh pristine reconstruction using every series entry at `patch -p1 --fuzz=0`, compilation of all 34 touched Python files and byte identity with `out/work-A`. 28 diffs, zero fuzz, PASS. Vendor Python also compiles. C++/CUDA extension builds, imports of the actual Linux dependencies, installed-image FI audit and all GPU gates remain operator checks. The native 0.7 source/helper and zero-change 0163 guard pass CPU tests on macOS and Linux with torch. The build also runs real CacheConfig/defaults plus full original target/MM/MTP constructor statements with GPU layers stubbed.
