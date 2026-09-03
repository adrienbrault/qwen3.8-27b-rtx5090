# BRIEF 18 review — 0135 embed UVA offload

## Result

The upstream patch is carried verbatim except for the complete `tests/basic_correctness/test_cpu_offload.py` diff. No 0136 follow-up is needed from source review. The main metal risk is not a whole-table copy in Python; it is whether the existing fused TP embedding kernel and captured graph deliver acceptable/correct zero-copy PCIe reads on the two RTX 5090s.

## 1. TP=2 embedding and loading

`VocabParallelEmbedding` creates one vocab shard per TP rank (`vocab_parallel_embedding.py:257-345`). The loader narrows the checkpoint tensor to that rank's vocabulary interval and writes it in place with `copy_`, then zero-fills padding (`vocab_parallel_embedding.py:457-497`). Because 0135 replaces `p.data` with the accelerator view before checkpoint loading (`offloader/uva.py:111-121`, and the new hook is immediately after construction), those writes target the UVA view's pinned host backing; no full device table is recreated.

The ordinary TP path compiles only mask/index arithmetic (`get_masked_input_and_mask`, `vocab_parallel_embedding.py:169-194`), gathers rows with plain `F.embedding(input_, layer.weight)` (`vocab_parallel_embedding.py:78-79`), masks the small output, and all-reduces it (`vocab_parallel_embedding.py:525-545`). There is no weight `.to()`, `.contiguous()`, `.clone()`, or dtype cast in that path. The 0116 TP=1 safety hunk masks invalid input IDs and masks the gathered output only (`vocab_parallel_embedding.py:499-508`); it also does not transform the weight.

On this CUDA, TP=2, unquantized embedding shape, `use_fused_embedding` is selected (`vocab_parallel_embedding.py:329-345`) and the alternate branch passes `self.weight` directly to `ops.vocab_parallel_embedding` (`vocab_parallel_embedding.py:510-523`). It likewise has no Python-side whole-table copy. A CUDA kernel may legally dereference the stable device alias of pinned host memory, but the custom kernel's UVA correctness/performance is a metal-only item; the fidelity ruler and decode probes below settle it.

Post-load processing calls `UnquantizedEmbeddingMethod.process_weights_after_loading`, which is a no-op unless the platform itself is CPU (`vocab_parallel_embedding.py:62-67`). The generic device-loading context recognizes `_vllm_is_uva_offloaded`; if a quantizer replaces a parameter it re-offloads the replacement and restores the marker (`model_loader/utils.py:154-197`). This embedding is unquantized, so it is not recreated. Conclusion: the rank-local BF16 shard remains backed by pinned host RAM after loading.

## 2. Hook placement and multimodal names

Both new-style and legacy constructors are hooked immediately after `model_class(...)` and before metadata recording (pre-patch `model_loader/utils.py:54-61` and `:90-94`). `BaseModelLoader` then loads checkpoint weights and only afterward performs post-load processing (`model_loader/base_loader.py:53-82`). Thus the target wrapper, its `Qwen3_5ForCausalLM`, and its inherited `Qwen3NextModel.embed_tokens` have all been constructed before `maybe_offload_embeddings` walks the completed top-level model. The embedding itself is directly constructed at `qwen3_next.py:635-651`; decoder layers alone go through `make_layers` at `:653-662`.

The multimodal wrapper constructs `language_model` inside `_mark_language_model` (`qwen3_5.py:538-540`). Vision is independently constructed under `_mark_tower_model`; that context routes the tower through the existing tower-offload hook (`models/interfaces.py:331-388`). The 0135 top-level walk can see all `VocabParallelEmbedding` descendants, but the exact-segment test in `UVAOffloader` is `".{param}." in ".{prefix}{name}."` (`offloader/uva.py:99-107`, modified but semantically retained). Therefore selector `embed_tokens` matches `...language_model.model.embed_tokens.weight`; it does not match `visual.pos_embed.weight` or `visual.patch_embed...`. Those vision members are named at `qwen3_vl.py:600-607`, so neither contains an exact `embed_tokens` segment.

## 3. DFlash2 drafter

The registered DFlash2 class inherits `DFlashQwen3Model` (`qwen3_dflash2.py:238-273`). Its base constructor allocates its own `VocabParallelEmbedding(config.vocab_size, config.hidden_size)` (`qwen3_dflash.py:380-406`), and its loader explicitly accepts/loads `embed_tokens` from the draft checkpoint (`qwen3_dflash.py:811-851`). The DFlash proposer uses the normal `get_model` path (`llm_base_proposer.py:1361-1375`), and `BaseModelLoader` calls `initialize_model` (`model_loader/base_loader.py:53-59`), so 0135 sees it under prefix `draft_model` before draft weight loading. The selector therefore matches this temporary table too.

After loading, DFlash uses the base sharing method (`llm_base_proposer.py:1455-1456`). Since the DFlash class has no `has_own_embed_tokens` attribute, that method takes its MTP/default branch and unconditionally replaces `self.model.model.embed_tokens` with the target table at PP=1 (`llm_base_proposer.py:1493-1530,1556-1559`). The launch is TP=2 but PP defaults to 1, so steady state owns only the target table. The draft allocation can nevertheless consume offload budget and pinned RAM transiently before replacement.

The local bundle does not include `/draft/config.json`, so the specific checkpoint's `vocab_size`, `hidden_size`, and hence exact temporary table bytes cannot be source-proven here. The settling operator experiment is read-only: `python3 -c 'import json; c=json.load(open("/draft/config.json")); print(c["vocab_size"], c["hidden_size"])'` inside the launched image (or against the mounted file on the host). Per rank, its BF16 bytes are `pad_vocab_size(vocab_size, 64) / 2 * hidden_size * 2`.

If the checkpoint is the expected same-width Qwen3.8 companion (248,320 × 5,120 BF16), each draft shard is 1,271,398,400 bytes = 1.18408203125 GiB. Target plus temporary draft is 2,542,796,800 bytes = 2.3681640625 GiB per rank, so `--cpu-offload-gb 2.37` admits both. Because the offloader checks its limit before each whole parameter, `1` GiB still admits the first 1.184-GiB target shard and then skips the draft shard; this is the recommended launch setting because the draft table is subsequently discarded. Expected steady/offloader count is `1.18` per rank. Treat the 2.37 figure as conditional until `/draft/config.json` confirms the dimensions.

## 4. CUDA graphs and compile

The V2 graph capture invokes the model directly inside `torch.cuda.graph` (`v1/worker/gpu/cudagraph_utils.py:375-399`), so decode capture includes the input lookup on the first PP rank. Breakable graphs likewise execute the runnable inside capture (`compilation/breakable_cudagraph.py:353-394`). A UVA alias is exposed as an accelerator tensor by `get_cuda_view_from_cpu_tensor` (`utils/torch_utils.py:898-920`), has stable storage, and is therefore presented to Dynamo/Inductor and graph capture like an ordinary CUDA tensor. The reviewed embedding branches directly consume that tensor; no per-forward `copy_` to a device table exists.

`sync_prev_onload` and `join_after_forward` are virtual no-ops in `BaseOffloader` (`offloader/base.py:93-99`). Only `PrefetchOffloader` overrides them. Calls around full and breakable capture/replay (`cudagraph_utils.py:375-394,439-452`; `breakable_cudagraph.py:377-387,419-423`) consequently do nothing for `UVAOffloader`; they do not add transfers or alter the UVA view. Source review finds no graph-specific incompatibility. Metal still must prove the custom fused gather under the 0129 drafter graphs and V2 target graphs.

## 5. Fallback trap and boot assertions

Before 0135, when UVA is unavailable or `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1`, `UVAOffloader` stores CPU parameters and wraps `forward`; each call builds `device_state` using `v.to(device)` and invokes `functional_call` (`offloader/uva.py:115-149`). That is catastrophic for a 1.184-GiB shard per step. 0135 deliberately sets `supports_direct_module_offload` only when `uva_offloading` is true, so direct embeddings are not wrapped in this fallback. UVA availability itself is pinned-memory availability (or CPU) (`utils/platform_utils.py:50-57`), and CUDA reports pinning available except restricted/disabled WSL cases (`platforms/cuda.py:290-309`).

There is no dedicated upstream `uva_offloading=True` log line. Assert the branch indirectly and fail closed:

```bash
[ "$(echo "$BOOTLOG" | grep -ac "Offloader set to UVAOffloader")" -ge 1 ] || fail "UVAOffloader not selected"
[ "$(echo "$BOOTLOG" | grep -ac "Total CPU offloaded parameters: 1.18")" -ge 1 ] || fail "target embed shard not UVA-offloaded"
[ "$(echo "$BOOTLOG" | grep -ac "CPU offload parameter selector(s) matched no parameters:.*embed_tokens")" -eq 0 ] || fail "embed_tokens selector unmatched (UVA unavailable/disabled or hook inactive)"
[ "$(echo "$ARGS" | grep -ac -- "--offload-backend uva")" -ge 1 ] || fail "UVA backend arg missing"
[ "$(echo "$ARGS" | grep -ac -- "--cpu-offload-params embed_tokens")" -ge 1 ] || fail "embed selector arg missing"
[ "$(echo "$ARGS" | grep -ac "VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1")" -eq 0 ] || fail "UVA explicitly disabled"
[ "$(echo "$ARGS" | grep -ac "VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1")" -eq 0 ] || fail "pinning explicitly disabled"
```

The combination matters: `Offloader set to UVAOffloader` identifies the configured class only, while the `1.18` line plus no unmatched warning proves that direct embedding offload actually happened.

## 6. Pinned host RAM and container limits

Weight pinning requires `is_pin_memory_available()` and absence of `VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY` (`offloader/base.py:23-32`). UVA additionally requires that capability and absence of `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA` (`offloader/uva.py:48-51`). The host-weight path first copies to CPU, calls `pin_memory()`, and then creates the CUDA alias (`offloader/uva.py:111-119`).

`--ipc=host` concerns shared-memory IPC and does not itself raise the process's locked-memory limit. CUDA pinned allocations normally use CUDA host registration/allocation; the local code does not request a Docker capability. Still, add `--ulimit memlock=-1:-1` to the operator's `docker run` as a conservative requirement for multi-GiB registration, and assert no `pin_memory`, `cudaHostAlloc`, or `cudaHostRegister` error at boot. This cannot be conclusively settled without the actual Docker daemon/driver limits.

The 4-GiB `OffloadingConnector` tier is a separate pinned allocation: it registers the shared mmap with `cudaHostRegister` (`v1/kv_offload/cpu/gpu_worker.py:193-215,763-766`). With only the steady target embeddings, host pinning is about 2.368 GiB across two ranks plus 4 GiB for KV, roughly 6.37 GiB before smaller pinned buffers. If both draft shards are transiently offloaded, peak is roughly 8.74 GiB. Both fit comfortably in 64 GB physically, but the existing container `--memory 52g` and memory gate still need to cover model/checkpoint loading and the rest of the engine.

## Operator launch block

Recommended additions (offload only the target table; the temporary draft table is matched but skipped once the one-GiB threshold has been crossed, then replaced by the target table):

```bash
IMG=vllm-qwen38:v0290rc1-nvfp4kv-revival-prs-embed
EXTRA_ARGS="--kv-cache-memory-bytes $KV_BYTES --offload-backend uva --cpu-offload-gb 1 --cpu-offload-params embed_tokens"
# Add to docker run:
--ulimit memlock=-1:-1
```

Expected `Total CPU offloaded parameters` after the target hook: `1.18` GiB per rank (1,271,398,400 bytes). The line may repeat because `wrap_modules` logs the cumulative total on later calls. If `/draft/config.json` confirms 248,320 × 5,120 and the operator intentionally wants the temporary draft table pinned too, use `--cpu-offload-gb 2.37` and expect the cumulative line to reach `2.37` per rank before sharing deletes the draft table.

The target frees exactly 1,271,398,400 device bytes per GPU. Adding that to each existing fixed pool keeps the same measured free-after-pre-warm margin:

| SEQS | old pin (bytes/GPU) | new pin (bytes/GPU) |
|---:|---:|---:|
| 8 | 13,800,000,000 | 15,071,398,400 |
| 16 | 13,280,000,000 | 14,551,398,400 |
| 32 | 12,740,000,000 | 14,011,398,400 |

These are arithmetic starting points, not a substitute for the existing `MIN_FREE_MIB=384` post-pre-warm guard. Lower `KV_BYTES` if that guard fails; do not raise it based only on boot-time free memory.

## Risk on metal, in operator order

1. **Boot residency:** first require the UVAOffloader, `1.18`, no-unmatched-selector, args/env, and memlock assertions above. This catches wrong image/CLI, UVA fallback, missing pinning, and selector drift before traffic.
2. **Embedding correctness:** run the fidelity ruler first. It is the fastest detector for a fused gather reading the UVA view incorrectly or checkpoint writes not reaching its backing.
3. **Long-context correctness:** run needles next; this catches rarer token-row/index and graph-shape failures.
4. **Decode behavior:** compare decode c1, then c8. Watch correctness, PCIe traffic, tokens/s, graph replay errors, and the ≥384-MiB free-VRAM guard. c1 exposes per-step UVA latency; c8 exposes contention.
5. **Prefill behavior:** run prefill last. It exercises many embedding rows at once and is the likely PCIe-bandwidth worst case, but it is less representative of the intended few-row decode win.

Open metal questions are the fused custom gather's behavior/performance on RTX 5090, actual Docker memlock behavior, and the absent draft config's exact dimensions. No source-only claim resolves those three.
