# BRIEF22 — R188: a per-layer Marlin (W4A16) allowlist for the NVFP4 GEMM path of vLLM 0.29.0rc2

You are working offline on a source dump. You have NO access to the serving host, no GPU, no model weights, no network. Do not try to import vllm or torch. One pass; deliver what is asked and stop.

## Why (measured evidence, do not re-derive)

Served engine: vLLM 0.29.0rc2 + our patch chain (image `vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616`), TP=2 on two RTX 5090 (sm_120), checkpoint `RedHatAI/Qwen3.8-27B-NVFP4` (compressed-tensors, mixed precision), DFlash2 speculative decoding with a W4A16 drafter.

R183b (2026-09-04) walked the NVFP4 GEMM kernel ladder with `VLLM_DISABLED_KERNELS` on the automatic path (fp8 and W4A16 layers untouched). The arm that disabled FlashInferCutlass + FlashInferB12x + Cutlass made the selector pick `MarlinNvFp4LinearKernel` for EVERY NVFP4 layer; the engine booted, KV pool unchanged (1,020,596 tokens), 0 error lines, and the teacher-forced dense ruler against the bf16 reference read:

| kernel for all NVFP4 layers | top-1 agreement vs bf16 | corpus PPL delta vs bf16 | truncated KL mean |
|---|---|---|---|
| FlashInferCutlass (served, W4A4) | 92.771% | +0.744% | 0.014082 |
| Marlin (W4A16: FP4 weights dequantized, bf16 activations) | 94.051% | +0.207% | 0.010046 |

Marlin's cost, same pool: 8-stream decode −7.8%, 16-stream −17.3%, TTFT at 100K tokens +32%. The all-or-nothing choice is too expensive; we want to run Marlin on a SUBSET of the NVFP4 layers (e.g. only `down_proj`, or only some layer indices) and keep the served W4A4 kernel elsewhere, then measure which subset buys most of the 0.54 pp for the least throughput. This brief asks only for the mechanism: an env-driven, name-regex allowlist that forces Marlin per layer.

## What the checkpoint quantizes (from its config.json, `checkpoint-config.txt` here)

- `group_0` FP8 W8A8 (float-quantized, channel weights, dynamic per-token activations): `re:.*self_attn\.(q|k|v|o)_proj$`, `re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$`, `re:.*lm_head`, `re:.*layers\.(56|57|58|59|60|61|62|63)\.mlp\.(gate|up|down)_proj$`.
- `group_1` NVFP4 W4A4 (`nvfp4-pack-quantized`, tensor_group 16, static input global scale): `re:.*mlp\.(gate|up|down)_proj$` — after `group_0` takes layers 56–63, that is layers 0–55 × {gate, up, down}: the ONLY NVFP4 W4A4 layers in the model. Each carries `weight_packed`, `weight_scale`, `weight_global_scale`, `input_global_scale` in the checkpoint.
- `ignore`: visual tower, GDN `in_proj_a/b`, norms, `re:^mtp.*`. KV scheme fp8.

In vLLM the Qwen3-Next model fuses `gate_proj`+`up_proj` into one `MergedColumnParallelLinear` named `...mlp.gate_up_proj` (`packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}`), and `down_proj` is a `RowParallelLinear`. The prefix the quant config sees for a layer is the vLLM module name, e.g. `model.language_model.layers.17.mlp.gate_up_proj` / `...layers.17.mlp.down_proj` (verify the exact prefix form in `compressed_tensors.py`: `get_scheme(layer=layer, layer_name=prefix)` and how `get_scheme_dict` resolves fused names via `packed_modules_mapping`). The allowlist must match vLLM module names, and your NOTES must state the exact string form a user should target (with the `model.language_model.` prefix or not — read the code and say which).

## How the served tree selects the kernel (read these files; `src/` = the `vllm/` package subtree of the served image, so `src/X` is `vllm/X`)

- `src/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py`: `get_scheme(layer, layer_name=prefix)` is called once PER LAYER (lines ~160/182/192/211); `_get_scheme_from_parts(..., layer_name=...)` returns `CompressedTensorsW4A4Fp4(use_a16=True)` when the checkpoint has no input quant (W4A16 checkpoints, e.g. the drafter) and `CompressedTensorsW4A4Fp4()` for W4A4 (line ~743).
- `src/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py`: `__init__(use_a16=False)` calls `init_nvfp4_linear_kernel(use_a16=use_a16)` once per scheme instance, i.e. once per layer. `create_weights` registers `input_global_scale` only when `not use_a16`; `process_weights_after_loading` computes the input scales only when `not use_a16`, then calls `self.kernel.process_weights_after_loading(layer)`.
- `src/model_executor/kernels/linear/__init__.py`: `init_nvfp4_linear_kernel(use_a16)` — `a16_kernels = (Marlin, Humming)`; `--linear-backend auto` + `use_a16` forces Marlin; otherwise walks `_POSSIBLE_NVFP4_KERNELS[CUDA]` in priority order (FlashInferCuteDsl, FlashInferCutlass, FlashInferB12x, Cutlass, Marlin, ...), honouring `VLLM_DISABLED_KERNELS` and `--linear-backend`; logs `Using %s for NVFP4 GEMM` with `info_once`.
- `src/model_executor/kernels/linear/nvfp4/marlin.py`: `MarlinNvFp4LinearKernel` — `can_implement` always True; `process_weights_after_loading` = `prepare_fp4_layer_for_marlin(layer)`; `apply_weights` = `apply_fp4_marlin_linear(weight, weight_scale, weight_global_scale, workspace, ...)`. R183b proved this kernel accepts the parameters created by the W4A4 (`use_a16=False`) path — that arm ran exactly this: W4A4 `create_weights`/`process_weights_after_loading`, Marlin selected by the ladder.
- `src/envs.py`: our chain's env knobs follow the `VLLM_SM12X_*` pattern (see `VLLM_SM12X_GDN_PACKED_BV`, declared at ~line 130 and parsed at ~line 1208). Add the new knob the same way.

## Design constraints (decide the rest yourself, write the reasoning in NOTES22.md)

1. New env `VLLM_SM12X_NVFP4_MARLIN_LAYERS` — a Python regex. Unset or empty = current behaviour, byte-for-byte the same selection for every layer (this is the served daily; nothing may change when the knob is unset). Set = for each NVFP4 W4A4 layer whose vLLM name matches (`re.search`, or `fullmatch` — pick one and document it), the scheme's kernel is `MarlinNvFp4LinearKernel`; all other layers keep the normal auto selection.
2. Do NOT implement it by flipping `use_a16=True` for matched layers: the checkpoint carries `input_global_scale` for those layers and the W4A4 `create_weights`/`process_weights_after_loading` paths are what R183b validated under Marlin. Keep the W4A4 scheme path and force the KERNEL per layer instead (thread `layer_name` from `_get_scheme_from_parts` into `CompressedTensorsW4A4Fp4(use_a16=False, layer_name=...)` and into `init_nvfp4_linear_kernel(...)` as a new keyword, default None, so every other caller is unchanged).
3. Only the compressed-tensors W4A4 scheme gets the allowlist (the checkpoint is compressed-tensors). Leave `modelopt.py`, `quark_nvfp4.py`, MoE paths alone unless a shared signature forces a default-preserving change.
4. If the regex matches and Marlin `is_supported()` is False, raise a clear `ValueError` at boot (never silently fall back — the experiment must fail loudly).
5. `VLLM_DISABLED_KERNELS` containing `MarlinNvFp4LinearKernel` together with a matching allowlist = raise. `--linear-backend` other than `auto` with a matching allowlist = raise (document).
6. Proof of path: for every matched layer log ONE line at INFO with a fixed greppable prefix, e.g. `NVFP4 Marlin allowlist: MarlinNvFp4LinearKernel for model.language_model.layers.17.mlp.down_proj`, and one summary line after selection is impossible without a hook, so instead log the regex once (`info_once`) when the first match occurs. The per-layer count of those lines is how the host operator proves the arm ran (expected counts: down_proj-only regex → 56 lines per rank; gate_up-only → 56; all → 112). Do not use `info_once` for the per-layer lines.
7. Invalid regex → `ValueError` at import/parse time with the regex in the message.
8. Memory: a matched layer's weights go through `prepare_fp4_layer_for_marlin` (repacked in place); unmatched layers keep the FlashInfer/Cutlass layout. State in NOTES whether the Marlin repack keeps or drops the W4A4-only parameters (`input_global_scale`, `input_global_scale_inv`, any FlashInfer-side workspace) and whether anything remains allocated twice. R183b's all-Marlin arm had the same KV pool as the served kernel, so per-layer mixing should not lose pool; say so if the code agrees.

## Deliverables (write them into `deliver/` under this directory)

1. `deliver/0139-nvfp4-marlin-allowlist-v0290.diff` — unified diff with `a/vllm/...` / `b/vllm/...` paths, applying with `patch -p1 --fuzz=0` from the site-packages root (the directory containing `vllm/`), like `patches-v0290/0137-fs-tier-eviction-v0290.diff` (its header: `--- a/vllm/v1/kv_offload/tiering/fs/manager.py` / `+++ b/vllm/...`). Base = the files in `src/` exactly as they are (they are the served image's files after our chain, so the diff must apply to THEM, not to upstream rc2).
2. `deliver/Dockerfile.marlinlist` — a layer on top of `ARG BASE` (the served image), same shape as the excerpt below: assert `vllm==0.29.0rc2`, find the package root with `importlib.util.find_spec("vllm")`, `patch --batch --forward -p1 --fuzz=0`, `python3 -m py_compile` every touched file, `echo PRS-0139-MARLIN-ALLOWLIST-APPLIED`, no ENV that enables the path (launch-time opt-in only).
3. `deliver/verify-0139.sh` — offline verification only: copy `src/` into a scratch tree as `.work22/base/vllm/...`, `patch --dry-run -p1 --fuzz=0` there (must apply clean; also run the REAL apply on the scratch copy and `py_compile` each touched file), then a pure-Python test of the matcher that does not import vllm or torch (put the regex/match helper in a small dependency-free module, e.g. `vllm/model_executor/kernels/linear/nvfp4/allowlist.py`, and test: unset → no match; `layers\.(\d+)\.mlp\.down_proj$` matches `model.language_model.layers.17.mlp.down_proj` and not `...gate_up_proj`; invalid regex raises). The script must exit 0 only if everything passes. Run it yourself before you finish and paste its output at the end of NOTES22.md.
4. `deliver/NOTES22.md` — design, exact env semantics and the module-name form to target, the three regexes for the arms we will run (`down_proj` only, all 56; `gate_up_proj` only, all 56; layers 0–18 all three projections — write them out so they can be pasted into `-e VLLM_SM12X_NVFP4_MARLIN_LAYERS='...'`), the expected log-line count per arm, what was NOT verified (no GPU, no model), and open risks (e.g. anything in `prepare_fp4_layer_for_marlin` that assumes ALL layers are Marlin, torch.compile/cudagraph interactions with two kernels in one graph, `expose_input_quant_key`).

Dockerfile shape to copy (from our `Dockerfile.pcieipc`):
```
ARG BASE
FROM ${BASE}
COPY 0139-nvfp4-marlin-allowlist-v0290.diff /opt/marlinlist-layer/
RUN --network=none set -eux; \
    python3 -c 'from importlib.metadata import version; assert version("vllm") == "0.29.0rc2"'; \
    PKG_ROOT="$(python3 -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("vllm").origin).parent.parent)')"; \
    cd "$PKG_ROOT"; \
    patch --batch --forward -p1 --fuzz=0 < /opt/marlinlist-layer/0139-nvfp4-marlin-allowlist-v0290.diff; \
    python3 -m py_compile <touched files>; \
    rm -rf /opt/marlinlist-layer; echo PRS-0139-MARLIN-ALLOWLIST-APPLIED
```

## Scope discipline

- Verify ONLY with `patch --dry-run` / real apply on the scratch copy, `py_compile`, and the dependency-free matcher test. No tests that import vllm/torch, no model execution, no GPU. One pass. Keep the diff small (target < 150 lines) and readable.
- Do not touch anything outside `deliver/` and `.work22/`. Do not modify `src/`.
- If something in the dump contradicts this brief, trust the dump and say so in NOTES22.md.
