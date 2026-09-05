# R188 / PRS-0139B: selectable per-layer NVFP4 W4A16 kernel

`0139b-nvfp4-a16-allowlist-v0290.diff` is a standalone replacement for 0139 against the supplied `src/`, with 146 unified-diff lines and the same five target files. Apply it to the served base before 0139, not on top of 0139. The original deliverables are preserved. Build `Dockerfile.a16list` with `deliver/` as context and the served image as `BASE`; it asserts vLLM 0.29.0rc2, locates the package root via `find_spec`, applies with `--batch --forward -p1 --fuzz=0`, compiles every target, and emits `PRS-0139B-NVFP4-A16-ALLOWLIST-APPLIED`. No enabling ENV is baked into the image.

## Differences from amended 0139

The compatibility knob `VLLM_SM12X_NVFP4_MARLIN_LAYERS` retains case-sensitive Python `re.search` semantics. Unset or exactly empty disables the override; whitespace is literal and the checkpoint `re:` marker must not be included. `VLLM_SM12X_NVFP4_A16_KERNEL` selects exactly `marlin` (unset default) or `humming`. Empty strings, other spelling/case, and surrounding whitespace are invalid kernel values. The setting affects only matching compressed-tensors NVFP4 W4A4 layers; it does not change the default kernel for native `use_a16=True` layers, including the drafter.

Both env getters return primitives only: regex text or None, and raw kernel text. No compiled regex enters `envs.compile_factors()` hashing. The dependency-free helper retains its existing filename, `vllm/nvfp4_marlin_allowlist.py`, and adds `parse_a16_kernel`. The selector parses the regex on entry to a named W4A4 selection, before matching; invalid syntax raises ValueError with the regex and variable name. Kernel text is validated on a match, with a ValueError naming the variable and invalid value. This is lazy selector-time parsing, not env-module import-time validation, following the amended 0139 supplied for this task. With no match the new kernel setting has no selection effect.

The selected class is taken from the existing `(MarlinNvFp4LinearKernel, HummingNvFp4LinearKernel)` a16 tuple. A matching layer fails with ValueError for any non-auto linear backend (even an explicit backend agreeing with the choice), the chosen class in `VLLM_DISABLED_KERNELS`, or enabled `VLLM_BATCH_INVARIANT`. Disabling the other class does not conflict. The chosen class's `is_supported()` is checked and its failure reason included in the error; no fallback occurs. Unmatched layers execute the original ladder. ModelOpt, Quark, MoE, FP8, and native W4A16 construction remain unchanged.

A successful first match logs via `info_once`:

```text
NVFP4 A16 allowlist regex: <regex> kernel=HummingNvFp4LinearKernel
```

Each successful matched construction logs via ordinary `info`:

```text
NVFP4 A16 allowlist: HummingNvFp4LinearKernel for model.language_model.layers.17.mlp.down_proj
```

Marlin uses the same forms with `MarlinNvFp4LinearKernel`. Count the per-layer prefix, excluding the regex line, separately per rank. These records prove selection, not successful loading/execution. INFO filtering may hide a rank's records.

## Exact launch settings and names

Humming with the M-ALL regex from `evidence-r188.txt` (112 runtime projections per rank, 224 across TP2 if both ranks' logs are collected):

```sh
-e VLLM_SM12X_NVFP4_A16_KERNEL=humming \
-e VLLM_SM12X_NVFP4_MARLIN_LAYERS='layers\.([0-9]|[1-4][0-9]|5[0-5])\.mlp\.(gate_up|down)_proj$'
```

Humming with gate/up only (56 per rank, 112 across TP2):

```sh
-e VLLM_SM12X_NVFP4_A16_KERNEL=humming \
-e VLLM_SM12X_NVFP4_MARLIN_LAYERS='layers\.([0-9]|[1-4][0-9]|5[0-5])\.mlp\.gate_up_proj$'
```

The supplied evidence file contains M-ALL but no gate_up arm line. The second regex is the gate_up restriction of that exact M-ALL expression, also present in the prior verifier. Use `marlin`, or omit the kernel env, for the corresponding Marlin arms. Keep settings fixed for the process lifetime.

For completeness, down-only and layers 0–18 all checkpoint projections remain:

```sh
-e VLLM_SM12X_NVFP4_MARLIN_LAYERS='layers\.([0-9]|[1-4][0-9]|5[0-5])\.mlp\.down_proj$'
# 56 per rank
-e VLLM_SM12X_NVFP4_MARLIN_LAYERS='layers\.([0-9]|1[0-8])\.mlp\.(gate_up|down)_proj$'
# 38 per rank: 19 layers x two fused runtime modules
```

`compressed_tensors.py` passes `prefix` unchanged through `get_scheme` into `_get_scheme_from_parts`; `get_scheme_dict` uses `packed_modules_mapping` when finding the checkpoint target, without replacing that runtime name. Target the brief's runtime strings `model.language_model.layers.17.mlp.gate_up_proj` and `model.language_model.layers.17.mlp.down_proj`. A fully anchored expression for those strings must include `^model\.language_model\.`; these suffix-search expressions do not require the root prefix. Gate and up are fused and cannot be selected independently. The model constructors and matching utility implementation are absent from this dump, so the runtime root prefix cannot be independently established beyond the supplied brief. Indices 56–63 are FP8 and never enter this override.

## Humming processing and SM120 support

The served CT scheme already dispatches processing generically via `self.kernel.process_weights_after_loading(layer)`, with no Marlin-only conditional. Matched layers stay `use_a16=False` so the W4A4 checkpoint's `input_global_scale` is registered and loaded. The scheme renames `weight_packed` to `weight`, reduces/inverts the global weight scale, processes input scales and alpha, then invokes only the selected kernel.

The supplied `nvfp4/humming.py` already implements the required native a16 processing path: restore `weight_packed` from `weight` and delete the old name; invert the scheme-normalized global weight scale back to the CT convention; call `prepare_humming_linear_layer_config` with compressed-tensors / `nvfp4-pack-quantized`, float 4-bit, group strategy, group size 16; obtain the Humming compute config; and allocate 1024 int32 locks on the weight device (4096 bytes per kernel instance). Its comment explicitly avoids a native group_tensor schema that mishandles scalar global scales. Runtime `apply_humming_linear` receives that layer config, compute config and locks. The same existing hook is used when a native `use_a16=True` layer selects Humming; the allowlist needs no duplicate repack or new scale conversion. Native auto W4A16 still forces Marlin in this tree.

Humming's `is_supported()` requires CUDA, `has_humming()`, and actual device capability >=75. Thus SM120 passes the architecture gate if Humming is installed; there is no SM120 exclusion. This is a source-level support result, not a GPU validation. The optional compute_capability argument is not consulted. `can_implement()` returns True. Both a16 kernels inherit `input_quant_key() -> None`; the unchanged `expose_input_quant_key(layer, self.kernel)` receives the final selected kernel before loading.

## Memory expectations and offline limits

Only one kernel and one preparation path are selected per layer; matched layers never first build a FlashInfer layout. The Humming wrapper stores its own configs and 4 KiB locks (448 KiB for 112 matched layers per rank, 224 KiB for gate_up only), whereas Marlin delegates to `prepare_fp4_layer_for_marlin` and uses `layer.workspace`. Neither visible wrapper explicitly deletes W4A4-only `input_global_scale`, `input_global_scale_inv`, or `alpha`; these exist on entry to repacking. There is no explicit retained dual weight layout added by this patch, and the visible rename deletes the previous weight attribute.

However, both `humming_utils.py` and `marlin_utils_fp4.py` are absent from the dump. The external Humming loader's precise packed scale layout, retained parameter cleanup, temporary allocations, workspace needs beyond locks, and any retained old storage cannot be inspected here. The visible API is per-layer, with no visible all-layers assumption; omitted internals could impose additional constraints. Humming memory cannot be ranked against Marlin solely from this wrapper.

Use R188's measured memory result rather than the earlier NOTES22 expectation of an unchanged pool: M-ALL needed the 13.5 GB KV pin; BRIEF22b reports only 347 MiB free at the 13.98 GB pin. The supplied evidence records successful M-ALL boot at pin=13500000000, pool=985621, min_free=153, versus control pool=1020596 at pin=13980000000. The 347 MiB figure is supplied by BRIEF22b, not a separate line in evidence-r188.txt. Do not assume Humming restores the larger pool; boot headroom and steady/peak memory remain host measurements. The expected W4A16 fidelity and Humming throughput improvement are hypotheses supported by the supplied experiment/census, not outcomes verified by this patch.

No GPU, model, weights, serving host, network, vllm/torch import, or Docker build was used. Humming installation, actual SM120 execution, CT loader handling of the extra W4A4 scales, numerical fidelity, throughput, TTFT, memory/KV capacity, mixed-kernel torch.compile/cudagraph behavior, and activation fusion through the omitted expose_input_quant_key implementation remain unverified. Selection conflicts, support handling, and log statements were inspected in source; the offline tests do not execute those runtime branches.

## Verification

Run `./deliver/verify-0139b.sh`. It recreates only `.work22/base/vllm` from `src/`, dry-runs and really applies the standalone patch with fuzz 0, py_compiles all five targets, and loads only the standard-library helper by file location. Tests cover unset/empty regex, down projection matching and fused-name exclusion, missing names, invalid regex, counts 56/56/38/112, default and both kernel values, and invalid kernel values. The actual two env getter lambdas are extracted with AST and evaluated against a fake getenv to verify primitive values and defaults without importing envs or vllm. Failure exits nonzero. `src/` was not modified.

Actual output (exit 0):

```text
Copied src/ to .work22/base/vllm/
patching file 'vllm/envs.py'
patching file 'vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py'
patching file 'vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py'
patching file 'vllm/model_executor/kernels/linear/__init__.py'
patching file 'vllm/nvfp4_marlin_allowlist.py'
patching file 'vllm/envs.py'
patching file 'vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py'
patching file 'vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py'
patching file 'vllm/model_executor/kernels/linear/__init__.py'
patching file 'vllm/nvfp4_marlin_allowlist.py'
py_compile: PASS (5 touched files)
Matcher: PASS (unset, empty, down_proj, fused-name exclusion, missing name, invalid regex)
Arm matcher counts: PASS (56, 56, 38; all-NVFP4 control 112)
Kernel/env parsing: PASS (actual primitive getters, default, both choices, invalid values)
PASS: offline verification complete; no vllm/torch imports
```
