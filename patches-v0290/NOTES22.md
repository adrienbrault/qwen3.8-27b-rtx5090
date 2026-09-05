# R188 / PRS-0139: per-layer NVFP4 Marlin allowlist

The patch is against the supplied served-tree `src/`, not upstream rc2. It changes four existing files and adds one dependency-free helper (138 unified-diff lines). Apply from the directory containing `vllm/` with `patch --batch --forward -p1 --fuzz=0`. Build with `deliver/` as the Docker context and the served image as `BASE`. The Dockerfile adds no enabling ENV; opt in at launch.

## Selection and environment semantics

`VLLM_SM12X_NVFP4_MARLIN_LAYERS` is a case-sensitive Python regex using `re.search`. Unset or exactly empty means no matches. Whitespace is literal, not trimmed; do not include the checkpoint-style `re:` marker. An absent layer name cannot match. The environment registry parses the value into a compiled regex; malformed syntax raises `ValueError` naming the variable and including the offending regex at parse time, even before attempting a match. As with existing `envs.py` getters, parsing is lazy on environment access (or environment-cache initialization), not necessarily at Python module import. Launch-time settings should remain fixed throughout the process.

The helper lives at `vllm/nvfp4_marlin_allowlist.py` so `envs.py` can import it without initializing the heavy `model_executor.kernels.linear` package. It imports only the standard library. Parsing is cached by value; no per-layer compiled-regex allocation is retained.

Only the compressed-tensors NVFP4 **W4A4** construction site passes `layer_name`. Its scheme remains `use_a16=False`; checkpoint loading, input scale processing and `expose_input_quant_key` are unchanged. Both constructor and selector accept the new keyword-only `layer_name=None`. W4A16 construction still passes only `use_a16=True`. ModelOpt, Quark, MoE, FP8 and unquantized paths are unmodified. Other selector callers omit the name and bypass the new branch.

A match instantiates Marlin before the existing selector ladder. Unsupported Marlin raises `ValueError` with the layer and support reason. A matching layer also raises for Marlin in `VLLM_DISABLED_KERNELS`, any `--linear-backend` other than `auto` (including explicit `marlin`), or `VLLM_BATCH_INVARIANT=1`. The last conflict is an additional design decision: the supplied tree otherwise forces deterministic CUTLASS/emulation, so silently taking either override would break the other request. Conflict checks precede support checks. No fallback occurs for matches. Nonmatches execute the original selector, including its existing backend, disabled-kernel and batch-invariant handling. With the knob unset/empty, every layer gets the same selection as before; no new selection logs are emitted.

## Names and launch arms

The selector receives the original vLLM `prefix`, unchanged: `get_quant_method(..., prefix)` calls `get_scheme(layer=layer, layer_name=prefix)`, which forwards that same name into `_get_scheme_from_parts`. `get_scheme_dict` passes `packed_modules_mapping` to `find_matched_target` to resolve checkpoint targets for fused modules; it does not replace the caller's name with the matched checkpoint target.

For the model described in the brief, target strings are exactly:

```
model.language_model.layers.17.mlp.down_proj
model.language_model.layers.17.mlp.gate_up_proj
```

There is no prefix stripping in this path: an anchored expression must include `^model\.language_model\.` for these strings. The suffix-search expressions below intentionally do not need that prefix. Match `gate_up_proj`, not the separate checkpoint `gate_proj` and `up_proj`; they share one kernel selection.

Evidence limit: `checkpoint-config.txt` confirms the `model.language_model.` checkpoint namespace, and `compressed_tensors.py` confirms that the runtime prefix is preserved. The dump omits the model constructors and the implementation of `find_matched_target`; consequently the actual model-generated runtime prefix and fused-mapping algorithm cannot be independently verified here. The exact model names above use the brief's supplied runtime naming. The suffix expressions work whether that root prefix is present or absent.

Paste one of these launch options (explicit index limits exclude FP8 layers 56–63):

**Down projection, layers 0–55: 56 per-layer INFO lines per rank.**

```sh
-e VLLM_SM12X_NVFP4_MARLIN_LAYERS='layers\.([0-9]|[1-4][0-9]|5[0-5])\.mlp\.down_proj$'
```

**Fused gate/up projection, layers 0–55: 56 lines per rank.**

```sh
-e VLLM_SM12X_NVFP4_MARLIN_LAYERS='layers\.([0-9]|[1-4][0-9]|5[0-5])\.mlp\.gate_up_proj$'
```

**All three checkpoint projections, layers 0–18 inclusive: 38 lines per rank (19 × two fused runtime modules, representing 57 checkpoint projections).**

```sh
-e VLLM_SM12X_NVFP4_MARLIN_LAYERS='layers\.([0-9]|1[0-8])\.mlp\.(gate_up|down)_proj$'
```

For an all-NVFP4 control, use `layers\.([0-9]|[1-4][0-9]|5[0-5])\.mlp\.(gate_up|down)_proj$`: 112 lines per rank. These counts assume the checkpoint assignment in the brief and one construction per module per rank. FP8 layers cannot be forced by a broad regex because their scheme never enters this branch. No-match regexes are valid and produce no allowlist logs.

Each successful matched selection uses ordinary `logger.info`, once per construction:

```
NVFP4 Marlin allowlist: MarlinNvFp4LinearKernel for model.language_model.layers.17.mlp.down_proj
```

The first successful match also calls `info_once` with `NVFP4 Marlin allowlist regex: <regex>`. Count only the exact per-layer message prefix `NVFP4 Marlin allowlist: MarlinNvFp4LinearKernel for `, separately per rank; the regex line is excluded. TP=2 totals are 112, 112, 76 (224 for the control) if both ranks' INFO output is collected. Logging configuration/rank filtering may hide emitted records. These are selection records, not proof that loading or execution subsequently succeeded.

## Weight layout, memory and open risks

The CT W4A4 path registers `input_global_scale`, renames `weight_packed` to `weight` and deletes the old attribute, normalizes weight/input global scales, and creates `input_global_scale_inv` and `alpha`. It then calls only the selected kernel's `process_weights_after_loading`. Matched layers therefore call `prepare_fp4_layer_for_marlin(layer)` directly; they never first build a FlashInfer layout. Unmatched layers retain normal kernel preparation. No second weight copy, alternate kernel instance, or FlashInfer workspace is introduced by this patch.

The available FlashInfer CUTLASS preparation replaces weight/scales with padded/swizzled parameters; it does not attach a per-layer workspace. Runtime workspace behavior inside its imported wrapper is outside the dump. Marlin's available wrapper reads `layer.weight`, `weight_scale`, `weight_global_scale` and `workspace`; it does not read `input_global_scale`, `input_global_scale_inv` or `alpha` in `apply_weights`.

**Missing-source limitation:** `src/model_executor/layers/quantization/utils/marlin_utils_fp4.py` is absent. Thus this dump cannot establish whether the actual Marlin repack keeps or drops the three W4A4-only scale/alpha parameters, retains any old weight storage, creates temporary/persistent copies, or assumes all layers use Marlin. The visible CT code leaves those parameters present on entry to the helper; this patch neither deletes them nor changes helper behavior. Claims that repacking is in-place or leaves no duplicate allocations beyond this boundary would require the omitted implementation. The wrapper API is per-layer, with no visible all-model assumption.

R183b's measured unchanged 1,020,596-token pool supports the expectation that mixing should preserve the pool. The visible code introduces no dual layouts for a matched layer, consistent with that expectation, but the missing repack/workspace internals and lack of runtime access prevent confirming equal pool capacity or memory peaks for mixed kernels. This remains a host measurement, not an offline result.

Marlin inherits `input_quant_key() -> None` from `NvFp4LinearKernel`; FlashInfer CUTLASS advertises `kNvfp4Dynamic`. `expose_input_quant_key(layer, self.kernel)` sees the final selected kernel before weight loading, which is the intended distinction for activation fusion. Its implementation and compiler fusion passes are absent, so avoidance of prequantized activations on matched layers is not verified. Mixed-kernel torch.compile specialization, fusion, cudagraph capture/workspaces, numerical quality, throughput, and TTFT remain unverified. Marlin's existing GPU-support warning text is unchanged, even when explicitly selected on sm_120.

## Offline verification

Only the requested scratch-copy dry-run/apply, `py_compile`, and dependency-free matcher tests were executed. No vllm/torch imports, network, GPU, weights, model execution, serving-host access, or Docker build. The script loads the helper by file location without traversing package imports and also checks the three arm counts and the all-NVFP4 control using synthetic names. It does not execute selector support/conflict/logging branches; those were inspected in source only.

Run `./deliver/verify-0139.sh` from any directory. It replaces only `.work22/base`, copies `src/` into `.work22/base/vllm/`, dry-runs and applies the patch with zero fuzz, compiles every diff target, then runs the matcher tests. Any failure exits nonzero. `src/` was not modified.

Actual verification output (exit 0):

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
PASS: offline verification complete; no vllm/torch imports

```

## Addendum 2026-09-05 (R188 first run): env value must be a primitive

The first R188 run (03:34 UTC) booted CTRL but every Marlin arm died in `profile_run` with `TypeError: normalize_value: unsupported type 'Pattern'` from `vllm/envs.py compile_factors()` — the AOT compile hash walks every registered env and the 0139 lambda returned a compiled `re.Pattern`. Fix (same date): the env entry is `str | None` (the raw regex text, itself a correct compile factor since it changes the GEMM kernel per layer), and `init_nvfp4_linear_kernel` calls `parse_allowlist()` (lru_cached) on it. The `import re` / module import added to `envs.py` are gone, so 0140's two `envs.py` hunks moved up three lines and their context lines were updated. Image `…-fi0616-marlinlist` rebuilt from the fixed diff; `docker run` check: `type(envs.VLLM_SM12X_NVFP4_MARLIN_LAYERS)` is `str`, `normalize_value` accepts it, matcher unchanged. R188 re-queued behind r190b.
