# R190 / 0140 — offline GEMM census and shared-layout dispatch

No performance result is claimed. First number to inspect: **percent reduction in graph-replay median µs at the actual padded c8 M (nominally 80), versus the automatically selected NVFP4 baseline**. If no kernel beats the current kernel by **more than 5% at any served M**, shape dispatch is dead. Do not enable an empty/no-benefit table merely because it builds.

## Packaging and installation

- `0140-nvfp4-shape-dispatch-v0290.diff` applies **on top of 0139**. The supplied primary Dockerfile applies the unchanged, included 0139 dependency and then 0140 to the default pcieipc base. Do not use that Dockerfile on a base already containing 0139 (apply only 0140 there).
- `0140a-nvfp4-shape-dispatch-v0290.diff` applies directly to `src/` / the pcieipc image without 0139. Its standalone Dockerfile applies only 0140a. Choose one route; never apply both.
- Build context is `deliver/`: `docker build --network=none -f Dockerfile.nvfp4-shape-dispatch -t vllm-gemm140 .`. Dockerfiles set no enabling ENV and write `/opt/prs-markers/0140`. Docker frontend/base must already be available offline.
- `sh deliver/verify-0140.sh` copies source into `.work/0140/vllm` and `.work/0140a/vllm`, checks dry-run and real apply exit status, compiles all touched Python files (including 0139), runs dependency-free tests in both trees, and checks census reproducibility. `src/` and evidence remain untouched. Scratch directories are replaced on rerun.

## Source findings and census

The brief's ungated arithmetic `(24*256 + 2*4*256)/2` is **4096**, not 3584. `Qwen3NextAttention.__init__` doubles Q for `attn_output_gate=true`, making the actual fused QKV N **7168**. `QwenGatedDeltaNetAttention.create_qkvz_proj` fuses QKV and Z: actual N **8192**, K 5120. `create_ba_proj` fuses B and A, N **48**, K 5120, unquantized with this CT config. These are significant corrections to the proposed budget.

`build_shape_census.py` statically expands audited source constructors and evaluates the evidence quantization regexes, including fused checkpoint-name mappings. Its output `gemm_shape_census.md` enumerates every decoder/head linear, the GDN conv1d Linear-backed weight, and image-prefill vision linears under default weight TP2. Vision data parallel mode would change those vision shapes; it is not asserted by the supplied launcher. Norms, embeddings and conv3d patch embedding are not GEMMs. The drafter and MTP checkpoint tensors are not target decoder modules.

Per rank, NVFP4 payload is 50,135,040 bytes/gate_up and 25,067,520 bytes/down (packed data + E4M3 scales). There are 56 of each. Last eight MLPs are FP8. Decoder/head payload including GDN B+A and conv weights totals **9,553,810,432 bytes/rank**, excluding global scalar metadata, bias, embeddings, norms, workspaces and activation traffic. This is a static storage/traffic proxy, not measured HBM reads.

`CompressedTensorsW8A8Fp8.create_weights` calls `init_fp8_linear_kernel` with `kFp8StaticChannelSym` and `kFp8DynamicTokenSym`. FlashInfer's FP8 candidate rejects those scales; with the normal auto ladder, `CutlassFP8ScaledMMLinearKernel` is selected. The benchmark calls that selector rather than assuming a class. Its weights follow the channel post-load transpose to column-major (K,N), and it calls kernel post-load preparation and apply including per-token activation quantization.

## Running the GPU census

Scripts run **inside** the served container, with the engine stopped and GPU 0 free (or use a separate instance of the same image). No distributed initialization or vLLM engine. Copy/mount all deliver scripts into a directory such as `/opt/gemm140`; execute there:

```sh
python nvfp4_gemm_census.py --json census-graph.json > census-graph.md
python nvfp4_gemm_census.py --mode eager --json census-eager.json > census-eager.md
python nvfp4_gemm_census.py --shapes gate_up,down --ms 10,80,160 \
  --kernels FlashInferCutlassNvFp4LinearKernel,CutlassNvFp4LinearKernel,FlashInferB12xNvFp4LinearKernel,FlashInferCudnnNvFp4LinearKernel \
  --json shortlist.json > shortlist.md
python make_dispatch_table.py census-graph.json > dispatch.json
```

Default M: 1,2,4,8,10,16,20,32,40,64,80,128,160,256,2048,8192. Default 10 warmups, median 30 CUDA-event observations; fewer than 20 iterations is rejected. Default graph mode captures **apply_weights** and times replay, eager mode times apply directly. Kernel preprocessing is outside timing; activation quantization is inside. GPU, capability and torch/vLLM/FlashInfer versions are printed and serialized. The markdown table includes µs, decimal effective TB/s = weight bytes/time, and effective TFLOPS = 2MNK/time. JSON is checkpointed during the sweep.

Every CUDA NVFP4 ladder class is enumerated: FI CuteDSL, FI CUTLASS, FI B12x, native CUTLASS, Marlin, TRT-LLM, cuDNN, FBGEMM, native B12x, emulation and Humming. Each receives synthetic uint8 E2M1 packed (N,K/2), E4M3 scales (N,K/16), and post-CT-inversion scalar global scales, then its actual `process_weights_after_loading` and `apply_weights`. Marlin gets params_dtype and partition metadata; Humming gets output_partition_sizes/has_bias and uses its CT adapter. No class is pre-skipped for a supposedly unavailable checkpoint layout: this source exposes constructible inputs for all. CuteDSL rejects SM120 (SM10x only); optional packages and backend-specific SM constraints are checked at runtime. `SKIP/FAIL` records include class, shape, optional M, and reason. A runtime rejection is **not** evidence that a backend ran slowly. Failed candidates are never selected by the generator. A fatal CUDA error can poison the context; rerun an affected subset in a fresh process.

NVFP4 candidates benchmarked explicitly may include classes disabled by the serving environment; baseline still comes from the normal ladder. Dispatch rejects disabled choices. The only FP8 candidate is `FP8Auto`, the exact served channel/token path; this is not a separate FP8 dispatch experiment. Also benchmark FP8 MLPs 56–63 and fused GDN QKVZ. Ungated QKV and separate GDN QKV/Z are diagnostic component shapes only.

All weights and activation inputs are synthetic, BF16 input/output; scalar scales are one, group scales are random finite positives. Same seed resets raw weight generation per candidate. Finite-output checks are smoke checks, not numerical validation. Buffers are reused and may be warm in cache; roofline TB/s is an effective metric and may exceed a streaming bandwidth measurement. Marlin/Humming are W4A16 and have different activation arithmetic. Their speed alone cannot justify an accuracy claim.

Fill in graph median µs for each actual shape/M/kernel, skipped reasons, baseline class, resolved padded capture M, per-rank free memory, and profiled invocation counts. Use census multiplicities to estimate the target GEMM subtotal; avoid double-counting component shapes, vision, and LM-head calls at different M. Run the shortlist three times and compare medians/variation before choosing winners. Check prefill M2048/8192 too. No measurements are fabricated in the delivered JSON/table generator.

## Dispatch semantics and changed functions

Knob: `VLLM_SM12X_NVFP4_DISPATCH_TABLE=/absolute/container/path/dispatch.json`. Unset returns the existing kernel with the existing tensors and activation-fusion contract; file reads and wrappers are not created. Empty string, missing/unreadable file, malformed JSON, extra keys, empty/invalid regex, invalid limit or unapproved kernel raises. The file is read once and cached: restart after changing it.

```json
{"rules":[
  {"layer":"\\.mlp\\.gate_up_proj$", "m_max":80, "kernel":"CutlassNvFp4LinearKernel"},
  {"layer":"\\.mlp\\.gate_up_proj$", "m_max":null, "kernel":"FlashInferCutlassNvFp4LinearKernel"}
]}
```

This is grammar only, **not a recommended measured table**. Regex uses `search` on the module prefix. Limits are inclusive positive integers (bool forbidden); null is unbounded. First matching file rule wins. No matching rule at a runtime M uses the pre-existing default kernel, explicitly represented in an unbounded bucket. Unmatched modules keep their original kernel. Every class in a matching module rule is validated at initialization even if an earlier rule shadows it. The source's `NvFp4LinearLayerConfig` contains no dimensions and all these `can_implement` methods return true; initialization checks are precisely as strong as the supplied kernel interfaces, not a substitute for shape smoke tests.

Allowed shared-layout classes: `CutlassNvFp4LinearKernel`, `FlashInferCutlassNvFp4LinearKernel`, `FlashInferB12xNvFp4LinearKernel`, `FlashInferCudnnNvFp4LinearKernel`. Their source postprocessors perform the same `swizzle_blockscale` and `pad_nvfp4_weight_for_cutlass`. Candidate objects are built at initialization; one common weight preparation serves all of them. **Zero additional packed-weight copies**. The FI CuteDSL class shares the layout but is excluded because its capability check forbids SM120. Native B12x, FBGEMM, TRT-LLM, emulation, Marlin and Humming use other representations and are excluded from B.

Duplicating every affected NVFP4 payload would cost at least **4,211,343,360 additional bytes/rank** before padding/workspaces, incompatible with the launcher's roughly 14 GB/rank pinned KV budget and narrow free-memory floor. Marlin cannot be compared through dispatch without a different memory-budget experiment. Backend workspaces and extra graphs can still consume memory despite sharing weights.

Unsupported combinations raise: non-SM12x, non-auto linear backend, batch invariance, any active 0139 allowlist, disabled/rejected candidate, non-shared default on an affected module, or a matching W4A16 module. Scope rules to target MLPs; target regexes must not accidentally reach a drafter. 0139's knob may exist but must be **unset** with dispatch. Switching activation quantizers can change numerics. For affected modules `input_quant_key()` is None, preventing an externally fused QuantizedActivation from being handed to a kernel expecting BF16. This can remove fusion and erase a microbench win; test integrated performance.

Changed existing functions:

- `CompressedTensorsW4A4Fp4.__init__`: conditionally wraps the existing selection. 0140 reuses 0139 layer_name plumbing; standalone adds that keyword and forwards it from `_get_scheme_from_parts` in compressed_tensors.py.
- `GPUModelRunner.load_model` in **both** `v1/worker/gpu_model_runner.py` and `v1/worker/gpu/model_runner.py`: reports the final target module census immediately after loading, before drafter construction and capture. Both runner variants are covered.
- `envs.py`: knob declaration/getter only. 0140 itself does not edit the NVFP4 ladder or any GPU kernel.
- New dependency-free `read_table`, `module_rules`, `buckets`; new `wrap_kernel`, `SharedLayoutDispatch` (prepare/input key/apply), `report_dispatch`.

Proof line, **once per process** (`scope="process"`, not local/global rank-zero suppression):

```text
NVFP4 dispatch table: <n> rules, <k> modules affected, kernels {<sorted class names>}
```

Two ranks give two lines. All target NVFP4 MLP modules means k=112; gate_up only means k=56. `NVFP4 dispatch module <prefix>: <bucket table>` is DEBUG at module initialization; use `VLLM_LOGGING_LEVEL=DEBUG` for those lines. The proof is an initialization census, not runtime branch counts. Inspect graph traces to prove the chosen kernels executed. k=0 means no target match and does not demonstrate a dispatch experiment.

## CUDA graphs, acceptance and measurement gates

`v1/cudagraph_dispatcher.py::_compute_bs_to_padded_graph_size` maps scheduled tokens to capture sizes; `_create_padded_batch_descriptor` uses that count and uniform decode query length `1+ns=10`. `gpu_model_runner.py` feeds sliced inputs at `batch_desc.num_tokens`. The newer `gpu/cudagraph_utils.py::_init_candidates` also derives candidates from resolved capture sizes and decode query length, and its runner loads this same model path. M here is **total activation rows**, computed as numel/last_dim, not concurrent requests.

Dispatch performs no device read, allocation, repack or table I/O during apply; its Python comparison depends only on tensor shape. For each fixed captured shape, the selected apply call is baked into that graph; replay does not rerun Python. Shape-dependent torch.compile guards/specializations must still be verified on the box. Bucket boundaries **do not have to align** with capture sizes for correctness. They must be fitted to **padded** M for performance: a threshold of 10 will not affect nominal c1 if its selected graph has 16 rows.

The supplied `launch-daily.sh` does **not** contain an explicit capture-size list. It delegates to `/srv/qwen5090/launch-daily-v0280.sh`, which is not supplied, and accepts `CC_EXTRA` in EXP mode. Therefore no exact served capture-size list can honestly be read from this evidence. Record resolved `compilation_config.cudagraph_capture_sizes` / graph capture logs at boot and add those M to `--ms`. Do not assume the microbench's default M grid is the graph grid.

The table generator chooses only >5% winners among B-compatible classes relative to `baseline_nvfp4`. It creates singleton winning M buckets with baseline guard buckets around them, so **unmeasured M and prefill retain the baseline kernel**. It deliberately does not extrapolate from M10 to M1..9. stderr prints the best-kernel-per-shape/M comparison; stdout is strict dispatch JSON. It rejects a non-shared baseline and duplicate samples; aggregate repeated trials explicitly first. Review kernel errors, variation and fidelity before mounting the file read-only and adding the knob through the launcher's EXP `EXTRA_ENV_APPEND` / `EXTRA_MOUNT_APPEND`.

Run A/B/A/B boots with the existing pcie_ipc path, same pool, same ns9/TP2. Use the operator's `decode_ss.py --conc 1 8 16 --tokens 1024 --runs 3 --kind code` and repeat prose, recording acceptance and steps/s as well as tokens/s; extend to >=5 seconds of steady decode where practical. Use `prof_decode_split.py` and `prof_summary.py` on matching R183-style decode traces. **Audit their kernel categorization**: existing substring heuristics can label a new FlashInfer GEMM as attention or Marlin as drafter. Expected direction is lower target GEMM µs/step and lower fixed-work step latency without a prefill/capacity penalty. Compare eager versus changing-input graph replays at each captured M, raw numeric differences, and matched-input decode fidelity at ctx0/30K before any promotion. Source-shared layouts alone do not prove numerical equality across backends.

Reject B if the >5% microbench criterion fails, if gains vanish in integrated target-GEMM time, if graph compilation/capture fails, if fidelity/acceptance leaves the accepted baseline envelope, or if graph/workspace memory violates the fixed pool/free-memory floor. A faster synthetic singleton GEMM is not sufficient evidence of end-to-end improvement.

## Offline result, limitations and provenance

Both patch routes passed fuzz-0 dry-run and actual apply, py_compile, 5 dependency-free tests each (strict parser failures, randomized first-match/bucket equivalence, exact wrapper preparation/3D-row routing with fake objects, config census math, conservative table generation), and reproducible static census output. No torch/vLLM imports, GPU execution, Docker build, compilation graph tracing, checkpoint load, kernel timing or numerical fidelity test was attempted offline.

New code is original glue, marked Apache-2.0 in patch modules. Layout and call conventions are derived from this supplied **vLLM 0.29.0rc2 + local patches 0101–0138 source dump**, vLLM project, Apache-2.0. Exact upstream commit is not supplied; do not invent a commit. The included 0139 diff is copied verbatim from evidence and retains its Apache-2.0 headers (local experiment, no upstream PR identified). FlashInfer 0.6.16, CUTLASS, Humming, FBGEMM, B12x and Marlin are invoked through installed vLLM adapters; no third-party GPU code is copied or vendored by 0140. For THIRD_PARTY.md: record “0140 shared-layout NVFP4 dispatch/census; original Apache-2.0 glue based on supplied vLLM 0.29.0rc2 adapters; optional local 0139 dependency.” Retain the image's existing third-party notices for backend implementations.
