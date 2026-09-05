# BRIEF23 — per-shape NVFP4/FP8 GEMM census microbench + shape-keyed kernel dispatch (patch 0140)
Read COMMON.md first. Implements R186-ANALYSIS.md §3 proposal 1 (census) and the "per-shape GEMM dispatch" lever (§2/§3 #2).
GEMM is 40%/37%/32% of the decode step at c1/c8/c16; today one NVFP4 kernel class is chosen once per process (the ladder in
`src/vllm/model_executor/kernels/linear/__init__.py:init_nvfp4_linear_kernel`, kernel classes in `kernels/linear/nvfp4/*.py`:
cutlass, flashinfer, marlin, humming, b12x, fbgemm, emulation) regardless of the GEMM shape. Different M (10/80/160 rows) and
N/K (gate_up 17408×5120 vs down 5120×8704) may prefer different kernels; nobody has measured it. Deliver two things:

## A. `deliver/nvfp4_gemm_census.py` — runs inside the served container, 1 GPU, no vLLM engine
- Builds each NVFP4 kernel class that `init_nvfp4_linear_kernel` could select on SM120 (read the ladder and each class's
  `can_implement`/`is_supported`), on random weights in the exact packed layout the class expects, for the per-rank shapes
  gate_up (N=17408,K=5120) and down (N=5120,K=8704), and ALSO the FP8 path used for attention/GDN projections (find the FP8
  scaled_mm kernel the compressed-tensors FP8 scheme uses on SM120: `kernels/linear/scaled_mm/`, `layers/quantization/
  compressed_tensors/schemes/`), shapes qkv (N=3584... derive: (24×256+2×4×256)/2=3584? compute it), o_proj (5120×3072), GDN
  in_proj_qkv (5120×5120), in_proj_z (3072×5120), out_proj (5120×3072), lm_head (124160×5120, FP8).
- Times M ∈ {1,2,4,8,10,16,20,32,40,64,80,128,160,256,2048,8192} per (kernel, shape) with CUDA events, warm-up, median of ≥20
  iters, and reports µs, effective TB/s (weight bytes / time), and TFLOPS. Emit one table (stdout, markdown) + JSON.
- Must instantiate the kernels through the SAME code path vLLM uses (the kernel class's `process_weights_after_loading` +
  `apply_weights`, or the lowest-level function each class wraps), so the census predicts serving behaviour. If a class needs
  weights from a real checkpoint layout you cannot construct, say which and skip it with a printed reason.
- Print the torch/vllm/flashinfer versions and the GPU name at the top. Argument `--shapes`, `--kernels`, `--ms` to subset.
- Also emit a per-layer CENSUS of the served model (no GPU needed for this part): walk the model definition and the quantization
  config to list every linear module per rank with (name, N, K, dtype/scheme, weight bytes) so the operator can multiply by the
  measured µs to get the time budget per module class per M. Put it in `deliver/gemm_shape_census.md`.

## B. `deliver/0140-nvfp4-shape-dispatch-v0290.diff` — shape-keyed dispatch behind `VLLM_SM12X_NVFP4_DISPATCH_TABLE`
- Knob value = path to a JSON file: `{"rules":[{"layer":"<regex on module prefix>","m_max":<int or null>,"kernel":"<class name>"}...]}`,
  first matching rule wins; `m_max` compares against the runtime M of the activation (rows) — note M is only known at `apply`
  time, so the patch must build the candidate kernel objects at init (weights processed once per kernel class it may need; if a
  class needs its own weight repack, the patch must keep BOTH repacked copies and document the VRAM cost per module) and pick
  at apply time from a small per-module table keyed by M buckets. If keeping two repacked weights is unacceptable for VRAM
  (27B NVFP4 ≈ 14 GB weights per rank pair), say so and restrict B to kernels that share the same weight layout, and state which do.
- Unset ⇒ byte-identical to today. Invalid JSON / unknown kernel / kernel that `can_implement` rejects for that module ⇒ raise at init.
- Proof line once per process: `NVFP4 dispatch table: <n> rules, <k> modules affected, kernels {…}`; plus a per-module debug line.
- The patch must apply at fuzz 0 on src/ BOTH with and without evidence/0139-nvfp4-marlin-allowlist-v0290.diff applied first
  (0139 also edits `init_nvfp4_linear_kernel` and adds `layer_name`); the simplest way is to reuse 0139's `layer_name` plumbing:
  deliver `0140` as applying ON TOP of 0139 (state that clearly) AND deliver `0140a` as a standalone variant if cheap. Verify both
  in verify-0140.sh (two scratch trees).
- Include the CUDA-graph angle: apply-time selection by M must be graph-safe (M is static per captured batch size, so the choice
  is baked per graph — confirm from `model_runner.py`/cudagraph_utils.py how M is padded and say whether the bucket boundaries
  must align with the graph capture sizes; if yes, document the capture-size list read from the served launcher).
Deliver NOTES23.md with: the census plan (which numbers I fill in), how the JSON table is then derived (a small
`deliver/make_dispatch_table.py` that reads the census JSON and prints the best-kernel-per-(module class, M bucket) table would be
ideal), and the falsification criterion (if no kernel beats the current one by >5% at any served M, B is dead).
