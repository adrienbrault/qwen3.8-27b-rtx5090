# BRIEF22b — 0139b: make the W4A16 kernel class selectable (Marlin | Humming) in the per-layer allowlist

Read the rules in BRIEF22.md's preamble (offline only: verify with patch --dry-run at fuzz 0 on a scratch copy of src/, py_compile,
dependency-free tests; no GPU/model execution; one pass). deliver/0139-nvfp4-marlin-allowlist-v0290.diff is the CURRENT 0139 (it was
amended on the box: the env entry is now `str | None` — the raw regex text — and `init_nvfp4_linear_kernel` calls `parse_allowlist()`
on it, because vLLM's `envs.compile_factors()` hashes every env value and rejects `re.Pattern`). Build on this version.

## Why (evidence-r188.txt, results of 2026-09-05)
The R188 engine A/B on the served TP2 box: routing all 112 MLP projections to MarlinNvFp4LinearKernel (W4A16: bf16 activations, no
NVFP4 activation quantization) moves the model much closer to bf16 — dense corpus PPL gap +0.667% → +0.207%, top-1 92.74% → 94.05%,
agentic PPL gap +2.748% → +1.435% — at c1 parity but c8 −15% and c16 −21% steps/s. The offline GEMM census on the same shapes
(gate_up N=17408 K=5120, down N=5120 K=8704, per rank) has HummingNvFp4LinearKernel (also W4A16, own weight layout) equal to Marlin at
M ≤ 16 and clearly faster from M = 40: down M=80 38.6 vs 48.8 µs, M=160 67.2 vs 81.6; gate_up M=80 65.2 vs 75.4, M=160 122.5 vs 132.8.
The fidelity gain comes from W4A16 itself, so Humming should keep it and give back part of the c8/c16 loss. 0139 hard-codes Marlin.

## Deliverable
`deliver/0139b-nvfp4-a16-allowlist-v0290.diff`: standalone replacement for 0139 against src/ (same files, same env
`VLLM_SM12X_NVFP4_MARLIN_LAYERS` semantics — keep the name for compatibility — plus a new env
`VLLM_SM12X_NVFP4_A16_KERNEL` with values `marlin` (default) | `humming`, raw string, primitives only in envs.py).
- The allowlisted layers instantiate the chosen class (`MarlinNvFp4LinearKernel` or `HummingNvFp4LinearKernel`, both already in
  `a16_kernels` in `vllm/model_executor/kernels/linear/__init__.py`), with the same conflict checks (linear backend auto, not in
  VLLM_DISABLED_KERNELS, no batch-invariant) and `is_supported()` gate; an unsupported choice raises with the reason.
- Proof lines: `NVFP4 A16 allowlist regex: <regex> kernel=<ClassName>` once, and per layer `NVFP4 A16 allowlist: <ClassName> for
  <layer>` (the R188 unit counts these lines; keep the per-layer form).
- Check in src/ what HummingNvFp4LinearKernel needs at weight-processing time (its own repack / scale layout, workspace) and whether
  `CompressedTensorsW4A4Fp4.process_weights_after_loading` in the served tree already handles a16 kernels generically or only Marlin;
  if Humming needs a different weight path, implement it exactly as the upstream a16 path does for the non-allowlist case (grep how
  `use_a16=True` layers are processed). State the SM120 support status of Humming from `is_supported()` in NOTES22b.md.
- Update deliver/verify-0139.sh → verify-0139b.sh (same offline checks; matcher tests + a test that the kernel env parses and
  defaults) and deliver/Dockerfile.marlinlist → Dockerfile.a16list applying 0139b (marker text PRS-0139B-NVFP4-A16-ALLOWLIST-APPLIED).
- NOTES22b.md: what differs from 0139, the Humming weight path, the exact env lines for the box (`VLLM_SM12X_NVFP4_A16_KERNEL=humming`
  with the M-ALL and gate_up regexes from evidence-r188.txt), memory expectations vs Marlin (R188: the M-ALL arm needed the 13.5 GB KV
  pin, 347 MiB free at 13.98 GB), and what stays unverified offline.
