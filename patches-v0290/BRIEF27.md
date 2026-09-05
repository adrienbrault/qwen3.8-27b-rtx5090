# BRIEF27 — pcie_ipc all-reduce output fused with residual add + RMSNorm (patch 0144)
Read COMMON.md first. Implements R186-ANALYSIS.md §3 "pcie_ipc + residual/RMSNorm fusion" (#10): after each layer's o_proj /
out_proj and down_proj all-reduce (TP2, PCIe, no NVLink) the model does `hidden = allreduce(x); hidden, residual = rmsnorm(hidden +
residual)` as separate kernels: the reduced tensor is written by the all-reduce kernel, re-read by the residual add, written,
re-read by RMSNorm. At M=10 rows × 5120 these are tiny tensors, so the cost is launch count + memory round trips (~2 extra
kernels × 128 all-reduce sites per step ≈ 256 launches; estimated 0.5–1.2 ms per step recoverable). Since 0138 the all-reduce
is FlashInfer's pcie_ipc kernel (src/pcie_ipc_ar21: csrc/bindings.cu = PyTorch binding; include/flashinfer/comm/
pcie_ipc_all_reduce.cuh = vendored UNMODIFIED from FlashInfer main df8b5c1, Apache-2.0; workspace.py = ownership/prep glue;
fixed_config.py = launch triples). vLLM already has a "fused all-reduce + RMSNorm" concept for the custom all-reduce and for
FlashInfer's trtllm fused kernels (grep `fused_allreduce`, `allreduce_rms`, `flashinfer_comm`, `FusedAllReduce` under
`src/vllm/compilation/` and `src/vllm/distributed/device_communicators/`), gated by compile passes (`--compilation-config`
`pass_config.enable_fi_allreduce_fusion` or similar) — read what exists, what it requires (world size, dtype, max tokens), and
whether the pcie_ipc path (added in `cuda_communicator.py` by 0138 — see evidence/0138 diff) is reachable by those passes.

Deliver `deliver/0144-pcie-ipc-fused-norm-v0290.diff` + `deliver/pcie_ipc_fused_norm.cu` (new kernel file, added to
`pcie_ipc_ar21/csrc/` — do NOT edit the vendored .cuh; include it and write a new kernel that reuses its IPC buffer/flag
protocol for the final reduce-and-write stage, so the reduced value is consumed in-register by residual add + RMSNorm and
written once as (hidden_normed bf16, residual bf16)):
- Knob `VLLM_SM12X_PCIE_IPC_AR_FUSED_NORM=1` (requires PCIE_IPC_AR=1, else raise). Plumb it the way vLLM's existing fused
  all-reduce+norm is plumbed (the compile pass that pattern-matches `all_reduce → add → rms_norm`), or, if that pass cannot target a
  custom op cheaply, via the model's decoder-layer forward for qwen3_5 (`src/vllm/model_executor/models/qwen3_5.py`, where
  `post_attention_layernorm`/`input_layernorm` consume `(hidden, residual)`) behind the knob — state which route you took and why.
- RMSNorm here is Gemma-style (`GemmaRMSNorm as Qwen3_5RMSNorm`: x * (1 + w), fp32 accumulate) — match `layernorm.py` exactly
  so outputs are bitwise-identical to the unfused path for the same reduction order; document any rounding-point difference.
- CUDA-graph safe (static workspace, no host sync); TP2 only; M up to the pcie_ipc max_numel in fixed_config.py (fall back to
  the unfused path above it, log once).
- Build recipe: how the .cu joins the existing `pcie_ipc_ar21/build.py` (read it; the image builds the extension at Dockerfile
  time — the Dockerfile.pcieipc recipe is in evidence/NOTES21.md); you cannot run nvcc — say so; give the compile command.
- `deliver/pcie_fused_norm_check.py` (runs in container, 2 GPUs, torchrun-style 2-process): compares fused vs unfused outputs
  for M ∈ {1,10,80,160,2048} at hidden 5120, prints max-abs-diff and µs per call for both.
NOTES27.md: provenance (FlashInfer commit + which functions/protocol reused ⇒ derivative work, note for THIRD_PARTY.md),
expected saving per step at c1/c8/c16 by launch count and bytes, falsification criterion.
