# COMMON.md — shared context for the R190 codex briefs (read first, then your BRIEF.md)

You are writing a patch/script deliverable for a vLLM 0.29.0rc2 serving stack on a dual RTX 5090 (SM120, 32 GB each, PCIe, no NVLink)
box, AMD 9800X3D, 64 GB RAM. You have NO GPU, NO network, NO access to the serving box. You work offline on a source dump and
evidence files in this directory. One pass, medium effort. Do not spend time trying to run anything that needs torch/vllm/GPU.

## The served configuration (what your patch will run under)
- Target: RedHatAI/Qwen3.8-27B-NVFP4 (config: evidence/target-config.json). Qwen3.5-family hybrid: 64 layers, layer_types
  3× linear_attention (Gated DeltaNet, "GDN") + 1× full_attention repeating (48 GDN layers, 16 attention layers).
  hidden 5120, intermediate 17408, attention 24 q heads / 4 kv heads / head_dim 256; GDN linear_num_key_heads 16,
  linear_num_value_heads 48, key/value head dim 128, conv kernel 4. vocab 248320, untied lm_head.
- Quantization (compressed-tensors): group_1 = `mlp.(gate|up|down)_proj` of layers 0..55 → NVFP4 W4A4 (tensor_group 16);
  group_0 = attention q/k/v/o, GDN in_proj_qkv / in_proj_z / out_proj, lm_head, and MLP of layers 56..63 → FP8 W8A8
  (channel weights, per-token activations). GDN in_proj_a / in_proj_b / norm are unquantized.
- Tensor parallel 2 (one process per GPU), so per-rank GEMM shapes are: gate_up_proj N=17408 (2×17408/2), K=5120;
  down_proj N=5120, K=8704; attention qkv N=(24×256+2×4×256)/2, K=5120; o_proj N=5120, K=3072; GDN in_proj_qkv
  N=(16×128 + 16×128 + 48×128)/2 = 5120, K=5120; in_proj_z N=3072, K=5120; out_proj N=5120, K=3072; lm_head N=124160, K=5120.
  Derive anything else from evidence/target-config.json and src/vllm/model_executor/models/qwen3_5.py (+ qwen3_next.py).
- Speculative decoding: DFlash (block drafter, `src/vllm/v1/worker/gpu/spec_decode/dflash/`), drafter config
  evidence/drafter-config.json (5 layers, hidden 5120, block_size 8, target_layer_ids [5,19,33,47,61], W4A16), num_speculative_tokens
  9 ("ns9"), draft TP2, drafter runs inside CUDA graphs. Per target decode step each request contributes 1+9 = 10 rows, so the
  target GEMM M at concurrency c1/c8/c16 is 10/80/160 (before acceptance; exact rows vary with padding). Compilation mode
  FULL_AND_PIECEWISE with CUDA graphs. KV: attention KV nvfp4, GDN SSM state bf16, LMCache-style tiers (not relevant here).
- Launch flags: evidence/launch-daily.sh (the served launcher). The image is `vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc`,
  i.e. vLLM 0.29.0rc2 + local patches 0101..0138 already applied (src/ IS that patched tree, dumped from the image) + FlashInfer 0.6.16.
  Patch 0139 (evidence/0139-nvfp4-marlin-allowlist-v0290.diff) is a SEPARATE layer that is NOT in src/; it is under test.
- Relevant existing local patches (already in src/): 0129 DFlash drafter CUDA graphs; 0133 packed GDN decode BV16
  (evidence/0133-...diff; knob VLLM_SM12X_GDN_PACKED_BV); 0138 pcie_ipc all-reduce (evidence/0138-...diff + evidence/NOTES21.md;
  package src/pcie_ipc_ar21; knob VLLM_SM12X_PCIE_IPC_AR=1, served ON).

## Measured time budget (evidence/R186-ANALYSIS.md §2, profiles in evidence/r183/prof-BASE/{c1,c8,c16})
Decode step after the pcie_ipc promotion, ms per target step (c1 / c8 / c16):
- step ≈ 18.4 / 22.3 / 33.5; NVFP4+FP8 GEMM ≈ 40% / 37% / 32% of it; "gaps" (no kernel running) 4.09 / 2.89 / 2.75 (23%/13%/8%);
  all-reduce residual after pcie_ipc 10–12% / 21% / 27%; GDN decode kernel 0.68 → 3.22 ms c1→c16;
  vocab all-gathers ≈3.1 per step, 1.01 / 1.99 ms at c8 / c16; drafter GEMMs ≈0.9 / 1.2 / 2.0 ms.
Read §3 of R186-ANALYSIS.md for the proposal your brief implements; the section number is in your BRIEF.md.

## Source layout (this directory)
- `src/vllm/` — the served vLLM tree (site-packages `vllm/`), patches applied, no .so/.pyc. Treat as READ-ONLY reference.
- `src/pcie_ipc_ar21/` — the 0138 all-reduce package (csrc/bindings.cu, include/flashinfer/comm/pcie_ipc_all_reduce.cuh vendored
  unmodified from FlashInfer main df8b5c1, workspace.py, fixed_config.py, NOTICE).
- `evidence/` — analysis, launcher, prior patches and notes, probes (decode_ss.py, prof_*.py), R183 profile summaries and CPU
  tables, R184 all-reduce bench outputs, R185 sheets, target/drafter configs.

## Deliverable contract (put everything under `deliver/`)
1. `deliver/<NNNN>-<slug>-v0290.diff` — unified diff, paths `a/vllm/...` / `b/vllm/...` (-p1 from the site-packages root; for the
   pcie_ipc package `a/pcie_ipc_ar21/...`). Must apply with `patch -p1 --fuzz=0 --batch --forward` on the pcieipc tree (src/).
   New behaviour ALWAYS behind an env knob named `VLLM_SM12X_<THING>` declared in `vllm/envs.py` (see 0133 for the pattern);
   knob unset ⇒ byte-identical behaviour to today (same kernels, same tensors, same numerics). Fail loudly (raise) on an invalid
   knob value or an unsupported combination; never silently fall back. Log ONE proof line per process at the point the knob takes
   effect (`logger.info_once(...)`), so the operator can grep it in `docker logs` (2 ranks ⇒ every line appears twice).
2. `deliver/Dockerfile.<slug>` — `ARG BASE` (default the pcieipc image name above), copies the diff, asserts
   `python -c "import vllm; assert vllm.__version__.startswith('0.29')"`, applies with `patch -d <site-packages> -p1 --fuzz=0 --batch
   --forward`, `python -m py_compile` on every touched .py, echoes `PRS-<NNNN>-<SLUG>-APPLIED` into a marker file
   `/opt/prs-markers/<NNNN>`, and sets NO enabling ENV (the launcher opts in). No network in the build (`RUN --network=none`).
3. `deliver/verify-<NNNN>.sh` — offline: copies src/vllm to a scratch dir `.work/<NNNN>/vllm`, runs the dry-run AND a real apply
   at fuzz 0 (check `$?` of the real apply, do not grep output), py_compiles touched files, runs any dependency-free unit tests you
   wrote (pure-Python: regex parsing, table lookup, shape math; NO torch/vllm imports). Exit non-zero on any failure. RUN IT.
4. `deliver/NOTES<NN>.md` — what the patch does, exactly which functions changed, the knob grammar, the proof line to grep, the
   measurement plan (which of the operator's probes to run, expected direction, what result would falsify the idea), risks,
   what you could NOT verify offline, and provenance (any technique/code borrowed: project, commit/PR, license) for THIRD_PARTY.md.
5. Any scripts the operator must run on the box go in `deliver/` too, with a one-line usage header; they run INSIDE the served
   container (`docker exec`), which has torch, vllm, triton, flashinfer, CUDA 13; assume 2 GPUs are free during a microbench.

## Rules
- Do NOT edit `src/` in place; work on a copy (`.work/`). Do NOT touch the evidence files.
- Verify ONLY with scratch-tree apply at fuzz 0 + py_compile + dependency-free tests. Do not import torch or vllm. No GPU.
- Keep the patch minimal and reviewable (< ~400 diff lines unless the brief says otherwise). No renames of existing knobs.
- If the brief asks for something the source shows is impossible or already done, say so in NOTES and deliver the rest.
- Write `CODEX-final.md` at the end (the -o file): 15 lines max — files delivered, verify result (paste the last lines),
  the proof line, and the one number the operator should look at first.
