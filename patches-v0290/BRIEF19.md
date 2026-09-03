# BRIEF19 — rc1 NVFP4-KV deep-context decode regression (30K ctx: 29 tok/s on v0.29.0rc1 vs 144 on v0.28.0)

Context: Qwen3.8-27B (hybrid GDN + full-attention, 64 layers, 4 full-attention layers per 8? see the model file) served
by vLLM on 2× RTX 5090 (sm_120), TP2, NVFP4 weights (RedHatAI compressed-tensors) and an NVFP4 KV cache that upstream
gates to SM100 — our patch chain (0101…0135) routes SM12x NVFP4 KV through the FlashInfer FA2 kernels ("FA2 fallback",
`VLLM_SM12X_NVFP4_XQA=0`, decode_backend=flashinfer-native) plus a custom store overlay (0102, csrc). The chain was rebased
from v0.28.0 to v0.29.0rc1 (NOTES16.md). Everything else on rc1 is at parity: fp8-KV decode at 30K = 135 tok/s on both
versions, short-context nvfp4 decode ≈ 225 tok/s on both, needles/fidelity pass. Only nvfp4 KV + long context is slow:

| cell (R167, 2026-09-03, spec ON = DFlash2 ns9 drafter in CUDA graphs) | short c1 (code) | 30K-ctx c1 (prose) | accept/draft @30K |
|---|---|---|---|
| rc1 chain, nvfp4 KV (A)                        | 226.7 tok/s | **28.95** (24.9–33.0) | 0.199 |
| rc1 chain, nvfp4 KV + embed offload (B)        | 223.1       | **26.2**              | 0.172 |
| rc1 chain, fp8 KV (C, XQA decode)              | 249.8       | 135.0                 | 0.130 |
| v0.28 candidate, nvfp4 KV (R166, same probe)   | ≈225        | **144–157**           | ≈0.2  |

Per step at 30K: ≈85 ms on rc1-nvfp4 vs ≈15 ms on v0.28-nvfp4 (decode_ss.py --ctx 30000, c1, 512 tokens). Short context
is normal, so the per-token cost that scales with context length is what regressed: the paged-KV decode attention over
the NVFP4 cache (FA2 split-KV decode path), or something around it that is O(context) (page-table / plan work per step,
a per-step de-quant or layout conversion of the whole cache, a graph-capture miss that runs the long-context shape eagerly
with per-page launches, the drafter re-reading the whole KV non-causally, …).

Library delta between the two images (torch identical):
- v0.28 image: vllm 0.28.0, torch 2.13.0+cu130, flashinfer-python/cubin/jit-cache 0.6.16.post3
- rc1 image:   vllm 0.29.0rc1, torch 2.13.0+cu130, flashinfer-python/cubin/jit-cache 0.6.18
FlashInfer 0.6.17/0.6.18 touched the FA2 kernels: #3890 "skip LogitsTransform on lanes beyond the split-KV chunk boundary
in FA2 kernels", #4272 "SM120 FP8 FMHAv2 self-attention", #4389 LSE reduction, plus GDN/KDA decode kernel changes
(#4374 cp host launch overhead sm120, #4128 MTP decode kernels across cache modes, #4444 pretranspose reuse, #4219).

## Inputs (local, read-only)
- `TREES/v028p/`  — v0.28.0 python tree with the v0280 chain applied (0101–0113 + 0116–0119 + 0129 + 0131) = the fast image
- `TREES/v029real/` — v0.29.0rc1 python tree with the rebased chain applied through 0134 = the slow image (0135 is embed-only)
- `TREES/v028src/`, `TREES/v029src/` — pristine 0.28.0 / rc1 trees
- `TREES/fi/src-0.6.16.post3/flashinfer/`, `TREES/fi/src-0.6.18/flashinfer/` — the two FlashInfer python packages (the
  FA2 decode/prefill wrappers, plan(), split-KV scheduling, JIT specs live here; the CUDA sources are not included —
  reason from the wrappers and the release notes; CUDA-side hypotheses are fine but label them as such)
- `PATCHES/patches-v0280/` and `PATCHES/patches-v0290/` — the two chains (0101 is the routing patch: compare the v0280 and
  v0290 versions of 0101/0103/0105/0109/0129/0131 — NOTES16.md lists what codex rebased and why)
- `PATCHES/patches-v0290/NOTES16.md`, `NOTES17-rc1-regressions.md`, `NOTES18.md` — rebase notes and rc1 regressions found so far
  (regression #3: draft kv_cache_dtype in the DFlash spec JSON breaks rc1 boot — it is dropped on rc1, so the DRAFTER'S
  attention on rc1 may run a different KV dtype/path than on v0.28; check what the drafter does with the target's nvfp4 KV
  on each version — 0105/0109 non-causal nvfp4 FA2 for the drafter)
- rc1 boot facts (nvfp4 route): `decode_backend=flashinfer-native, kv_cache_dtype=nvfp4, arch=sm120`;
  `VLLM_SM12X_NVFP4_XQA=0: SM12x NVFP4 decode uses the FlashInfer FA2 fallback`; `NVFP4KV-SM120: linear-V-scale store
  overlay ACTIVE`; `0129 UNIFORM_BATCH graph support`; `Graph capturing finished … 0.53 GiB / 0.75 GiB`.

## What to deliver (into OUT/)
1. `NOTES19.md`: ranked hypotheses with the code evidence for each (file:line in the two trees / two FlashInfer packages),
   and for each hypothesis the ONE cheapest measurement that confirms or kills it on the box. The box will run these cells
   anyway (r168): FlashInfer-swap (v0.28 tree + FI 0.6.18, rc1 tree + FI 0.6.16.post3), spec OFF on both versions, masked
   XQA on rc1, fp8 control, and a torch-profiler kernel table of ~32 decode steps at 30K per cell. Say what each outcome
   would mean and what you would look for in the kernel table (kernel names, counts per step, graph-launch counts).
   Specifically compare between v028p and v029real: (a) the FA2 decode call path for nvfp4 KV in
   vllm/v1/attention/backends/flashinfer.py (wrapper construction, plan() args, use_tensor_cores, split-KV / paged_kv
   layout, workspace sizes — 0131 shrinks the int workspace: does the plan re-run per step on rc1?), (b) cudagraph dispatch
   for decode with nvfp4 on rc1 (vllm/v1/cudagraph_dispatcher.py, compilation config: did rc1 move decode of this backend
   to PIECEWISE/eager for long context? any `uniform_decode` / `max_query_len` batch-descriptor change), (c) the GDN
   decode path (0108/0133, fused_recurrent) for anything O(context), (d) the DFlash2 speculator (0113/0129 speculator.py,
   qwen3_dflash.py) — what the drafter reads per step and whether rc1 dropped the draft KV dtype, (e) FlashInfer wrapper
   API differences 0.6.16.post3 → 0.6.18 that vLLM's FA2 nvfp4 call would hit (new default kwargs, changed
   `plan` signature, `kv_layout`, `use_fp16_qk_reduction`, split-KV chunk heuristics, `q_data_type`/`kv_data_type` handling).
2. If a code-level cause is found: `0136-<name>-v0290.diff` against `TREES/v029real` (tree-relative `a/vllm/...` paths,
   applies with `patch -p1 --fuzz=0`), minimal, with an env knob only if the change is a heuristic; plus `verify-0136.sh`
   (dry-run apply + py_compile, no GPU).
3. If the cause is inside FlashInfer 0.6.18's kernels: say so, name the PR, and give the vLLM-side workaround candidates
   (pin the wrapper kwargs, force a different tile/split-KV setting, or pin FI 0.6.16.post3 on the rc chain — is that
   safe with vLLM 0.29's flashinfer usage? list the API calls 0.29 makes that 0.6.16.post3 lacks, if any).
Rules: do not touch the box; do not edit the input trees in place (copy to OUT/work if needed); no speculation dressed as
finding — mark each claim as VERIFIED (code read) or HYPOTHESIS.
