# BRIEF24b — follow-up to BRIEF24: tune the kernel the served ns9 step ACTUALLY runs (patch 0145)
Your NOTES24 finding is confirmed by the operator's profiles: under ns9 the served step never calls the packed kernel. The R183
profiles (evidence/r183/prof-BASE/{c1,c8,c16}/*.summary.txt) show `fused_sigmoid_gating_delta_rule_update_kernel` at 2544 calls
per 53 profiled steps = 48 calls per step (one per GDN layer), with average per-call time 10.0–10.4 µs at c1 (0.42–0.44 ms/step)
and 53.6–56.3 µs at c16 (2.27–2.39 ms/step, ≈7% of the 33.5 ms step); at c16 `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`
also appears (384 calls, 0.58 ms/step: prefill chunks mixed into decode steps). Keep 0141 as delivered (spec-off only) and now
deliver **0145** for the spec path.

Kernel: `src/vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py` — `fused_sigmoid_gating_delta_rule_update_kernel`
(heuristics IS_VARLEN / IS_SPEC_DECODING, `do_not_specialize=["N","T"]`, launch `BK=next_pow2(K)=128, BV=min(next_pow2(V),32)=32,
num_stages=3, num_warps=4, grid=(NK=1, NV=4, N*HV)`; row loop `for i_t in range(0, T)` with the num_accepted_tokens-driven initial
state slot selection at line ~106 and per-row bf16 checkpoint stores for rollback). Called from
`src/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` line ~1540 with cu_seqlens = spec_query_start_loc,
ssm_state_indices = spec_state_indices_tensor (per request: 1+9 slots), num_accepted_tokens, use_qk_l2norm_in_kernel=True,
inplace_final_state=True; per rank H=8, HV=24, K=V=128, per-request T=10 rows (1 bonus + 9 draft), N = number of spec requests
(1/8/16 at c1/c8/c16). So at c1 the grid is 1×4×24 = 96 CTAs of 4 warps, each walking 10 rows sequentially with a 128×32 fp32
state tile in registers: ≈10 µs per call means the kernel is latency-bound (10 sequential row iterations, each a load of q/k/v/
gates + a 128×32 rank-1 update + a bf16 checkpoint store of 128×32), with only 96 CTAs on 170 SMs.

Deliver `deliver/0145-gdn-spec-update-tuning-v0290.diff` (+ Dockerfile.gdn-spec-tuning, verify-0145.sh, tests, NOTES24b.md,
`deliver/gdn_spec_microbench.py`), same contract as COMMON.md, new knob `VLLM_SM12X_GDN_SPEC_CFG` with the grammar you defined
for 0141 (`bv=,bk=,warps=,stages=,split=[,variant=]`), unset ⇒ byte-identical, proof line once
`GDN spec update: variant=<v> bv= bk= warps= stages= (HV=24 K=128 V=128 T=10)`:
- Variants worth implementing (choose the 2–3 with the best expected payoff at c1 AND no c16 regression):
  (a) smaller BV (16/8) to raise CTA count at c1 (192/384 CTAs), fewer warps per CTA (1–2) so the 128×BV tile keeps the same
  per-thread register footprint; (b) the checkpoint store: today each row stores the bf16 state for rollback — check whether the
  store of row t can be issued asynchronously / coalesced (it is already fire-and-forget in Triton, but the layout and the
  `tl.store` mask may serialize; consider storing every row's checkpoint from a double-buffered register tile), and whether the
  L2-norm of q/k per row can be hoisted (they are per-row inputs, so only if the caller can precompute — say so; do not change the
  caller); (c) split-V across CTAs per head without atomics (the delta-rule update is rank-1 per row: state[k, v] += beta * (v_row
  - state·k)[v] * k[k] — the (state·k) reduction is over K, so a V-split is exact; a K-split is NOT, do not do it);
  (d) `num_stages`/software pipelining of the per-row loads (prefetch row t+1's q/k/v/gates while updating row t).
- Keep the recurrence math, reduction order over K, the sigmoid gating, the L2-norm, and every bf16 rounding point identical;
  say precisely which variants are bitwise-identical by construction (CTA reorder / V-split) and which may differ (warp count
  changing the reduction tree). The operator gates finalists with a bf16 decode ruler.
- Microbench (runs in the container, 1 GPU, Triton compiles at first call): random inputs at the exact per-rank shapes,
  N ∈ {1,8,16} × T=10 with realistic num_accepted_tokens (random 1..10 per request) and 1+9 state slots per request,
  baseline config vs the sweep, correctness vs baseline outputs AND final/checkpoint states AND a pure-torch fp32 reference of
  the recurrence, µs per call; and a CUDA-graph replay timing (this kernel runs inside the piecewise graph).
- NOTES24b: expected best case at c1/c8/c16 (ms per step, from the per-call numbers above), the CTA/occupancy math for each
  variant, and the falsification criterion (no config ≥10% faster than baseline at c1 with no c16 loss ⇒ dead).
Verify ONLY with scratch apply at fuzz 0 + py_compile + dependency-free tests. One pass. Write CODEX-final.md (15 lines).
