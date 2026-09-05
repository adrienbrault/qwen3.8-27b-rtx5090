# BRIEF24 — GDN fused-recurrent decode kernel tuning for 48 value heads / bf16 state (patch 0141)
Read COMMON.md first. Implements R186-ANALYSIS.md §3 proposal "GDN tuning, 48 value heads" (#4). The GDN decode kernel goes from
0.68 ms (c1) to 3.22 ms (c16) per step, over 48 GDN layers; at c16 it is ≈10% of the step. Shapes per rank at TP2: the 48 value
heads and 16 key heads split across 2 ranks ⇒ HV=24, H=8 per rank, K=V=128, rows = 10/80/160 (c1/c8/c16; spec tokens make each
request 10 rows, and the packed decode path processes them as a sequence per request with state carried in registers/smem).
SSM state is bf16 in memory (24 heads × 128 × 128 × 2 B = 786 KB per request per layer per rank), recurrence in fp32.

Files: `src/vllm/third_party/flash_linear_attention/ops/fused_recurrent.py` — the packed decode kernel
`fused_recurrent_gated_delta_rule_packed_decode_kernel` (line ~265) and its launcher `fused_recurrent_gated_delta_rule_packed_decode`
(line ~383) with `_get_packed_decode_launch_config` (line ~353) added by patch 0133 (evidence/0133-...diff; knob
`VLLM_SM12X_GDN_PACKED_BV` accepting unset/16/32). Callers: `src/vllm/model_executor/layers/mamba/` (grep for
`fused_recurrent_gated_delta_rule_packed_decode`) and `layers/fla/ops/gdn_replayssm_spec_decode.py`; the non-packed path is the
original `fused_recurrent_gated_delta_rule_fwd_kernel` in the same file. Study how the kernel tiles (BV, BK, num_warps, stages),
how it loads/stores the 128×128 state per head (bf16 → fp32 → bf16 per step? or once per request across the 10 spec rows?), how
it handles the per-row gating and the beta/delta update, and where the memory traffic is (state read+write dominates at c1: 48
layers × 786 KB × 2 per step ≈ 75 MB ⇒ 0.68 ms is ≈110 GB/s, far below the 1.8 TB/s of a 5090 — so the kernel is latency/occupancy
bound, not bandwidth bound; at c16 the 3.22 ms is ≈ 375 GB/s, still far from the roofline).

Deliver `deliver/0141-gdn-decode-tuning-v0290.diff`:
- EXTEND `VLLM_SM12X_GDN_PACKED_BV` (do not rename): keep unset/16/32 semantics, add `VLLM_SM12X_GDN_DECODE_CFG` = a small grammar
  `bv=<n>,bk=<n>,warps=<n>,stages=<n>,split=<n>[,variant=<name>]` that overrides the launch config, and implement 2–3 kernel
  VARIANTS selected by `variant=`: e.g. (a) state kept in registers across all spec rows of a request with a single bf16
  load/store per request (if not already so), (b) head-split across CTAs (grid over heads × V-tiles, more CTAs at c1 where only
  24 heads × few tiles fill 170 SMs poorly), (c) 2-row or 4-row unrolled recurrence for the 10-row spec chain. Keep fp32
  recurrence math and identical rounding points so numerics stay bitwise-identical where the algorithm is the same; where a variant
  changes reduction order, say so in NOTES (the operator gates with a bf16 decode-fidelity ruler).
- Unset ⇒ byte-identical; invalid grammar or a variant that does not support the shape ⇒ raise. Proof line once:
  `GDN packed decode: variant=<v> bv=<> bk=<> warps=<> stages=<> (HV=24 K=128 V=128)`.
- `deliver/gdn_decode_microbench.py` runs inside the container: builds random inputs at the exact per-rank shapes for rows
  10/80/160 (and 1/8/16 = spec off), runs the baseline launch config and every variant/config in a small sweep grid, checks
  max-abs-diff vs baseline outputs AND vs a pure-torch fp32 reference of the recurrence, and prints µs + GB/s per config.
  Triton compiles at first call — you cannot run it; verification is py_compile + your own careful reading. Say so.
- Explain in NOTES24.md why the current config is slow at c1 (occupancy math: CTAs launched vs 170 SMs) and what the sweep should
  find; give the expected best-case ms at c1/c8/c16 and the falsification criterion.
