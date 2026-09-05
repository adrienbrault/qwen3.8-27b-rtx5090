# BRIEF26 — vocabulary all-gather audit and tagging under DFlash + TP2 (patch 0143)
Read COMMON.md first. Implements R186-ANALYSIS.md §3 "vocab all-gathers" (#3): ≈3.1 all-gathers per decode step of vocab-sized
tensors, 1.01 ms at c8 and 1.99 ms at c16 (4% at c16). lm_head is FP8, N=124160 per rank (vocab 248320 sharded), M = rows
(10/80/160 × sampling positions). Sources: `src/vllm/model_executor/layers/logits_processor.py` (`use_all_gather` line ~91,
`tensor_model_parallel_all_gather(logits)` line ~130, and two more at ~246/~288 for a sharded top-k/values+ids path already
present — read what selects it), `src/vllm/v1/worker/gpu/sample/batch_shard.py` + `sampler.py` + `logprob.py`, the engine flag
`--enable-batch-sharded-sampling` (grep `batch_sharded_sampling` in `src/vllm/config/parallel.py`, `engine/arg_utils.py`,
`v1/worker/gpu/model_runner.py`), and the DFlash speculator (`spec_decode/dflash/speculator.py`) which also computes logits
for draft verification and for the drafter's own sampling (the drafter's lm_head: is it the target's, sharded the same way?).
The operator is measuring `--enable-batch-sharded-sampling` tonight on the served stack (R187 arm BSS); do not implement a
distributed top-k yet — its need depends on that result.

Deliver:
1. `deliver/NOTES26.md` §1 AUDIT: for one decode step at c8 with ns9, enumerate every vocab-sized collective (all-gather,
   all-reduce, gather, broadcast) with: call site (file:line), tensor shape and dtype, bytes at c8/c16, which phase (target
   logits for verification, bonus token, drafter logits, logprobs), and whether it is inside a CUDA graph. Reconcile with the
   measured ≈3.1 per step (why not an integer? probably drafter graph vs eager difference or logprobs on some requests).
   §2: what `--enable-batch-sharded-sampling` changes exactly in this code (which collectives disappear, which appear — e.g.
   a smaller gather of sampled ids — and whether it composes with DFlash's verification path or is bypassed for spec decode;
   if bypassed, the R187 BSS arm will measure nothing and that must be said plainly).
   §3: DESIGN (no code) for a sharded top-k/top-p sampler under spec decode: per-rank top-k on the local 124160 columns →
   all-gather of (k values, k ids) per row → merge → sample; bytes per step vs today; how greedy (temperature 0) and the
   rejection-sampling verification (`accept` logic) would consume it; numerics: identical for greedy, and for sampling whether
   the gumbel path (`sample/gumbel.py`) can be made rank-consistent. Estimate the saving at c8/c16 (ms/step) from the byte counts
   and the 25.56 GB/s PCIe payload rate (evidence/r184 has the all-gather/all-reduce microbench numbers; use them).
2. `deliver/0143-vocab-collective-tags-v0290.diff` — knob `VLLM_SM12X_COLLECTIVE_TAGS=1`: wrap the vocab collectives you found
   with `torch.cuda.nvtx.range("vocab_ag:<phase>:<rows>x<cols>")` (and, for the eager ones, an optional per-step counter logged
   once after 50 steps: `vocab collectives per step: {phase: count, bytes}`), so the operator's profiler split
   (evidence/prof_decode_split.py) can attribute them by phase. Unset ⇒ byte-identical. Tiny patch.
