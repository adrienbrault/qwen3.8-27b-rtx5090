# BRIEF29 — sync-free attention-metadata build on the served decode path (patch 0146)
Read COMMON.md first. This brief does NOT come from R186-ANALYSIS.md §3; it comes from the R190d runtime census (your own 0142c
instrumentation, evidence/r190d/sync-census-table.txt, 67 c1 decode steps, identical every step):

    target        5 × api:to          vllm/v1/attention/backend.py:473     (blocking seq_lens.to("cpu"))
    target        1 × api:replay      vllm/v1/worker/gpu/cudagraph_utils.py:452
    sample+draft  1 × api:to          vllm/v1/attention/backend.py:473
    sample+draft  1 × api:replay      vllm/v1/worker/gpu/cudagraph_utils.py:452
    output        1 × api:synchronize vllm/v1/worker/gpu/async_utils.py:168 (copy_event.synchronize — the sampled-token readback, keep)

So 6 of the 9 host-sync points per step are the SAME deprecated property, `CommonAttentionMetadata.seq_lens_cpu`
(src/vllm/v1/attention/backend.py ~line 465–475: `self._seq_lens_cpu = self.seq_lens.to("cpu")`, a synchronous D2H copy that
also waits for the stream to drain). Upstream marks it deprecated: "Prefer using device seq_lens directly to avoid implicit H<>D
sync which breaks full async scheduling". There is NO acceptance read before the draft (0142b removed 0 syncs; that route is closed).
The R183 profile has the c1 step at 18.4 ms with a 4.09 ms no-kernel gap (23%); the host cannot run ahead of the GPU while it
blocks on the stream six times per step. This brief targets that gap.

Callers of the property in src/ (find every one; these are what grep shows):
- src/vllm/v1/attention/backends/flashinfer.py ~1746 (`seq_lens_cpu = common_attn_metadata.seq_lens_cpu`) — the served attention
  backend (patch 0103 XQA-NVFP4 decode + FlashInfer prefill; read which path consumes the host values: plan() for prefill
  wrappers, paged-KV indptr/last_page_len construction, the decode/prefill split, the trtllm/XQA decision?).
- src/vllm/v1/attention/backends/utils.py ~481 (`seq_lens_np = common_attn_metadata.seq_lens_cpu.numpy()`) — a shared helper;
  find who calls it on the served path (GDN/mamba metadata builder? split_decodes_and_prefills? reorder?).
- src/vllm/v1/attention/backend.py ~490 (`num_computed_tokens_cpu` property = seq_lens_cpu - query lens).
- The DFlash draft build (src/vllm/v1/worker/gpu/spec_decode/dflash/) accounts for the 6th copy: find where the drafter's
  CommonAttentionMetadata is built and which consumer reads seq_lens_cpu there.
Also explain the multiplicity: why 5 builds in the target phase (KV-cache groups of the hybrid: attention group(s), GDN group(s),
the drafter's target-layer taps?) — see src/vllm/v1/worker/gpu/model_states/mamba_hybrid.py and attn_utils.py.

The constraint you must respect: under speculative decoding the host only holds `seq_lens_cpu_upper_bound` (input_batch.py;
num_scheduled_tokens + context, an UPPER bound because accepted lengths live on the device). Any consumer that needs the EXACT
per-request length (FlashInfer plan() host arrays, page counts) cannot use the bound unless the kernel masks by a device-side
length. So the deliverable has two tiers; deliver tier 1, and tier 2 only where the source shows it is safe:

Tier 1 — `deliver/0146-syncfree-seqlens-v0290.diff`, knob `VLLM_SM12X_SEQLENS_ONE_COPY=1`: exactly ONE D2H copy of seq_lens per
  decode step, issued as early as the device value is final (right after the runner updates seq_lens for the step, before any
  metadata build), non-blocking into a persistent pinned host buffer with a CUDA event recorded after it; every consumer that
  today calls `seq_lens_cpu` (all six builds, target and draft) gets the SAME host tensor after ONE `event.synchronize()` at the
  first use. Also cache `num_computed_tokens_cpu` from it. Sync count per step must go 9 → 4 (1 D2H event wait + 2 replays +
  1 output readback). Proof line once per process: `SM12X seq_lens one-copy active: <n> seq_lens_cpu consumers share one D2H per step`.
  Unset ⇒ byte-identical. Correctness: the host values are the same numbers as today (same device tensor, copied once), so the
  bf16 decode-fidelity ruler must be bitwise 20/20 against a same-artifact control; state in NOTES why the copy is complete
  before every consumer (event ordering vs the stream that writes seq_lens; CUDA-graph replay boundaries; preemption; a request
  finishing mid-block; the drafter graph's static buffers).
Tier 2 — `deliver/0146b-devicelen-consumers-v0290.diff`, knob `VLLM_SM12X_SEQLENS_DEVICE=1` (requires 0146): for every consumer on
  the SERVED decode path that can take the device tensor or the CPU upper bound without changing results (XQA/trtllm decode
  kernels take seq_lens on device; GDN decode reads device state; the prefill/decode split can use the upper bound if the
  per-request query length is what it actually needs), drop the host read entirely so the decode-only step has ZERO seq_lens
  syncs and the host can run ahead. Where exact host values are required (prefill plan()), keep the tier-1 copy on that path
  only. If tier 2 is not safely possible on this source, say exactly why in NOTES29.md and deliver tier 1 alone.

Measurement plan for NOTES29.md (the operator runs it): (1) 0142c census with each knob on: expected api:to backend.py:473 = 1
(tier 1) / 0 in decode-only steps (tier 2); (2) decode ruler ctx 0 / 30K vs a same-compile-artifact control: 20/20, median 0;
(3) probes/decode_ss.py code c1 (runs 3), c8, c16 steps/s; the falsifier is a c1 step that does not move with the syncs gone
(then the 4.09 ms gap is launch-bound, not sync-bound, and the next brief is CUDA-graph coverage, not syncs). Do not claim a
speedup; predict the direction and the ceiling (the c1 gap is 4.09 ms of 18.4).

Files to read first: src/vllm/v1/attention/backend.py (CommonAttentionMetadata), src/vllm/v1/worker/gpu/model_runner.py (where
seq_lens is computed per step; grep seq_lens, num_computed_tokens, non_blocking, copy_stream, async), attn_utils.py,
model_states/mamba_hybrid.py, input_batch.py, spec_decode/dflash/speculator.py, attention/backends/{flashinfer.py,utils.py},
and evidence/r190d/0142c-dflash-sync-census-v0290.diff + NOTES25.md for the census semantics. Patch budget: < 400 diff lines total.
