# BRIEF25 — graph-boundary host round trips in the DFlash decode step (patch 0142)
Read COMMON.md first. Implements R186-ANALYSIS.md §3 "graph-boundary metadata/sampling round trips" (#5). At c1 the profiled
step has 4.09 ms with NO kernel running (23% of 18.4 ms); at c8 2.89, c16 2.75 ms. evidence/r183/prof-BASE/c1/*.cpu.txt and
*.summary.txt list the CPU-side ops and the cudaStreamSynchronize / cudaMemcpy DtoH counts; evidence/prof_decode_split.py shows
how the split was derived. The decode step is: target forward (piecewise CUDA graph, then attention/GDN, sampling) → accept
draft tokens → drafter forward (full CUDA graph since 0129) → next step. Suspects for the gaps: (1) device→host copies of
accepted-token counts / sampled ids / selector indices between the target graph, the sampler, and the drafter graph;
(2) per-step Python metadata construction (slot mapping, positions, query start locs) done on the host then H2D-copied;
(3) the ~3 vocab all-gathers per step (separate brief, ignore); (4) the pcie_ipc all-reduce host-side launch overhead (ignore).

Files: `src/vllm/v1/worker/gpu/spec_decode/dflash/{speculator.py (DFlashSpeculator, propose() ~line 330), cudagraph.py, utils.py}`,
`src/vllm/v1/worker/gpu/model_runner.py` (the decode step; grep `.tolist()`, `.item()`, `.cpu()`, `nonzero`, `synchronize`,
`non_blocking`), `src/vllm/v1/worker/gpu/cudagraph_utils.py`, `src/vllm/v1/worker/gpu/sample/{sampler.py,batch_shard.py}`,
the rejection/acceptance logic (grep `accept` under `src/vllm/v1/worker/gpu/`), and `src/vllm/v1/worker/gpu/states.py` or
similar for the persistent batch state. patch 0129/0131 context is in evidence/NOTES21.md only indirectly; read the code.

Deliver TWO patches:
1. `deliver/0142-dflash-sync-census-v0290.diff` — instrumentation only, knob `VLLM_SM12X_SYNC_CENSUS=1`: for N steps (default 50,
   after 20 warm-up) count per decode step the host syncs (`torch.cuda.synchronize`, `.item()`, `.tolist()`, `.cpu()`,
   `nonzero` with implicit sync, `torch.cuda.Event.synchronize`, non-`non_blocking` H2D/D2H copies) attributed to the call site
   (file:line of the vLLM frame), and the CUDA graph launches (replays) per step, then log ONE table (`info`) and disable itself.
   Implementation hint: monkey-patch the handful of tensor methods on entry to the step and restore on exit, or use
   `torch.cuda.set_sync_debug_mode("warn")` + a warnings hook to capture the stacks (cheaper and exact: it fires on every
   synchronizing op). Prefer the debug-mode route if the installed torch supports it (check `src/` for the torch version pin in
   evidence/launch-daily.sh / NOTES21) and fall back to monkey-patching. Zero overhead when unset.
2. `deliver/0142b-dflash-device-side-accept-v0290.diff` — knob `VLLM_SM12X_DFLASH_DEVICE_ACCEPT=1`: keep the accepted-token count,
   the next-step positions, and the drafter selector indices ON DEVICE between the target sampler and the drafter graph (and into
   the next target step), removing the D2H→Python→H2D round trips you identify as removable by reading the code. Where the host
   truly needs the value (scheduler output: number of tokens generated per request for the next scheduling round), keep ONE
   D2H copy per step, asynchronous, consumed as late as possible (at the end of the step after the drafter graph is launched).
   Unset ⇒ byte-identical. Proof line once: `DFlash device-side accept: <k> host syncs removed per step`.
   If a round trip cannot be removed without a scheduler change, describe it in NOTES and leave it.
Risk: this is the highest-correctness-risk brief of the set; the gate is the bf16 decode-fidelity ruler + greedy-equivalence,
not throughput. In NOTES25.md list every tensor whose lifetime you changed and why it is safe under CUDA graph replay
(static addresses, no reallocation) and under preemption / request finish mid-block.
