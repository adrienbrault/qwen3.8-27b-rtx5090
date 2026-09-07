# R166, the nvfp4-KV candidate made daily-grade: the KV pool pinned in bytes instead of sized by utilization; every promotion gate except SWE-bench and the tier half of the revisit gate passes paired against the fp8 daily (2026-09-03, `results/2026-09-03-r166-gates`, `scripts/serve-nvfp4-candidate.sh`, `scripts/r166-candidate-gates.sh`)

[← all results](../RESULTS.md)

**Why the candidate would not boot with headroom.** vLLM v0.28 sizes the KV pool from `--gpu-memory-utilization` before it captures CUDA graphs, and its pre-capture graph estimate is zero (R164). The nvfp4 route's graphs are large (0.72 / 1.20 / 1.70 GiB at 8 / 16 / 32 sequences with patch 0131), so at util 0.90 the candidate booted with 3 MiB free at 16 sequences and not at all at 32. `--kv-cache-memory-bytes` takes the pool size verbatim and skips that profiling (`gpu_worker.py:489`; the pinning approach was validated in seanyourhighness's overlay repo, see THIRD_PARTY.md). The first pinned boot showed the second half of the problem: the pinned path also skips the profiler's activation-peak reserve, so 13.41 GiB gave the predicted pool (931,214 tokens) with only 101 MiB free after the pre-warm. The launcher now pins per sequence count and fails its own boot below 384 MiB free after pre-warm.

| SEQS | pin per GPU | pool (tokens) | graphs | free after pre-warm | full-window streams |
|---|---|---|---|---|---|
| 8 (the daily's contract) | 12.85 GiB | **892,276** (fp8 daily: 657,269, +36%) | 0.72 GiB | **1,099 MiB** (fp8 daily: 367) | 3.40x (fp8 2.51x) |
| 16 | 12.37 GiB | 858,823 | 1.20 GiB | 857 MiB | 3.28x |
| 32 | 11.87 GiB | 823,724 | 1.70 GiB | 597 MiB | 3.14x |

All three booted first try with every boot assert green (XQA off, batched-token cap 8192, 512 MiB FlashInfer workspace, 0131 active, drafter graphs captured, pinned budget honoured, checkpoint identity). The pool is deterministic by construction.

**Gates, paired the same hour on the experiment port.** fp8 arm = the daily shape (util 0.92, 8 sequences); candidate = the launcher above in experiment mode, 8 sequences.

| gate | fp8 daily shape | nvfp4 candidate | verdict |
|---|---|---|---|
| needles 9K/20K/131K/220K/258K ×2 | 8/8 (the 258K rows were a probe overrun; the fp8 arm has no real 258K pair, the probe was fixed after it ran) | **10/10**, two real 258K rows | pass |
| needles at wider layouts | – | 16 seq: 131K/220K 4/4, 131K under 8 concurrent 20K loaders 2/2; 32 seq: 258K 2/2 | pass |
| warm revisit 32K | 7.49 s → 0.46 s | 7.61 s → 0.68 s, 50,048 block hits | pass for the GPU prefix cache; the disk-tier half is open, see below |
| benchy c1, 5 runs | 277.5 ± 21.7 (247–312) | 272.3 ± 28.1 (234–321) | parity; both arms show the box's two modes |
| benchy c8 | 651.0 ± 6.6 | 617.1 ± 12.9 | −5% on the ramp-inclusive number |
| steady-state decode, code c8 | 1,143 | **1,148** | parity |
| steady-state decode, prose c8 | 894 | 877 | −2% |
| prose c1 at 30K context | 150.6 | 144.4 | −4%, inside both spreads |
| tool-eval 69×4 | 89.0 ± 1.2 | **89.0 ± 0.8** | parity |
| fidelity vs the FP8 reference (ΔNLL / top-1 / KL) | +1.29% / 0.9267 / 0.101 | +1.61% / 0.9238 / 0.108 | 0.29 pp top-1, under the 0.4 gate |
| nvfp4 KV vs fp8 KV directly, same checkpoint and image | – | ΔNLL +0.32%, top-1 0.932, KL 0.092 | just above the ruler's own run-to-run noise |
| engine error lines / preemptions | 0 / 0 | 0 / 0 (one preemption at 16 seq while a manual probe overlapped the flood) | pass |

**What the bigger pool buys, and what it does not.** At 32 sequences the ladder is c8 631 / c16 617 / c32 650 (fp8: 643 / 640 / 667) and the 1 s sampler saw at most 14 requests running with none waiting (fp8: 17). That is one sample from a pp2048/tg256 ramp, where the fp8 17 came from long 2K-token generations with every request in flight, so 14 is consistent with the layout arithmetic below (which predicts about 14.7) rather than a measured ceiling. The nvfp4 boot sets the attention block to 2,944 tokens (fp8: 1,664) so that one attention page still equals one mamba page in bytes; pages are the same size in bytes on both routes and each request holds one page in every cache group, so admission of short requests follows pool bytes, and the pinned nvfp4 pool is smaller in bytes. The +36% token pool is long-context capacity (3.40 vs 2.51 full-window streams), not more concurrent short requests. The daily's 8-sequence contract is unaffected.

**Probe calibration.** The needle probe's filler assumed 1.3 tokens per word; the served tokenizer gives 1.45, so every requested depth landed at 1.117× in real tokens (the "220K" rows were 245K, and 258K requests overran the window with HTTP 400 on every image). Calibrated now; depths labelled before this run are about 12% deeper than their labels.

**The revisit gate is only half done, and the half that passed proves less than it looks.** Every revisit that passed, here and in a follow-up unit that restarted the container over the kept tier, was a GPU prefix-cache hit (the restart send came back cold, 7.98 s against a 7.54 s control, with disk reads in the log that could not be attributed because they overlapped the boot pre-warm). No test had checked that nvfp4 blocks read back from the CPU/disk tier decode to the right answer, and that is the failure mode a 4-bit layout bug would produce: fluent, confident, wrong. The needle probe now has an evict-and-re-ask mode (`--evict N`: N unique 90K prompts push the needle out of the GPU pool, then the byte-identical prompt is asked again, so the answer must come through the tier; the same seed after a container restart re-asks the same prompt; each pass records the engine's external and tier hit counters). A paired unit (`scripts/r166c-tier-gate.sh`) runs it on both arms after the campaign, and the campaign's own post-run needle gate uses it too, on the engine that just served 500 instances.

**The paired tier unit ran (2026-09-03 19:31 UTC, `results/2026-09-03-r166c-tier`) and found the gate unpassable as written, on both KV formats.** Every needle hit on both arms (cold, after eviction, after a container restart), but the engine's external-hit counter stayed at zero on every pass, fp8 included: the evicted re-ask took a full recompute (25 s at 131K) while the disk tier read 4–7 GB in the background, and the restart revisit did the same. The reason is in the tiering code: the disk lookup is asynchronous and promotes blocks into the CPU tier in the background, so the request that triggered the lookup is already being recomputed, and only a later touch of the same prefix can be served from the CPU tier (4 GiB by default, which an eviction flood flushes). That is a property of the tier configuration, not of nvfp4, and it means the first revisit of an evicted prefix is always recomputed on the daily as well. The follow-up (`scripts/r166d-tier-served.sh`, `needle_depth.py --evict-reasks 3`) re-asks the evicted needle three times in a row so the later touches can be tier-served, records whether any pass was actually served and whether it still hit, and adds a 24 GiB CPU-tier arm if the daily-sized tier still serves nothing.

**Status.** Every gate except SWE-bench and the tier half of the revisit gate passes or is a stated trade. The candidate's SWE-bench Verified campaign (same harness and versions as R160, cold tier, needle gate before and after) runs next on the pinned launcher; promotion is decided after it pairs against R160's 386/500.
