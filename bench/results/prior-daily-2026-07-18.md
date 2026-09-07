# Prior daily (2026-07-18): Lorbus INT4-AutoRound + fp8 + FlashInfer + MTP ns=4 (PR #42603)

[← all results](../RESULTS.md)

Config: Qwen3.6-27B INT4 ([Lorbus AutoRound](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound)) + `fp8_e4m3` KV + FlashInfer 0.6.15 + `--mamba-cache-mode align` + MTP `ns=4`, image `k8v4-so-pr42603` (base image plus [PR #42603](https://github.com/vllm-project/vllm/pull/42603)). Pool **287,323 tok** at util 0.98 / `--max-num-batched-tokens 4096`, 200K max-len. The decode, long-context and tool-eval tables below were measured at the earlier util 0.94 config, pool 253,521; decode is bandwidth-bound and util-invariant, so they hold unchanged. See [the util sweep and ceiling probe](prior-daily-2026-07-18.md#pool-vs-util-util-is-the-only-pool-lever) for the pool increase.

The bug it works around is a known, still-open upstream class: MTP × fp8 KV × Blackwell `sm_120` illegal-memory-access under concurrency ([vllm#40756](https://github.com/vllm-project/vllm/issues/40756) on the same Qwen3.6-27B-FP8 model, and [vllm#35288](https://github.com/vllm-project/vllm/issues/35288), "MTP corrupted output at concurrency ≥ 4"). Under concurrency the crash is 100% reproducible (`rejection_sampler.py:267 parse_output` → `cudaErrorIllegalAddress`). Single-stream and `ns=1` are both clean, and `CUDA_LAUNCH_BLOCKING=1` masks it, which points to a timing race. The hypothesis in [PR #42603](https://github.com/vllm-project/vllm/pull/42603) is that the MTP draft loop in `llm_base_proposer.py` writes shared cudagraph buffers, then launches the draft forward that reads them without a sync; its one-line `synchronize()` is what this image grafts. **The PR was closed unmerged**, with maintainers disputing the race explanation pending a proven root cause, so the graft is a locally validated workaround: on this profile the crash is 100% reproducible without it and was never observed with it. Device-wide barriers placed around the proposer in `gpu_model_runner` do not stop it, and `--mamba-cache-mode all` and a draft-token sanitizer both failed too. Full bisection: [HISTORY.md](../../docs/HISTORY.md).

Stability: every axis that reliably IMA-crashed pre-patch showed zero crashes. Those axes are full concurrent c4/c8 (pp512+pp4096, ×3), a repeat c8 ×5 stress, deep pp30000 × c4, **deep pp90000 × c4** (worst case: deep, concurrent, K=4), and the full 69×2 tool-eval under load.

Decode, measured with llama-benchy 0.3.8, `--pp 512 4096 --tg 128 --concurrency 1 2 4 8 --runs 3 --skip-coherence`, `t/s (total)` (aggregate), util 0.94, `--no-async-scheduling`:

| decode t/s (total) | c1 | c2 | c4 | c8 |
|---|---|---|---|---|
| @512 | 114 | 212 | 355 | 496 |
| @4096 | 129 | 164 | 198 | 157 |

Long context (c1), prefill / e2e-TTFT / decode. Decode is **flat ~128–133 t/s from 30K→180K**, with no deep crater, from the fp8 + FlashInfer attention kernel, now with `ns=4` spec:

| context | prefill t/s | e2e TTFT | decode t/s |
|---|---|---|---|
| 30K | 3,653 | 7.4 s | 128 |
| 90K | 2,924 | 27.9 s | 132 |
| 180K | 2,249 | 72.4 s | 133 |

Deep context also holds under concurrency: pp90000 × c4 stays alive at ~102 t/s/req. **tool-eval-bench: 90** (full 69-scenario suite ×2 trials; mean 88 ± 2.8). The PR #42603 sync is performance-neutral: an `align`+sync image matched an `all`-without-sync image on decode, so restoring the sync costs no measurable throughput.

## Pool vs util: util is the only pool lever

Sweep of `--gpu-memory-utilization × --max-num-batched-tokens` over `{0.94, 0.95, 0.96} × {8192, 4096}`, everything else fixed. Each cell was checked for a boot-profiling OOM and a runtime cold-start OOM (8 simultaneous fresh ~16K-token completions, which a ramping benchmark never trips). Prefill t/s is `--pp … --concurrency 1`:

| util | mnbt | KV pool | pp512 | pp4096 | pp30000 | pp90000 | 8× cold-start burst |
|---|---|---|---|---|---|---|---|
| 0.94 | 8192 | 253,521 | 132 | 888 | 7,388 | 2,919 | alive |
| 0.94 | 4096 | 253,521 | 134 | 902 | 7,506 | 2,843 | alive |
| 0.95 | 8192 | 261,971 | 137 | 906 | 7,498 | 2,921 | alive |
| 0.95 | 4096 | 261,971 | 139 | 920 | 7,637 | 2,838 | alive |
| 0.96 | 8192 | 270,422 | 132 | 932 | 8,359 | **2,650** | alive |
| **0.96** | **4096** | **270,422** | 135 | 908 | 7,519 | **2,833** | alive |

Findings: (1) `mnbt` does not change the pool, which is identical at each util, because chunked prefill already bounds the transient, so `mnbt` only sets the chunk size and not the steady-state allocation. util is the only pool lever here: **+8,450 tok per 0.01**. (2) No OOM occurred anywhere in the sweep. (3) The only interaction observed: at high util, `mnbt 8192` slows deep prefill ~9% from allocator pressure at the big-pool and big-chunk corner, while `mnbt 4096` recovers to the 0.94 baseline, so `mnbt 4096` is the daily. This reverses the earlier `mnbt 4096` rejection, which was measured on the TurboQuant NVFP4 config where lowering `mnbt` freed pool at a prefill cost. Neither effect holds on this AR + fp8 stack.

Ceiling probe: **0.97 and 0.98 both survive**, text and vision. Two effects make naive burst tests misleading: (a) identical prompts are collapsed by prefix caching and never fill the pool, so capacity bursts must use prompts that differ from token 0; (b) text-only bursts miss the vision-encoder transient, a common post-profiling OOM on a multimodal daily. With both fixed:

| util | pool | text burst ~98% of pool | text burst ~104% (oversubscribed) | vision burst | mixed |
|---|---|---|---|---|---|
| 0.97 | 278,873 | ✅ 8× 200 | ✅ 8× 200 | *(covered by 0.98)* | |
| **0.98** | **287,323** | ✅ 8× 200 | ✅ 8× 200 | ✅ **8× concurrent, 4× 2048² images each — 8/8 real replies** | ✅ 4 vision + 4 deep-text (~30K) |

Zero OOM or IMA anywhere, and oversubscription preempts cleanly. The daily runs util 0.98 / `mnbt` 4096 for a **pool of 287,323**, leaving ~600 MB VRAM. What 0.98 lacks is margin for what these probes cannot cover: multi-day fragmentation or a future colocated sidecar. If either occurs, fall back to 0.96 (270,422) or 0.94 (253,521).

---

*The sections below are prior dailies and alternatives on the Unsloth NVFP4 model, kept for the record.*
