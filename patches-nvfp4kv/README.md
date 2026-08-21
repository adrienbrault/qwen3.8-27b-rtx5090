# patches-nvfp4kv — FA2 NVFP4 KV cache on the RTX 5090 (prepared 2026-08-16)

Goal: run `--kv-cache-dtype nvfp4` on flan's 5090 (sm_120) on the **same digest-pinned
vLLM nightly as the daily** (`0.27.2rc1.dev77+gac7509e2b`, FlashInfer 0.6.16.post3), so
the KV pool roughly **×1.7** vs fp8 (4.5 bits/elem incl. per-16 E4M3 scales vs 8) with the
FlashInfer FA2 tensor-core kernels — the pathway drowzeys shipped for the DGX Spark
(GB10, sm_121) in
[keys-vLLm.0.27-Qwen3.8-27B-…-NVFP4-KV-4M-KV-token-pool…](https://github.com/drowzeys/keys-vLLm.0.27-Qwen3.8-27B-ADay777Ablit-NVFP4-A4Q-NVFP4-KV-4M-KV-token-pool-MTP3-Single-DGX-Spark).

**Expectation management:** their "4.34M-token pool" is a **121 GB unified-memory** box.
On 32 GB: fp8 plain nightly = **207,042 @0.98** → nvfp4 ≈ **330–360K** (R35 measured
+73% same-model in July with the jethac fork: 136K→235K). Still: 262K-native single
sessions, ~×1.7 live concurrency at 45K, no sidecar, no 24 GB pinned DRAM. It is a
**PLAIN** profile: LMCache's rc4 patches 0001/0002 are fp8-page-specific — nvfp4 trades L2
persistence for live pool (until LMCache learns the `[data|scale]` page).

## Why three pieces (and why the writer patch is REQUIRED, not optional)

| piece | what | status upstream |
|---|---|---|
| FlashInfer FA2 NVFP4 attention on sm120 (JIT, head_dim 256) | kernels | **merged & shipped** in 0.6.16.post3 (#3097 Apr, #3640 SM120 Jun, #3897 SM121 Jul). Nothing to build. |
| vLLM routing: sm120 nvfp4 → FA2 (was SM100-trtllm-gen-only) | Python, PR **#49891** (ch2lab, tested on 5090 + Qwen3-Next hybrid + MTP) | OPEN, conflicts with main since 2026-08-07 → **rebased here as 0001** |
| vLLM store kernel: V block scales | csrc | in-tree writer (`csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu`, still identical on `main` 2026-08-16) **always swizzles V scales** in the SM100 trtllm-gen 4-token pattern. The FA2 reader (`include/flashinfer/attention/prefill.cuh` `page_produce_kv_sf`) reads K **and V** scales **linearly**. Mismatch ⇒ wrong scale on most V groups ⇒ **fluent output, wrong long-context recall, no crash** — plausibly the exact failure seanyourhighness reported on #49891 ("fluent, non-NaN, only visible on long-context recall"). drowzeys' 8-line fix (linear V scales when `major >= 12`) is **0002**, delivered as a runtime overlay op (no vLLM rebuild). |

## Files

- `0001-sm120-nvfp4kv-fa2-routing.diff` — PR #49891 rebased onto ac7509e2b, **narrowed**:
  `v1/attention/backends/flashinfer.py` (+ `gpu_worker.py` JIT warmup before memory
  profiling). 6/23 hunks were hand-merged (upstream drifted: `nvfp4_4over6` variant →
  `.startswith("nvfp4")`; upstream added XQA-decode-on-SM12x for fp8). Deliberate
  deviations from the PR, all to keep **fp8/auto on sm120 byte-identical to upstream**:
  - HND layout forced on sm120 **only when cache dtype is nvfp4** (PR forced it for all).
  - `use_trtllm_decode_attention`/XQA left as upstream for fp8; for nvfp4 on sm120 the
    builder forces `use_trtllm_decode_attention=False`, `flashinfer_trtllm_api_decode_kernel=None`
    (XQA/trtllm-gen cannot read NVFP4) → native FA2 decode; spec-decode verify (q_len>1)
    goes through the FA2 prefill wrapper (the "tested prefill fallback" from the PR thread),
    with the PR's cudagraph prefill wrappers for uniform 1+ns batches.
  - dropped the redundant `supports_trtllm_attention` hunk (upstream already handles SM12x).
- `0001b-prefill-wrapper-signature.diff` (the `batch_size`/`use_cudagraph` params of `_get_prefill_wrapper` — they sat in the XQA-drift hunk that failed the rebase and were dropped; first boot died with `unexpected keyword argument 'batch_size'`), `0002-nvfp4-writer-linear-vscale-sm12x.diff` — csrc diff (upstream-repo paths). Applied to
  a csrc checkout at build; the patched `.cu` + `overlay/overlay_binding.cpp` compile into
  `torch.ops.vllm_sm12x.reshape_and_cache_nvfp4` (arch `sm_120a`; GB10 would be `121a`).
- `0002b-flashinfer-route-nvfp4-store-to-overlay.diff` — routes the store to the overlay when
  `use_fa2_nvfp4_kv`. **`VLLM_SM12X_NVFP4_LINEAR_VSF=0`** = in-tree swizzled writer (diagnostic
  lever: predicts depth-needle failures; never for serving). Missing overlay logs a loud
  warning and falls back (launcher makes that FATAL).
- `0003-pr49891-mtp-drafter-full-cudagraph.diff` — PR's MTP drafter FULL-cudagraph routing
  (`llm_base_proposer.py`, `gpu_model_runner.py`, dflash/extract_hidden_states signature).
  Perf lever, orthogonal to nvfp4, **off by default** (`--build-arg WITH_0003=1`) so a recall
  failure can be attributed (KV path vs drafter routing).
- PR's KV-offloading abort fix (offloading connector) is **not included** — unused here.
- Not ported from drowzeys' stage1: their `compressed_tensors.py` tweak (accept `kv_cache_scheme`
  num_bits=4 in a checkpoint's quant config) — only needed for checkpoints that ship a 4-bit KV
  scheme (their aday777); saka carries none, `--kv-cache-dtype nvfp4` is CLI-side. Revisit if a
  future checkpoint errors on `kv_cache_scheme`.
- `overlay/` — `overlay_binding.cpp`, `build_overlay.py` (AOT at docker build; no GPU
  needed), `vllm_sm12x_nvfp4kv.py` (runtime loader; JIT fallback if the .so is missing).
- `Dockerfile.nvfp4kv` → `vllm-qwen38:nvfp4kv`. Base digest is the daily's — do not bump.
- `upstream-pr49891-original.diff` — the PR as fetched 2026-08-16, for audit.
- Launcher: `scripts/serve-nvfp4kv.sh` (:8029/vllm-exp, plain, FLASHINFER forced,
  async-off, align, mnbt 4096, `NS=4` default with `NS=3`/`NS=0` envs, util 0.95, max-len 200K
  for protocol-matched A/Bs; 262,144 native is the later stretch).

## Build + boot (on flan)

```bash
# 1) ship the build context (from the Mac; repo-first, then apply from the artifact)
rsync -a patches-nvfp4kv/ flan:/srv/qwen5090/nvfp4kv-build/
# 2) build (~5-10 min: git csrc checkout + one nvcc + patch), no GPU needed
ssh flan 'cd /srv/qwen5090/nvfp4kv-build && sudo docker build -f Dockerfile.nvfp4kv -t vllm-qwen38:nvfp4kv .'
# 3) boot the audition engine (daily/eval engine must be down; first boot JITs FA2 nvfp4 kernels ~15 min)
ssh flan 'bash -s' < scripts/serve-nvfp4kv.sh
```

## Gauntlet (promotion blockers in bold; record everything under
`/srv/qwen5090/results/2026-08-XX-qwen38-nvfp4kv/`)

1. **Boot facts**: `GPU KV cache size` (expect 330–360K @0.95/200K), `Maximum concurrency`,
   `kv_cache_dtype=nvfp4`, backend line (`FLASHINFER`, decode_backend), unified block size
   (drowzeys saw 2848 on GB10 → the mnbt≥block_size + MTP-autocap gotcha), overlay ACTIVE line,
   MTP acceptance in the metrics. NO pool band on first boot: log, then set.
2. **Depth needles, cold + warm**, `kv_quality.py` style, 9K/20K/40K/60K
   (+100K/150K at 200K max-len). Run **MTP on (NS=4) AND MTP off (NS=0)** — separates the two
   risk layers (writer/reader layout vs spec-decode verify routing on the native path).
   Then the **diagnostic**: `LINEAR_VSF=0` must FAIL the deep needles; if it passes, the
   whole layout theory is wrong — stop and re-derive.
3. **The "sean gate" verbatim: cold + cached multi-request 32K+ recall under MTP on the exact
   selected backend** (concurrent 3–6 loaders, ≥30 samples). Fluent-but-wrong is the failure
   mode; single short probes cannot see it.
4. Killer 16/16 (fresh-seed 8×24K burst), vision burst 8/8, structured-output probe.
5. Quality: tool-eval 69×2 @T0.6 effort=medium vs the fp8 band (nightly plain 91±0, tier-rc4
   92.5±0.7). Retrieval-clean but −3+ pts = 4-bit-VALUES cost (R32 saw 82 vs 85, all in the
   structured-output cluster; re-check that cluster specifically).
6. Perf: llama-benchy pp{512,4096} × c{1,2,4,8} AND **deep pp30K c1 decode vs fp8** — July's
   FA2 nvfp4 was −8..−23% at depth (split-KV disabled in the nvfp4 FA2 path, FlashInfer
   #3684 era). Measure; do NOT assume drowzeys' "≈neutral" transfers from a 273 GB/s box.
7. Optional: rebuild with `WITH_0003=1` → repeat 2–3 + benchy (MTP drafter FULL cudagraph).
8. Verdict + FINDINGS round + public-repo update (numbers only after measurement).

## Lineage / credit
tiffany940107 (SM120 NVFP4 attention JIT, FlashInfer #3640) · bkryu (#3897) · Tom-Zheng
(#3097 FA2 NVFP4 KV) · jethac (SM121 + A4Q fp4-QKᵀ, #3684) · ch2lab (vLLM #49891) ·
drowzeys (GB10 integration incl. the linear-V-scale writer diagnosis) · rebase/narrowing
+ 5090 adaptation here. Upstream drafts (writer arch-conditional + #49891 rebase notes)
are candidates for `results/.../upstream-drafts/` — filing stays user-gated.
