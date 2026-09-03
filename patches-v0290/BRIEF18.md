# BRIEF 18 — 0135: input-embedding table in pinned host RAM (UVA) on the v0.29.0rc1 chain

Scope: package vLLM PR #53981 ("Route directly constructed embeddings through CPU offload", open, base main 2026-08-27)
as `0135-embed-uva-offload-v0290.diff` on top of our v0.29.0rc1 chain (…-revival-prs tree = 0101–0134 applied), and
REVIEW the parts its author did not exercise on our shape. One pass, `reasoning_effort=medium`. Verify ONLY with
`patch --dry-run -p1 --fuzz=0` + `python3 -m py_compile` on a scratch copy; no tests, no GPU, no model execution.
Never touch the host flan (no ssh, no docker). The operator builds the image and runs every launch.

## Why

Qwen3.8-27B's input embedding is an untied BF16 table 248,320 × 5,120 = 2.37 GiB (`model.language_model.embed_tokens.weight`,
lm_head separate, FP8). On the dual-RTX-5090 box it is sharded by vocab under TP=2 (`VocabParallelEmbedding`), 1.19 GiB per
GPU, and it is read a few rows at a time. That is KV-cache memory: at the candidate's 15,466 B/token/GPU the shard is ≈80K
pool tokens (+9% on the 892K SEQS-8 pool). vLLM already has the mechanism (`--offload-backend uva --cpu-offload-gb N
--cpu-offload-params embed_tokens`: pinned host copy, `get_cuda_view_from_cpu_tensor` UVA view, zero-copy reads over
PCIe, CUDA-graph safe) but the offloader is only applied inside `make_layers`, so the selector silently matches nothing for
`embed_tokens` built directly (issue #53931 — filed with exactly our model as the repro). PR #53981 routes directly
constructed `VocabParallelEmbedding` modules through the UVA offloader after construction, before checkpoint loading,
and warns on unmatched selectors. Reference for the same idea in SGLang: PR sgl-project/sglang#37826 (Triton host gather;
in vLLM the plain `F.embedding` on the UVA view plays that role — do NOT port the Triton kernel).

## Inputs (all local, read-only; `$B` = `<scratchpad>/embed-offload`)

- `$B/tree/vllm` — the vLLM package extracted from image `vllm-qwen38:v0290rc1-nvfp4kv-revival-prs` (rc1 + 0101–0134
  applied). Tree root = `$B/tree` (so `patch -p1 -d $B/tree` with `a/vllm/...` paths). Copy it before applying anything.
- `$B/pr53981.diff` — the PR diff as fetched (`gh pr diff`). Its `tests/basic_correctness/test_cpu_offload.py` hunk must
  be dropped (no tests/ in the image). The operator confirmed `git apply --exclude='tests/*' --check` passes on this tree.
- `$B/sglang-37826.diff` — the SGLang PR, reference only.
- `$B/refs/0116-dflash-nvfp4-revival.diff` — OUR hunk in `vocab_parallel_embedding.py` (tp1 masking path) — check the
  interaction with a UVA-view weight. `$B/refs/0134-…diff` + `$B/refs/Dockerfile.prs` — the format of a ported upstream PR
  in this chain and how the image layer applies it (`patch --batch --forward -p1 --fuzz=0` from the dist-packages dir).
  `$B/refs/verify.sh` — the chain verifier to extend.
- `$B/refs/launch-daily-nvfp4-candidate.sh` + `launch-daily-v0280.sh` — how the engine is launched (EXTRA_ARGS / EXTRA_ENV
  reach `vllm serve`; boot asserts are `grep -c` lines on the container log and `docker inspect` args). The KV pool is
  PINNED with `--kv-cache-memory-bytes` (no util-based profiling), so freed VRAM does NOT become pool by itself — the
  pin has to be raised by the freed bytes.

## Review items (answer each in NOTES18.md with file:line evidence from `$B/tree`)

1. TP=2 path: `VocabParallelEmbedding.forward` with `tp_size > 1` (masked_fill → `quant_method.embedding` → masked
   output → all-reduce) when `self.weight.data` is a UVA view. Does anything in that path, in our 0116 hunk, or in
   `UnquantizedEmbeddingMethod.embedding`, materialise or copy the whole table (`.to`, `.contiguous`, `.clone`, dtype
   cast, `torch.compile` constant-folding of the parameter)? Does the per-rank shard land in host memory correctly
   (offload happens before load: the loader must write INTO the view; check `VocabParallelEmbedding.weight_loader` and
   `process_weights_after_loading` for anything that re-creates the parameter on the device).
2. Where exactly the PR hooks (`maybe_offload_embeddings` call sites in `model_loader/utils.py`) relative to OUR
   chain's model construction (qwen3_next.py, the multimodal wrapper, `SupportsMultiModal._mark_tower_model`), and
   that `model.visual.pos_embed` / `patch_embed` are NOT matched by the exact-segment selector `embed_tokens`.
3. The DFlash2 drafter (`/draft` = `dflash2-qwen38-syvai-w4a16`, loaded through our 0107 quantized-draft loader; find
   the drafter model class in `$B/tree/vllm/model_executor/models/` and `vllm/v1/spec_decode/`): does it own an
   `embed_tokens` or share the target's? If it owns one, the selector matches it too (state the bytes, and whether the
   draft model's construction path even passes through the PR's hook). Say what `--cpu-offload-gb` must be so BOTH
   tables fit if both are matched (budget is per GPU, checked per parameter in `UVAOffloader._maybe_offload_to_cpu`).
4. CUDA graphs: our 0129 drafter full graphs + the V2 runner's decode graphs + 0119 uniform width capture the
   embedding lookup. A kernel reading pinned host memory through UVA inside a captured graph is legal; confirm nothing
   in `cudagraph_utils.py` / `breakable_cudagraph.py` (`sync_prev_onload` / `join_after_forward` calls are for the
   PREFETCH backend) changes behaviour for UVA, and that `torch.compile` (inductor) treats the UVA parameter as an
   ordinary CUDA tensor (no `copy_` into a device buffer per forward).
5. Fallback trap: when `is_uva_available()` is False or `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1`, the offloader wraps
   `forward` with `functional_call` and streams the module's state to the device EVERY forward (1.19 GiB per step —
   catastrophic, silent). Give the operator the exact log lines / conditions to ASSERT at boot that the UVA path is in
   force (e.g. "Offloader set to UVAOffloader", "Total CPU offloaded parameters: …", absence of the unmatched-selector
   warning the PR adds, and anything that identifies the UVA vs functional_call branch).
6. Pinned host memory inside the container: `should_pin_memory()` conditions; whether 2.37 GiB (+ drafter) of
   `pin_memory()` needs any docker flag (we already run `--ipc=host`; note `--ulimit memlock` if relevant) and how it
   interacts with the tier's 4 GiB CPU KV buffer (OffloadingConnector) — both are pinned allocations on a 64 GB host.

## Deliverables (write into `$B/out/`)

1. `0135-embed-uva-offload-v0290.diff` — the PR minus its tests hunk, tree-root relative (`a/vllm/...`), applying with
   `patch --batch --forward -p1 --fuzz=0` on a fresh copy of `$B/tree` AFTER nothing else (the tree already has 0101–0134).
   Prefer verbatim; if a review item forces a change (e.g. the TP path), make it a SEPARATE `0136-…diff` with a rationale,
   never fold it into the port.
2. `Dockerfile.embed` — `FROM vllm-qwen38:v0290rc1-nvfp4kv-revival-prs`, applies 0135 (+0136 if any) the way
   `Dockerfile.prs` does, py_compiles the touched files, greps a marker (`maybe_offload_embeddings` in
   `model_loader/utils.py`), prints `EMBED-0135-APPLIED`. Target tag `vllm-qwen38:v0290rc1-nvfp4kv-revival-prs-embed`.
3. `verify-0135.sh` — dry-run + real apply + py_compile on a scratch copy of `$B/tree`, plus a CONTROL (second apply
   must fail), exit non-zero on any failure.
4. `NOTES18.md` — the six review answers, the launch block for the operator (exact `EXTRA_ARGS` additions:
   `--offload-backend uva --cpu-offload-gb <N> --cpu-offload-params embed_tokens`; the asserts from item 5 as grep
   patterns; the expected "Total CPU offloaded parameters" value per rank), the KV-pin arithmetic (bytes freed per GPU →
   new `--kv-cache-memory-bytes` at SEQS 8/16/32 keeping the same ≥384 MiB free-after-pre-warm margin; current pins
   13,800,000,000 / 13,280,000,000 / 12,740,000,000), and a "risk on metal" list ordered by what the operator should
   look at first (boot log line, then which probe catches it: fidelity ruler, needles, decode c1/c8, prefill).

## Constraints

- Do not reformat or reflow untouched code; hunks minimal; `--fuzz=0` must hold.
- One pass. If something cannot be settled from source, say so in NOTES18 as an open question with the experiment that
  settles it; do not invent behaviour.
