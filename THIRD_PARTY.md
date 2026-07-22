# Third-party code and attribution

This repository is MIT-licensed (see [LICENSE](LICENSE)) for the **original** work: documentation, benchmark harnesses/probes, serve scripts, and the patch-installer tooling. Everything below is derived from Apache-2.0 projects and **remains Apache-2.0-derived** — the MIT grant does not extend to upstream-derived content, and the patched files inside any image you build are derivative works of their upstream projects.

## Derived from vLLM (Apache-2.0)

Upstream: https://github.com/vllm-project/vllm — [Apache-2.0](https://github.com/vllm-project/vllm/blob/main/LICENSE).

| file | derivation |
|---|---|
| `patches/vllm-only.diff` | **Verbatim redistribution** of the vLLM-source portion of [vllm#40914](https://github.com/vllm-project/vllm/pull/40914) (TurboQuant K+1 spec-verify routing), authored by **@Sandermage**, contributed to vLLM under Apache-2.0. Redistributed unmodified so the setup is reproducible while the PR is unmerged. |
| `patches/so_reasoning_44993.diff` | **Redistribution** of the diff from [vllm#44993](https://github.com/vllm-project/vllm/pull/44993) (structured-output grammar across reasoning boundary), authored by **@yuyue0225sc**, contributed to vLLM under Apache-2.0. PR open at time of writing. |
| `patches/install_pr42603_sync.py` | Original installer (MIT), but the one-line change it grafts is **from [vllm#42603](https://github.com/vllm-project/vllm/pull/42603)** (closed unmerged) and the anchor snippets it matches are quoted vLLM source. The grafted `llm_base_proposer.py` in the image is Apache-2.0-derived. |
| `patches/fix_spec_output.py`, `patches/fix_spec_guard.py`, `patches/tq_auto_fallback.py`, `patches/tq_splits.py` | Original installers (MIT) that modify vLLM source files at build time and quote short anchor snippets to locate edit sites. The resulting patched files are Apache-2.0-derived. |
| `patches/lmcache/0003-vllm-connector-eagle-hybrid-hit.diff`, `patches/lmcache/0005-vllm-residual-mamba-connector-prefill-boundary.diff` | Original fixes (offered upstream under Apache-2.0), expressed as diffs **against vLLM source** — the context lines are vLLM code and the patched files are Apache-2.0-derived. |
| `patches/lmcache-0.5.1-format10-NL_X_NB_NH_BS_TWO_HS.patch` | Withdrawn historical patch (see `patches/README.md`); diff against LMCache source, Apache-2.0-derived. |

Files modified in the image by the entries above include: `vllm/v1/spec_decode/llm_base_proposer.py`, `vllm/v1/structured_output/__init__.py`, `vllm/v1/core/sched/scheduler.py`, `vllm/v1/worker/gpu_model_runner.py`, `vllm/v1/attention/backends/turboquant_attn.py`, `vllm/model_executor/layers/quantization/turboquant/config.py`, `vllm/config/attention.py`.

## Derived from LMCache (Apache-2.0)

Upstream: https://github.com/LMCache/LMCache — [Apache-2.0](https://github.com/LMCache/LMCache/blob/dev/LICENSE).

| file | derivation |
|---|---|
| `patches/lmcache/0001-fix-fused-hybrid-subpage-view.diff`, `0002-strided-fp8-regroup.diff`, `0007-sidecar-vram-staging-batch.diff`, `0008-fs-native-cap-enforcement.diff` | Original fixes (offered upstream under Apache-2.0), expressed as diffs against LMCache source (Python and `csrc`) — context lines are LMCache code and the patched build is Apache-2.0-derived. |

## Base images

- `vllm/vllm-openai:nightly` (pinned by digest in `patches/Dockerfile`) — Apache-2.0. This repo ships no vLLM binaries; the Dockerfiles pull the official image and patch it locally.
- `lmcache/vllm-openai:*` (historical, docs only) — Apache-2.0.

## Models (not redistributed here)

- [`natfii/Qwen3.6-27B-VLM-NVFP4-MTP`](https://huggingface.co/natfii/Qwen3.6-27B-VLM-NVFP4-MTP) — **the current daily's weights** (validated snapshot revision `2e46c0ed7606f35e357bc5674d20c710fc51b178`). Qwen3.6 base license (Apache-2.0).
- [`Lorbus/Qwen3.6-27B-int4-AutoRound`](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) — the previous daily, still the W4A16 reference.
- [`unsloth/Qwen3.6-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4) — earlier daily (TurboQuant era).

## Benchmarks and tools (not redistributed here)

- [Terminal-Bench / Harbor](https://www.tbench.ai/) — Apache-2.0.
- [SWE-bench](https://github.com/SWE-bench/SWE-bench) and [R2E-Gym](https://github.com/R2E-Gym/R2E-Gym).
- [llama-benchy](https://github.com/eugr/llama-benchy).
- [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench).
- [aider](https://github.com/Aider-AI/aider) — Apache-2.0.

## If you upstream this

The fixes in `fix_spec_output.py` and the six `patches/lmcache/` diffs belong in vLLM/LMCache, not in a patch repo. They are offered to those projects under Apache-2.0, on the same terms as any other contribution.
