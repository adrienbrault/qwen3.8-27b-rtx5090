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

### Patch stacks added after 2026-07-22

| files | provenance |
|---|---|
| `patches/rc4/` (8 diffs + `rc4-stack.diff`) | Original fixes and the combined tier-rc4 stack, expressed as diffs against LMCache 0.5.4rc4 and vLLM source — context lines are Apache-2.0 code; offered upstream under Apache-2.0. |
| `patches-nvfp4kv/` | `upstream-pr49891-original.diff` is a verbatim redistribution of ch2lab's vLLM PR [#49891](https://github.com/vllm-project/vllm/pull/49891) (Apache-2.0, credited in the directory README's lineage section); the remaining diffs and the `overlay/` package are rebases/derivatives of that PR and of vLLM/FlashInfer code, Apache-2.0. |
| `patches-dflash2/` | Derived from vLLM PR [#52816](https://github.com/vllm-project/vllm/pull/52816) and syv-ai's backport (Apache-2.0, linked in the directory README). |
| `patches-v0280/` (the v0.28.0 daily generation, 2026-08-28) | `0101` is a re-rebase of **ch2lab's** vLLM PR [#49891](https://github.com/vllm-project/vllm/pull/49891) (sm12x NVFP4→FA2 routing) onto v0.28.0. `0102` carries **drowzeys'** linear-V-scale writer fix (first shipped in their [DGX Spark repo](https://github.com/drowzeys/keys-vLLm.0.27-Qwen3.8-27B-ADay777Ablit-NVFP4-A4Q-NVFP4-KV-4M-KV-token-pool-MTP3-Single-DGX-Spark)). `0103` (XQA-NVFP4 decode) implements the architecture demonstrated by the author of vLLM issue [#49011](https://github.com/vllm-project/vllm/issues/49011) (XQA decode + FA2 prefill on sm120); **hikarioyama's** [vllm-nvfp4-kv-sm120](https://github.com/hikarioyama/vllm-nvfp4-kv-sm120) (FA2 explicit-SF-stride) was prior art consulted for the gate/kernel mechanics. `0104` re-rebases the drafter FULL-cudagraph routing from PR #49891. The in-progress DFlash2-draft-on-NVFP4 route (`0105`) ports the non-causal `backend="fa2"` wrapper approach from **seanyourhighness's** [vllm-sm12x-nvfp4-dflash2](https://github.com/seanyourhighness/vllm-sm12x-nvfp4-dflash2) overlay (Apache-2.0), whose repo also validated the `--kv-cache-memory-bytes` pool pinning we use and whose DFlash2 selector-walk sampling rewrite informs our acceptance investigation. The dropped-async-flag doctrine follows **Ronald1995's** vLLM PR [#24799](https://github.com/vllm-project/vllm/pull/24799) (async scheduling × spec decode). All diffs are against Apache-2.0 vLLM/FlashInfer source; the patched images are Apache-2.0-derived. |

## Base images

- `vllm/vllm-openai:v0.28.0` (the current daily generation's base, `patches-v0280/Dockerfile.v0280-nvfp4kv`) — Apache-2.0.
- `vllm/vllm-openai:nightly` (pinned by digest in `patches/Dockerfile`, earlier generations) — Apache-2.0. This repo ships no vLLM binaries; the Dockerfiles pull the official image and patch it locally.
- `lmcache/vllm-openai:*` (historical, docs only) — Apache-2.0.

## Models (not redistributed here)

- [`RedHatAI/Qwen3.8-27B-NVFP4`](https://huggingface.co/RedHatAI/Qwen3.8-27B-NVFP4) — **the current daily's weights (since 2026-09-02)**: llm-compressor NVFP4 with 303 modules kept at 8-bit and an FP8 `lm_head`; served with its own chat template. Qwen3.8 base license. Chosen by the R156 fidelity ladder (bench/RESULTS.md); [`unsloth/Qwen3.8-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) and [`kelnei/Qwen3.8-27B-NVFP4`](https://huggingface.co/kelnei/Qwen3.8-27B-NVFP4) tie it on fidelity (same recipe family) and were the other two candidates.
- [`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090-LMHead4`](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090-LMHead4) — the 2026-08-22→09-02 daily (see the README provenance note: bit-identical to our 2026-08-22 download of the parent repo); every 2026-08 number in this repo was measured on it. Qwen3.8 base license.
- [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (bf16) — the ground truth for the fidelity ladder; [`RadixArk`](https://huggingface.co/RadixArk), [`QUASAR`](https://huggingface.co/QUASAR) QAT and the fp8 reference were the other rungs.
- [`syv-ai/DFlash2-Qwen3.8-27B-W4A16`](https://huggingface.co/syv-ai) — the DFlash2 draft used in speculative-decoding experiments; [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) is the original bf16 draft.
- [`sakamakismile/Qwen3.8-27B-MTP-NVFP4`](https://huggingface.co/sakamakismile/Qwen3.8-27B-MTP-NVFP4) — the 2026-08-14→22 daily; its Qwen3.8 XML chat template is what the gittensor daily served with.
- [`natfii/Qwen3.6-27B-VLM-NVFP4-MTP`](https://huggingface.co/natfii/Qwen3.6-27B-VLM-NVFP4-MTP) — the Qwen3.6-era daily (validated snapshot revision `2e46c0ed7606f35e357bc5674d20c710fc51b178`). Qwen3.6 base license (Apache-2.0).
- [`Lorbus/Qwen3.6-27B-int4-AutoRound`](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) — an earlier (Qwen3.6-era) daily, still the W4A16 reference.
- [`unsloth/Qwen3.6-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4) — earlier daily (TurboQuant era).

## Benchmarks and tools (not redistributed here)

- [Terminal-Bench / Harbor](https://www.tbench.ai/) — Apache-2.0.
- [SWE-bench](https://github.com/SWE-bench/SWE-bench) and [R2E-Gym](https://github.com/R2E-Gym/R2E-Gym).
- [llama-benchy](https://github.com/eugr/llama-benchy).
- [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench).
- [aider](https://github.com/Aider-AI/aider) — Apache-2.0.

## GPU driver P2P (dual-5090 host config)

The serving host runs [QuixiAI/open-gpu-kernel-modules](https://github.com/QuixiAI/open-gpu-kernel-modules) (Eric Hartford) — NVIDIA's open GPU kernel modules (dual MIT/GPL) with PCIe peer-to-peer force-enabled for consumer GeForce cards. The technique originates with [tinygrad / George Hotz's P2P patch](https://github.com/tinygrad/open-gpu-kernel-modules), later simplified by **aikitoria**, and ported to the 610 driver series with Blackwell support by QuixiAI. `scripts/gpu-p2p-610.sh` (ours, MIT) automates their install on Ubuntu, and the `RMForceStaticBar1=1` + `NVreg_EnableResizableBar=1` regkey pair plus the `iommu=pt` requirement come from that fork's documentation and our validation on a 2x RTX 5090 / X870 box.

## If you upstream this

The fixes in `fix_spec_output.py` and the six `patches/lmcache/` diffs belong in vLLM/LMCache, not in a patch repo. They are offered to those projects under Apache-2.0, on the same terms as any other contribution.

## Upstream work audited 2026-08-31 (verdicts in bench/RESULTS.md; credited regardless of adoption)

- **[vllm#53479](https://github.com/vllm-project/vllm/pull/53479)** (mamba-align boundary states) and **[vllm#53670](https://github.com/vllm-project/vllm/issues/53670)** (EAGLE last-block drop cost on hybrid GDN) — the prefix-cache line we backported and measured; the issue's warm-revisit methodology inspired our `probes/warm-revisit` probe.
- **[vllm#53426](https://github.com/vllm-project/vllm/pull/53426)** + **[vllm#51575](https://github.com/vllm-project/vllm/pull/51575)** — the `skip_draft_when_k0` mechanism and its runtime-K propagation; our 0115 backport reproduces both on v0.28.0.
- **[maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal](https://huggingface.co/maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal)** — calibrated W4A4 DFlash2 drafter; auditioning it exposed and fixed a real bug in our quantized-draft loader patch (0114).
- **[seanyourhighness/vllm-sm12x-nvfp4-dflash2](https://github.com/seanyourhighness/vllm-sm12x-nvfp4-dflash2)** — pinned all-NVFP4 DFlash2 overlay for sm12x on vLLM v0.27.1; independent evidence for the DFlash2-on-NVFP4 route and source of the eager-drafter/XQA-interference note.
- **[vllm#53979](https://github.com/vllm-project/vllm/pull/53979)/[#53978](https://github.com/vllm-project/vllm/pull/53978)/[#53977](https://github.com/vllm-project/vllm/pull/53977)** — the upstream FA2 non-causal NVFP4 route, now built and load-bearing here (patches 0116/0117; see bench/RESULTS.md R155).
