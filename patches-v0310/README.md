# patches-v0310: the served chain on vLLM v0.31.0

The served image since 2026-10-17, `vllm-qwen38:v0310-r27c-A_S-L1a-ubplan` (sha256:631a6f24ccca2f5470a3d0772d833670a16613c69bb4191663f718e473086f5e), is three builds on the serving host, offline (`--network=none --pull=false`) and without a GPU:

1. `vllm-qwen38:v0310-r27c-A-nvfp4kv-mtp` from [Dockerfile.v0310](Dockerfile.v0310): `vllm/vllm-openai:v0.31.0` (FlashInfer 0.7.0.post1, torch 2.13.0, CUDA 13.0), the 0102 store overlay compiled for `sm_120a`, the chain in [series](series) applied one logical layer at a time with `patch --fuzz=0` by [apply_layer.py](apply_layer.py) (which writes a marker per patch into `/opt/prs-markers/`), the FlashInfer workspace guard [guard_flashinfer_workspace.py](guard_flashinfer_workspace.py) and its audit [audit_fi_r27c.py](audit_fi_r27c.py), the vendored `pcie_ipc_ar21` extension, and the CPU tests in [test_cpu.py](test_cpu.py).
2. `vllm-qwen38:v0310-r27c-A_S-splitkv` from [Dockerfile.A_S-splitkv](Dockerfile.A_S-splitkv): one Python edit that makes patch 0136's split-KV override accept FlashInfer 0.7's `(kv_data_type, device)` guard signature. It asserts the hash of the file it edits, so it applies only to an image built from the 0136 diff as it was before that change.
3. `vllm-qwen38:v0310-r27c-A_S-L1a-ubplan` from [Dockerfile.A_S-L1a-ubplan](Dockerfile.A_S-L1a-ubplan): the A_S image plus [L1a-draft-ub-plan.diff](L1a-draft-ub-plan.diff), a Python layer that lets the MTP drafter plan pure draft decode from the CPU sequence-length upper bound and orders consecutive plans' staging copies with CUDA events. It is inactive unless `VLLM_SM12X_DRAFT_UB_PLAN=1` ([r27-l1a](../bench/results/r27-l1a.md), 2026-10-17, results `2026-10-16-r27-audition-qualify-231459-865333`, `2026-10-17-r27-audition-screen-022747-4123606`).

The [0136 diff](0136-opt-in-nvfp4-prefill-split-kv-v0310.diff) in this directory already carries the signature change (with a comment in place of the layer's inline comment). For A_S behavior an image built from this directory therefore needs only the first build; the second build would refuse it at its hash check. The L1a build also pins the exact served A_S source, so the public chain's different comment does not satisfy its hash check. The served L1a image uses the original A_S base (sha256:2a081946bc9c0b6694cb246391a91ba19995d7dd55245afabdb282f97cf1808e, built 2026-10-16 04:27 UTC); its source pins are retained.

[PORT-STATUS.md](PORT-STATUS.md) lists every patch of the v0.29 chain and what it became on v0.31.0: regenerated, dropped, or found upstream (0152, 0155, 0156 and 0158 are upstream in v0.31.0). [assert_sources.py](assert_sources.py) checks the pristine source hashes in [pristine-hashes.json](pristine-hashes.json) before anything is applied.

## Build context

`Dockerfile.v0310` copies from a build context laid out as follows; the paths are the Dockerfile's.

| context path | contents |
|---|---|
| `out/A/patches-v0310/` | this directory's `series`, the `*.diff` files, `overlay/` and `pcie_ipc_ar21/` |
| `out/A/apply_layer.py`, `guard_flashinfer_workspace.py`, `assert_sources.py`, `pristine-hashes.json` | the files of the same names here |
| `out/audit_fi_r27c.py`, `out/test_cpu.py` | the files of the same names here |
| `vllm-v0.31.0/csrc/`, `vllm-v0.31.0/cmake/patches/` | from the vLLM v0.31.0 source tag |
| `assets/wheels/` | an `arctic_inference-0.1.1` wheel built in the base image (the package ships only an sdist) |
| `assets/debs/` | `.deb` files for `patch`, `g++` and `ninja-build`, needed only if the base image lacks them |

```bash
docker build --network=none --pull=false --build-arg VLLM_BASE=vllm/vllm-openai:v0.31.0 \
  --build-arg CUDA_TOOLCHAIN=nvidia/cuda:13.0.3-devel-ubuntu24.04 \
  -f <context>/out/A/Dockerfile.v0310 -t vllm-qwen38:v0310-r27c-A-nvfp4kv-mtp <context>
docker build --network=none --pull=false -t vllm-qwen38:v0310-r27c-A_S-splitkv -f Dockerfile.A_S-splitkv .
```

The serving host ran these builds from that context with the legacy builder (`DOCKER_BUILDKIT=0`). A build from a clean clone of this repository has not been run for this chain; the v0.29 chain's was reproduced on 2026-09-26 ([R738](../bench/results/r738-build-verify.md)).

## L1a requires the served A_S source hashes

With the exact A_S image already local, run the third build from the repository root; its context is the public `patches-v0310/` directory:

```bash
docker build --network=none --pull=false -t vllm-qwen38:v0310-r27c-A_S-L1a-ubplan \
  -f patches-v0310/Dockerfile.A_S-L1a-ubplan patches-v0310
```

The recipe copies only `L1a-draft-ub-plan.diff`, verifies the diff and both input-file hashes, applies with `patch --fuzz=0`, verifies both output hashes, checks the helper and admission-log strings, and compiles the edited Python files. It also tries a GPU-free import and env-off helper check. An import failure prints a warning and continues; the hash checks, grep assertions and Python compilation are mandatory, and the ACTIVE boot line checks the import on the serving route. This round mirrors the supplied recipe and measured image; it does not rebuild the image or relax the source pins.

## Running it

[scripts/serve-v031-l1a-daily.sh](../scripts/serve-v031-l1a-daily.sh) is the serving host's launcher and [scripts/serve.sh](../scripts/serve.sh) the same engine with the host paths as settings. Both set `VLLM_SM12X_DRAFT_UB_PLAN=1`, `VLLM_SM12X_NVFP4_PREFILL_SPLIT_KV=1` and `VLLM_USE_V2_MODEL_RUNNER=1`, pass `--mamba-backend triton --gdn-prefill-backend flashinfer`, and check after boot the markers 0136, 0147, 0148 and 0154 and that `0158` is listed in the upstream-source marker file. L1a additionally requires the ACTIVE and USED draft-plan lines and rejects INACTIVE. Measurements: [bench/results/r27-l1a.md](../bench/results/r27-l1a.md). The previous [scripts/serve-v031-splitkv-daily.sh](../scripts/serve-v031-splitkv-daily.sh) and its [A_S measurements](../bench/results/r27-v031-splitkv.md) remain available. Provenance: [THIRD_PARTY.md](../THIRD_PARTY.md).
