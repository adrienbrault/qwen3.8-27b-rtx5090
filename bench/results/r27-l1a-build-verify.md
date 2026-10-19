# r27 L1a build verify: a clean-clone build of the served v0.31 image matches it, and the image is published

Date: 2026-10-19, 02:43 to 03:05 UTC. Results `2026-10-19-r27-l1a-build-verify` ([raw](2026-10-19-r27-l1a-build-verify/)), runners [cleanroom.sh](2026-10-19-r27-l1a-build-verify/cleanroom.sh) and [functional-check.sh](2026-10-19-r27-l1a-build-verify/functional-check.sh).

## What was run

The checkout with [scripts/build-v031-image.sh](../../scripts/build-v031-image.sh), packed into a tarball and unpacked into an empty directory, built by `NO_CACHE=1 bash scripts/build-v031-image.sh` inside a `docker:27-dind` container with its own empty image storage. The build saw only the checkout, the two pinned base images pulled by digest, the vLLM source archive and the PyPI files the script pins. The serving host's own images and build cache were not visible to it.

The rebuilt image was then exported under a separate tag, checked with the script's identity probe on the host's NVIDIA runtime, booted through [scripts/serve.sh](../../scripts/serve.sh), and scanned with the layer audit used for the earlier published tags.

## Result

| | |
|---|---|
| Build | rc 0, all three builds with `--no-cache` (35, 2 and 3 steps, no `Using cache` line in any log); the NVFP4-KV overlay was recompiled with nvcc |
| Base images | `vllm/vllm-openai@sha256:c1c9f6fd…` and `nvidia/cuda@sha256:b7ae301d…` verified through `RepoDigests` |
| vLLM source | archive of commit `db9527a4…` (tag v0.31.0), SHA-256 `e20baf18…`; its `csrc/` and `cmake/patches/` are identical to a `git clone` of the tag |
| Content, served vs rebuilt | inventory 7,138 files each, none missing or extra; 7,112 of 7,112 sources (`.py`, `.json`, `.cu`, `.cuh`) identical; 19 of 19 shared objects identical, including the recompiled `vllm_sm12x_nvfp4kv.so`; the two Ninja logs differ ([content-report.txt](2026-10-19-r27-l1a-build-verify/content-report.txt), [manifest-sha256.txt](2026-10-19-r27-l1a-build-verify/manifest-sha256.txt)) |
| Identity probe | `IDENTITY OK 0.31.0 2.13.0+cu130 0.7.0.post1` |
| Boot through serve.sh | the served and the rebuilt image each boot, pass the script's boot-log and patch-marker checks, and report a KV pool of 1,391,795 tokens |
| Greedy output | 3 of 5 prompts identical across all three boots (served, rebuilt, served); on the other 2 the two boots of the served image already differ from each other within the first tokens, so greedy output does not separate the images on those prompts ([greedy-hashes.txt](2026-10-19-r27-l1a-build-verify/greedy-hashes.txt)) |
| Layer audit | no credentials or private files; every hit is in an upstream base layer and of the classes the 2026-09-26 audit listed (CA bundles, key-format constants, documentation example keys, the public Hugging Face test token in `transformers/testing_utils.py`, a token-shaped hash in a `RECORD` file) |
| Wall time | about 2 min for the three builds; the bases had been pulled into the build container's storage by the earlier runs below, so their pull is not included |
| Disk | 39 GB of image storage inside the build container, bases included |

The arctic-inference wheel is rebuilt from the pinned sdist and is 1 byte larger than the served one (244,345 against 244,344 bytes); its package is outside the manifest's scope. The vLLM shared objects come unchanged from the base image; the overlay `.so` is the only compiled file the chain produces.

## Fixes found by the clean-room runs

Three earlier runs on the same day failed before this one, each on a step the serving host's build had satisfied from local state:

1. The base pins were compared with `.Id`. On the containerd image store `.Id` is the registry digest; on Docker's classic store it is the config digest, so a stock install refused the correct base. The script now checks `RepoDigests` and builds from the resolved local ID.
2. The arctic-inference sdist needs CMake, Ninja and nanobind to build, and the vLLM base has none of them. The wheel is now built in a throwaway container with hash-pinned build tools ([arctic-build-requirements.txt](../../scripts/arctic-build-requirements.txt)); the tools do not reach the image.
3. A tarball packed on macOS carried AppleDouble `._*` files into the overlay, which the inventory check reported as 4 extra files. A git checkout does not contain them.

## Published image

`ghcr.io/adrienbrault/qwen3.8-27b-rtx5090:v0310-r27c-A_S-L1a-ubplan`, digest `sha256:40b61ed95254de5201cb69a8a02de9da0569fc4ed759cd843d25b9b7b3d73f2c`, is the rebuilt image from this run, pushed 2026-10-19. Anonymous pulls of the manifest return 200.
