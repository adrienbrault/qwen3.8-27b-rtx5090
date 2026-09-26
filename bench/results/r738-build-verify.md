# R738: a no-cache rebuild of the published chain reproduces the served image

Date: 2026-09-26, 01:58 to 02:07 UTC. Results `2026-09-26-r738-build-verify` ([raw](2026-09-26-r738-build-verify/)), unit [unit.sh](2026-09-26-r738-build-verify/unit.sh).

## What was run

The repository at commit `05c64b3`, exported with `git archive` into an empty directory (committed files only), built with `NO_CACHE=1 IMAGE_REPO=vllm-qwen38-verify bash scripts/build-served-image.sh`. That is the nine layers listed under "Engine" in the README, every layer with `--no-cache`, tagged beside the served images rather than over them. The unit then wrote a manifest of the rebuilt image and of the image the daily serves (`vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc-bsshash-mtppcie-mtpcache-eagleshift`, the `DAILY_IMG` of [scripts/serve-r231-nvidia-daily.sh](../../scripts/serve-r231-nvidia-daily.sh)) and compared the two.

Host: Ryzen 7 9800X3D, the build niced beside a running serving engine. The pinned base image (`vllm/vllm-openai@sha256:383e409f…`) was already local, so neither the time nor the disk figure below includes its 8.65 GB pull.

## Result

| | |
|---|---|
| Build | rc 0, every layer rebuilt (no `CACHED` step in any layer log) |
| Identity check | `IDENTITY OK vllm 0.29.0rc2 torch 2.13.0+cu130 flashinfer 0.6.16.post3` |
| Manifest, served vs rebuilt | 10,197 lines each, diff empty, same sha256 ([manifest-sha256.txt](2026-09-26-r738-build-verify/manifest-sha256.txt); the manifests themselves stay on the serving host) |
| Image IDs | served `496fe1c83492`, rebuilt `4e14e23b10df` (distinct images, same content in the compared scope) |
| Wall time | 9 min 2 s: first layer 4 min 37 s (142 s of it the vLLM source clone), FlashInfer swap 3 min 35 s, the other seven layers about 40 s together |
| Disk | +28.06 GB on the root filesystem (Docker's image and build-cache totals agree) |

## What "identical" covers

The manifest hashes the 9,926 non-binary files under `vllm/`, `flashinfer/`, `arctic_inference/`, `pcie_ipc_ar21/`, `/opt/prs-markers` and `/opt/vllm-sm12x`, and lists the versions of all 271 installed Python distributions. It does not compare compiled files (`.so`, `.cubin`, `.pyc`), the image config (environment variables such as `PYTHONPATH=/opt/vllm-sm12x`), the source tree in `/opt/vllm-src`, the contents of torch or other packages, or system files.

The build is reproduced, not hermetic: `arctic-inference==0.1.1` is installed with its dependencies resolved at build time, the apt packages are unpinned, and the vLLM source is cloned by tag. All of these resolved to the served versions on 2026-09-26.

## Disk to plan for

On a host without the base image: 28 GB for the build, plus about 39 GB for the base under the containerd image store (8.65 GB of compressed blobs and 30.5 GB unpacked), about 67 GB. The README and the script header say 70 GB.

## Fix found by this run

With `DOCKER="sudo docker"`, `NO_CACHE=1` placed `--no-cache` before `build` (`sudo docker --no-cache build`), which fails. R738 ran with the default `DOCKER=docker` and was not affected. The flag now goes after `build` whatever `DOCKER` expands to.
