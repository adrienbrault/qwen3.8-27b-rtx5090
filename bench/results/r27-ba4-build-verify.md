# r27 BA4 build verify: a clean-clone build of the served BA4 image matches it except for 3 bytes of a temporary-file name in the recompiled NVFP4-KV overlay (2026-10-20, results `2026-10-20-r27-ba4-build-verify`)

[← all results](../RESULTS.md)

Date: 2026-10-20, 11:37 to 12:08 UTC. Results `2026-10-20-r27-ba4-build-verify` ([raw](2026-10-20-r27-ba4-build-verify/)), runner [cleanroom.sh](2026-10-20-r27-ba4-build-verify/cleanroom.sh); on the serving host `public-27b-cleanroom-20261020T113727Z`, `public-27b-cleanroom-20261020T114533Z`, `public-27b-cleanroom-20261020T115450Z` and `2026-10-20-ghcr-leak-audit-1204`.

## What was run

`git archive` of the checkout at commit 59eed92, unpacked into an empty directory and built by `NO_CACHE=1 BA4=1 bash scripts/build-v031-image.sh` with the default `VLLM_SOURCE_SHA256` inside a `docker:27-dind` container with its own empty image storage, capped at 6 CPUs. The build saw only the checkout, the two pinned base images pulled by digest, the vLLM source archive and the PyPI files the script pins.

The rebuilt BA4 image was exported under a separate tag. Containers on the `runc` runtime, with no network and no GPU device, then listed the contents of it and of the BA4 image the serving host serves (`vllm-qwen38:v0310-r27c-A_S-BA4-L1a-ubplan`, sha256:1df9b4c4e1d2) for the comparison: the build's manifest tool over vLLM, FlashInfer and the overlay, and the SHA-256 of every file of the `vllm_ba4` package and of `/opt/l1a-ba4` except bytecode. The layer audit used for the earlier published tags then scanned every layer of the rebuilt image.

The serving host's GPUs were serving the 27B configuration during the run. The script's identity probe (`CHECK=1`, which loads the compiled ops with the NVIDIA container runtime) and boots through [scripts/serve.sh](../../scripts/serve.sh) were not run.

## Result

| | |
|---|---|
| Build | rc 0, all four builds with `--no-cache` and no `Using cache` line; inside the BA4 build the layer's installer verified every pinned input and output hash and its CPU tests ran 60 tests, all passing, in 116.7 s; 11:45:33 to 11:53:58 UTC, base pulls into the empty storage included |
| L1a stage against the served L1a inventory | inventory 7,138 files each, none missing or extra; 7,112 of 7,112 sources identical; 18 of 19 shared objects identical ([l1a-content-report.txt](2026-10-20-r27-ba4-build-verify/l1a-content-report.txt)); the script's own BA4 check against the derived inventory reads the same ([ba4-content-report.txt](2026-10-20-r27-ba4-build-verify/ba4-content-report.txt)) |
| BA4 image against the served BA4 image | the same inventory result ([served-vs-rebuilt-report.txt](2026-10-20-r27-ba4-build-verify/served-vs-rebuilt-report.txt)); the inventory the script derives for BA4 from the served L1a inventory and the layer's pinned outputs equals the served BA4 image's line for line ([manifest-sha256.txt](2026-10-20-r27-ba4-build-verify/manifest-sha256.txt)); all 69 files of `vllm_ba4` and `/opt/l1a-ba4` identical ([ba4-files-sha256.txt](2026-10-20-r27-ba4-build-verify/ba4-files-sha256.txt)) |
| The one differing shared object | `vllm-sm12x/build/vllm_sm12x_nvfp4kv.so`, the overlay that nvcc recompiles, differs from the served one in 3 bytes: the name of nvcc's temporary file, which carries a process-ID field and appears in a static-initializer symbol, reads `tmpxft_00000033` in the served build and `tmpxft_00000032` in this one. Its object file and the two Ninja logs differ with it. The 2026-10-19 clean-clone build matched the served `.so` byte for byte ([r27-l1a-build-verify](r27-l1a-build-verify.md)); a third run on 2026-10-20 at 11:54 UTC produced the same `.so` as this one |
| Layer audit | 281 hit lines, the same set as the 2026-10-19 audit of the rebuilt L1a image; every hit lies in the first 30 of the 40 layers of the `vllm/vllm-openai` base, of the classes listed there (CA bundles, key-format constants, documentation example keys); no credentials or private files (results `2026-10-20-ghcr-leak-audit-1204`) |
| Identity probe, boot | not run |

## Fix found by the clean-room run

The first run (11:37 UTC, the checkout at 2f794bd) failed in the BA4 build: the layer's logging tests read three boot-log fixtures, `fixtures/logging/run2-ba4.log`, `run3-ba4.log` and `run5-dynamo-echo.log`, that the repository's `*.log` ignore rule had kept out of every commit, so a clone lacks them (`FileNotFoundError`, 8 failures and 1 error). Commit 59eed92 tracks them, and an offline test in [scripts/tests/test_build_v031.py](../../scripts/tests/test_build_v031.py) fails when git ignores any file under `patches-v0310/` or `scripts/`.

## The image

The rebuilt image is tagged `ghcr.io/adrienbrault/qwen3.8-27b-rtx5090:v0310-r27c-A_S-BA4-L1a-ubplan` on the serving host (local ID sha256:a003a1822c03) and is not published. Without admission reports its layer stays off and it computes as the L1a image does ([r27-ba4](r27-ba4.md#reproducing-it)).
