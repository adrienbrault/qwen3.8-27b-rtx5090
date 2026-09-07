# PROMOTED: the daily is now the v0.28 generation (2026-08-28, `results/2026-08-28-r108-promote`)

[← all results](../RESULTS.md)

`serve` for this stack is now vLLM v0.28.0 plus `patches-v0280/`: nvfp4 KV, XQA decode, MTP ns=4, async scheduling and the native OffloadingConnector disk tier, replacing the 0.26-nightly plus LMCache generation. The tier's backing store is a fixed-size 200G loopback ext4 image. After LMCache's unenforced-cap incident (876G disk-fill, July), the cap is enforced by construction rather than trusted to eviction code.

Promotion evidence (tool-eval, ×4 trials each, same day, same harness): previous daily 90.0 ± 1.2, new engine without tier 90.0 ± 1.8 for quality parity, and with tier 88.2 ± 1.0. The roughly 1.8-point delta is attributable entirely to the disk tier's write traffic during agentic bursts (responsiveness subscore 63 vs 80, a wall-clock effect rather than fidelity), and tier tuning remains open. Final promotion checks on the promoted config: needles correct through cold, 8-flood eviction, divergent-suffix and container restart (fs tier persists), decode code c1 205.9 / c8 1103 aggregate, and tier at 41G/200G.

Operational findings: the OffloadingConnector's 4G CPU-staging mmap (`/dev/shm/vllm_offload_*.mmap`) leaks past `docker rm -f`, and four engine swaps consumed 16G of host RAM until a fuser-guarded sweep went into the launcher. Boot asserts that grep engine logs must not use `grep -q` under `set -o pipefail`, because early-exit SIGPIPE fails the pipeline on a successful match. The launcher fails closed on overlay-ACTIVE, `decode_backend=xqa`, connector init, pool band, and a MemAvailable gate before every engine swap.
