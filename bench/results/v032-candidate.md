# vLLM 0.32.0 candidate CPU preparation

Prepared 2026-10-21, results [`2026-10-21-v032-cpu`](2026-10-21-v032-cpu/). This records source/build validation, not an inference benchmark. The served 0.31.0 configuration remains unchanged. No GPU startup, inference, fidelity or performance result is claimed for this candidate.

The [candidate package](../../patches-v0320/README.md) ports the retained daily features onto vLLM commit `5df63a7203cdb9328d824a27b682e77c7b9928be`, using native NVFP4 writing and SM120 XQA. The official Linux amd64 image is pinned by registry manifest digest; every changed source file and patch is hash-checked. The [port table](../../patches-v0320/PORT-STATUS.md) accounts for the former series and the [notice](../../patches-v0320/NOTICE.md) records attribution.

Local CPU work includes a fresh-source reconstruction and behavior checks on Linux aarch64 with CPU torch, the filesystem eviction tests against patched source in the official Linux amd64 image, and image builds with networking disabled. The fixed PCIe extension compiles for SM120a with nvcc and loads without a GPU. The CLI schema check uses the actual 0.32 parser with CPU-only SM120 metadata; this is not evidence of a successful GPU boot. The recorded checks and image IDs are under the results directory.

The native control and BA4 integration images have separate candidate tags. BA4 remains disabled, uses a new serving-evidence protocol and binds to the newly built control image ID. Old 0.31 reports cannot authorize the new engine. Future GPU qualification must measure native writing/XQA correctness, the pool and headroom at the same byte pin, TP2 collectives and the long-context multi-agent workload, followed by fresh BA4 admission if that layer is used.
