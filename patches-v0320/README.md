# vLLM 0.32.0 candidate for Qwen3.8-27B

Prepared on 2026-10-21. This directory contains a source port and local CPU builds, not a promoted serving configuration. The served 0.31.0 launcher remains unchanged. The first GPU screen boots successfully and measures the same token pool. Numerical qualification and BA4 admission remain incomplete; [dated results](https://github.com/adrienbrault/qwen3.8-27b-rtx5090/blob/main/bench/results/v032-gpu-screen.md) describe performance.

The base is `vllm/vllm-openai:v0.32.0`, pinned to its Linux amd64 manifest digest in `manifest.json`. The image's installed versions were checked locally: vLLM 0.32.0, torch 2.13.0+cu130, FlashInfer 0.7.0.post1 and Triton 3.7.1. The native NVFP4 writer and SM120 XQA verification replace the older writer/FA2 graph overlays. `PORT-STATUS.md` accounts for every previous patch.

## Build without GPU access

Run `bash build.sh` on a Linux amd64 Docker host, or an amd64-capable Docker installation. It pulls only the pinned base when missing, builds with networking disabled and compiles the fixed-tactic PCIe extension for SM120a without a visible GPU. The base already supplies nvcc, g++, ninja and patch; no package installation or vLLM source rebuild is performed. It creates two candidate-only tags:

- `vllm-qwen38:v0320-native-r1`: native attention plus retained runtime, MTP, memory and offload features.
- `vllm-qwen38:v0320-native-ba4-r1`: the same control plus BA4 integration, disabled by default.

The installer checks every patch checksum, input hash and output hash, and compiles all changed Python files. BA4 binds its evidence contract to the freshly built control image ID and uses a new serving protocol. Old 0.31 reports cannot authorize it. Fresh diagnostics and serving-profile evidence are required before enabling BA4; the generic candidate command leaves it disabled even when selecting the integration image. Build IDs can differ between Docker image stores, so build the pair in the image store used for their qualification.

## CPU validation

Download the pinned upstream archive and check its SHA256 before use:

```sh
mkdir -p out
curl --fail --location https://codeload.github.com/vllm-project/vllm/legacy.tar.gz/5df63a7203cdb9328d824a27b682e77c7b9928be -o out/source.tar.gz
python3 test_port.py --source-archive out/source.tar.gz
```

The expected archive SHA256 is `33247d5def035a78ccc8b2e5499a429ab82358e426c6aa0fe91cb6c60bab9c3a`. The tests require CPU torch. They reconstruct a fresh tree, apply both image layers, check rejection of repeated installation and invalid BA4 control IDs, and exercise the changed MTP, GDN, autotune, capture and evidence behavior. They do not execute attention kernels or validate GPU numerical output. The carried filesystem eviction tests also run against the patched source using pytest; their transfer jobs use 0.32's `chunk_ids` field. `check_cli.py`, run inside the built image, validates the actual parser using CPU-only SM120 metadata; it does not validate hardware startup.

## GPU screen and deferred qualification

`python3 plan.py` prints a proposed container command and does nothing else. It neither inspects nor modifies the host. `--ba4-image` selects the integration image with BA4 still disabled. The command uses loopback port 18029, a candidate-only model alias, a read-only checkpoint mount and a fresh child of `v032-candidate` for both tier and compile-cache state on every generation. This prevents different checkpoints or rebuilt images from sharing cached state merely because they all mount at `/model`. Filesystem capacity accounting and pins require one manager per tier root; do not share these directories with another server. Generate a new command for each boot; do not reuse a printed command after changing its checkpoint or image. It contains no daily stop, cleanup, power/clock change, queue registration or service installation. An authorized GPU-exclusive unit must own its lifecycle before this command is run; another port does not provide GPU isolation.

Before any future run, confirm an authorized idle window, acquire the shared GPU queue/lock through the normal unit protocol, and record the selected daily as the restoration target. Use the existing screen/qualify instruments rather than building another gate framework. The proposed configuration is TP2, MTP3, no EAGLE block drop, 16 sequences, 262144 context, NVFP4 KV pinned at 15900000000 bytes per card, bf16 recurrent state, embedding UVA and bounded CPU/filesystem KV tiers. The token pool and headroom must be measured on this image; they are not inherited from 0.31.

The first contact should verify healthy boot, native writer/XQA selection, TP2 target and draft collectives, embedding sharing/offload, parser/tool/image smoke, cached versus cold continuation, and error/Xid logs. Then perform a screen of at most one hour against the current daily, with equal power limits and the same checkpoint: short/deep decode, prefill and a six-agent deep-context workload with staggered tool results. Record per-request TTFT, inter-token gaps and completion time alongside throughput. Native NVFP4 writing changes numerics: run the standard qualify, bf16 fidelity ladder and long-generation decode comparison before any promotion. Re-admit BA4 separately using fresh hardware/source/image-bound reports. No GPU screen is queued by this package.

Optional `--dynamic-mtp` adds a batch-size schedule (K3 at 1–2, K2 at 3–4, K1 at 5–16) to the printed command; it remains disabled by default. The 2026-10-21 GPU screen verified the effective depths but measured 21–33% lower concurrent decode and a 5.0% longer agent replay. [Dynamic MTP results](https://github.com/adrienbrault/qwen3.8-27b-rtx5090/blob/main/bench/results/v032-dynamic-mtp.md) record the rejected policy.
