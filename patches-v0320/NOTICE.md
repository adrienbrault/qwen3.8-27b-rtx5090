# Sources and licence

The base vLLM source is Apache-2.0, pinned to commit `5df63a7203cdb9328d824a27b682e77c7b9928be` ([v0.32.0](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)). The image digest, archive checksum and per-file patch hashes are in `manifest.json`.

The source patches are ports of the Apache-2.0-derived v0.31.0 recipe series in [adrienbrault/qwen3.8-27b-rtx5090](https://github.com/adrienbrault/qwen3.8-27b-rtx5090/tree/53fe278/patches-v0310). Their original authors, upstream PRs and design attribution remain recorded in that recipe's [THIRD_PARTY.md](https://github.com/adrienbrault/qwen3.8-27b-rtx5090/blob/53fe278/THIRD_PARTY.md). The new grouping combines 0108/0133, 0135/0137/0154, 0136/0147/0160/0162, 0138/0148 and the BA4 layer; it does not claim new upstream authorship.

The vendored PCIe package retains its own `pcie_ipc_ar21/NOTICE`: its FlashInfer header is from commit `df8b5c1`, Apache-2.0; the ownership adapter, workspace and fixed tuning data are from the pinned recipe above. The BA4 implementation is carried from that recipe, with the new engine's serving evidence protocol and control-image binding.

Native attention capabilities are supplied by vLLM [#46963](https://github.com/vllm-project/vllm/pull/46963) and [#60452](https://github.com/vllm-project/vllm/pull/60452). The port preserves the native zero-draft and pure-spec GDN handling in [#56531](https://github.com/vllm-project/vllm/pull/56531) and [#58763](https://github.com/vllm-project/vllm/pull/58763), and per-rank autotune-cache behavior in [#57635](https://github.com/vllm-project/vllm/pull/57635). This package contains no model weights or private agent transcripts.
