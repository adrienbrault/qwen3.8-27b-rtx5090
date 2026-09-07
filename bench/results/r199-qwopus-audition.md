# R199: Qwopus3.8-27B-Flash audition, stopped early (2026-09-05, `results/2026-09-05-r199-qwopus`)

[← all results](../RESULTS.md)

Three checkpoints of the Jackrong fine-tune were fetched (`scripts/r199-fetch-qwopus.sh`): bf16, Shiftedx NVFP4-MTP (ModelOpt NVFP4 W4A4 on every linear, GDN projections included), sojufx NVFP4 (ModelOpt mixed: FP8 attention and GDN, MLP W4A16 NVFP4 on the Marlin path, bf16 lm_head). The Qwopus tokenizer is byte-identical to Qwen3.8's, so the position-aligned fidelity rulers transfer across the two models. `scripts/r199-qwopus-audition.sh` smoke-boots the quants first, then generates Qwopus bf16 references by the R156 method, then runs one battery per quant. The user stopped the run during the bf16 dense dump, so no fidelity or tool-eval number exists.

What was measured: sojufx boots on the daily route (nvfp4 KV, DFlash2 ns7, pcie_ipc, batch-sharded sampling) at the 13.98 GB pin with a 1,052,277-token pool; code c1 decode 268.4 tok/s at 0.404 accepted tokens per draft, against 274.2 at 0.409 for the RedHat checkpoint on the same route (R196) and 278.9 at 0.415 for the daily (R197). The DFlash2 drafter trained on the base target fits the fine-tune equally well.

Two loader fixes came out of it: the daily launcher's checkpoint-identity check now recognizes ModelOpt candidates, and `scripts/graft_vision.py` rebuilds a text-only ModelOpt export's vision tower from the bf16 source with hard links plus one extra shard, because vLLM's Qwen3.5 wrapper builds the vision tower under the checkpoint's quantization and Shiftedx ships neither the vision weights nor a `model.visual*` exclusion.
