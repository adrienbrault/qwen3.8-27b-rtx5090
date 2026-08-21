# DFlash2 on the 5090 (Qwen3.8-27B saka W4A4 + incoai/Qwen3.8-27B-DFlash2)

User ask 2026-08-21: try [DFlash2](https://inco.ai/blog/dflash2/) instead of MTP. Facts gathered before building:

- **What**: DFlash (block-parallel drafter, 5 decoder layers tapping target layers 5/19/33/47/61, SWA-2048 attention, `is_causal: false`) + a 2M-param candidate **selector** (top-16 per slot, walks the best path) + **two-tap depthwise convs** (16.5M). Blog: 2.7–3.4× vs AR on H200 at T=1.0/xhigh, mean acceptance 4.80 (GSM8K); DSpark 4.27.
- **vLLM**: [PR #52816](https://github.com/vllm-project/vllm/pull/52816), merged 2026-08-21 05:27 UTC — not in any release, not in the 08-21 nightly (cut 03:46). Python/Triton only → `0001-pr52816-dflash2.diff` (vllm/ files) on the `ba07e4a48` nightly. **Forces the V2 model runner** (`use_v2_model_runner` → DFlash2 draft); V2 has kv_connector, multimodal, Mamba-align support on main.
- **Drafter**: `incoai/Qwen3.8-27B-DFlash2` (= `z-lab/Qwen3.8-27B-DFlash2`), 3.85 GB bf16, `DFlash2DraftModel`, `block_size 8` → **`num_speculative_tokens` must be 7**, `method` is **`dflash`** (the 2 comes from the draft architecture). On flan: `/srv/qwen5090/models/dflash2-qwen38-incoai`.
- **Constraints**: `--async-scheduling` raises with dflash (we already run async-off); greedy is unsupported by the selector (T=0.6 override is fine); the target LM head must be readable for top-k — merged PR does it through `LogitsProcessor.get_top_k_tokens`, so saka's unquantized `lm_head` (in the compressed-tensors `ignore` list) is fine.
- **Prior on this box** (FINDINGS 2026-07-13, DFlash v1 on a source build): 2.0× single-stream but bf16-KV only (21K ctx), no batch scaling, 0.8× prefill. Community sm120 datapoint (dfischermittwald, RTX PRO 6000, T=1.0): NVFP4 target + DFlash2 n=7 **110 t/s flat to 32K** vs MTP n=3 83 vs AR 51; FP8 target + DFlash2 collapses to 50 at 32K.
- **Cost**: +3.85 GB draft weights (≈ −40K fp8-KV tokens of pool at util 0.95) vs the 0.8 GB MTP head.

Gauntlet (same order as nvfp4kv): boot facts (V2 runner line, pool, spec method, acceptance) → needles cold/warm → sean gate → killer/vision/SO → tool-eval 69×2 → benchy ladder + deep pp30K vs MTP daily → verdict.
