# BRIEF28 — DFlash drafter recalibration / distillation pipeline against the RedHat NVFP4 target (scripts, no patch)
Read COMMON.md first. Implements R186-ANALYSIS.md §3 "drafter acceptance" (#8): the served drafter is syvai's DFlash2 Qwen3.8-27B
drafter, W4A16 pack-quantized (evidence/drafter-config.json: 5 layers, hidden 5120, intermediate 17408, 32/8 heads, head_dim 128,
target_layer_ids [5,19,33,47,61], block_size 8, mask token 248070, vocab 248320), trained by its author against the bf16 target;
the served target is RedHatAI's NVFP4 W4A4 quantization of that model, and the drafter's acceptance rate on it is lower than on
bf16 (R186: +0.02 absolute acceptance ≈ +4.3% code / +7.7% prose decode speed). The drafter reads the target's hidden states at
layers [5,19,33,47,61] (DFlash design: `src/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` — read how the features are
gathered and fed, and which drafter modules exist: `src/vllm/model_executor/models/` grep `dflash`/`DFlash`) and predicts the
next block of 8 tokens with a mask token. Hardware for any training: the same box — 2× RTX 5090 32 GB, 64 GB host RAM, no
other GPU; the bf16 target (27B × 2 B = 54 GB) does NOT fit alongside anything. NVFP4 target ≈ 14 GB per rank pair (TP2).

Deliver (scripts run inside the served container or a sibling `vllm`/`llmcompressor` container — say which, and list the pip
packages needed with versions):
PHASE A — runnable now, 1 GPU: recalibrate the drafter's W4A16 quantization against activations produced on the NVFP4 target.
- `deliver/build_calib_corpus.py`: builds a calibration set (≈512–2048 sequences, 1–4K tokens) that matches the served traffic
  shape: code (repository files + diffs), tool-call transcripts (JSON tool calls/results, the Qwen chat template with tools), and
  prose, sampled from LOCAL text sources the operator provides (argument: directories; no network) and from the served endpoint's
  own generations (argument: an OpenAI-compatible URL; optional). Emit a JSONL with the chat-templated text.
- `deliver/recalibrate_drafter.py`: takes the bf16 drafter (argument: HF dir of the bf16 DFlash2 drafter, e.g. the incoai/syvai
  upstream — the operator has or can fetch it), the NVFP4 target served via vLLM OR loaded in transformers with the
  compressed-tensors loader on one GPU (say which is feasible in 32 GB: the target alone is ~14 GB NVFP4... but does transformers
  run NVFP4 on SM120? if not, the hidden states must come from the vLLM server: check whether vLLM can return hidden states at
  chosen layers — it cannot via the API; so the realistic route is: run the target in transformers with compressed-tensors
  dequantized to bf16 layer-by-layer on 1 GPU with offload, OR use a vLLM plugin hook. Investigate and pick), captures the
  target hidden states at [5,19,33,47,61] for the corpus, runs the drafter forward on them, and re-quantizes the drafter to
  W4A16 (compressed-tensors pack-quantized, group 128? read the drafter config for the exact scheme) with llm-compressor using
  those activations as calibration (GPTQ/AWQ per the scheme). Output: a drafter checkpoint dir in the SAME format as the served
  one (so evidence/launch-daily.sh's `--speculative-config` accepts it unchanged; read the launcher for the model path/format).
- `deliver/measure_acceptance.py`: runs against the served endpoint (:8029 in experiments) and reports mean accepted tokens per
  step per content kind from the vLLM metrics (`vllm:spec_decode_num_accepted_tokens_total` etc. — grep the metric names in
  `src/vllm/v1/metrics/`) on the same 3-kind corpus the operator's decode probe uses (evidence/decode_ss.py, kinds code/prose/tool).
PHASE B — DESIGN ONLY (NOTES28.md, with a VRAM ledger, no training loop): distilling/fine-tuning the drafter (bf16, ~1.5B?
compute its parameter count from the config) against the NVFP4 target's hidden states and next-token distributions. Ledger:
target NVFP4 inference on 1 GPU (weights + activations at 4K ctx), drafter training on the other GPU (params + grads + Adam
states + activations at batch B), host RAM for cached hidden states (5 layers × 5120 × bf16 × tokens = 51 KB/token → 100M tokens
= 5 TB, so: streamed, not cached; what token budget is realistic at ~X tok/s of teacher forward?). State whether NVFP4-target
hidden states are an acceptable teacher (that IS the deployment condition, so arguably better than bf16), what the original
DFlash recipe used (cite the DFlash paper/repo if you know it; do not invent), the expected acceptance gain range, and the
go/no-go criterion. Do not write a training loop.
Verify phase A scripts with py_compile + a `--dry-run` mode that exercises argument parsing and corpus building on a tiny
inline sample without torch. State all unverifiable assumptions (package APIs) explicitly.
