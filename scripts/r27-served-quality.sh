#!/usr/bin/env bash
# r27-served-quality (2026-10-20): the 2026-10-19 BA4 quality line (tool-eval 91.0 +- 1.4, GSM8K 0.905) was
# measured before no-block-drop and the 15.90 GB pin; re-run both against the LIVE served daily on :8020 (no boot, no GPU
# lock: the daily keeps serving, so other requests may share the GPUs during the run).
#   tool-eval: the invocation of the 2026-10-19 BA4 gate (69 scenarios x 4 trials, T 0.6, top-p 0.95, top-k 20, parallel 8).
#   GSM8K: the think-off recipe of the 2026-10-19 BA4 gate (5-shot, chat template, T 0, n 200, num_concurrent 4). that gate set
#     thinking off with the server flag --default-chat-template-kwargs; the live daily keeps thinking on, so the same
#     kwargs ride on each request (lm_eval 0.4.12 parses JSON --gen_kwargs and spreads extra keys into the payload).
#   LAUNCHER=<served launcher> RESULTS=<results dir> MODEL_DIR=<weights> bash scripts/r27-served-quality.sh   (TOOL_ONLY=1: tool-eval only)
set -uo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
R=$RESULTS/$(date -u +%F-%H%M%S)-r27-served-quality; mkdir -p "$R"
U=http://127.0.0.1:8020
log(){ echo "$(date -Is) [r27-served-quality] $*" | tee -a "$R/audit.log"; }
log "launcher md5 $(md5sum < $LAUNCHER | cut -c1-8); $(curl -s -m 5 $U/metrics | grep -aoE 'num_gpu_blocks="[0-9]+"' | head -1)"
( cd "$HOME" && timeout 5400 tool-eval-bench --base-url "$U/v1" --model qwen3.8-27b --temperature 0.6 --top-p 0.95 \
    --top-k 20 --trials 4 --parallel 8 --json-file "$R/tooleval-served.json" > "$R/tooleval-served.log" 2>&1 )
log "tool-eval rc=$? $(python3 scripts/tooleval_summary.py "$R/tooleval-served.json" served 2>&1 | tail -1)"
# TOOL_ONLY=1: a second tool-eval trial set only.
if [ "${TOOL_ONLY:-0}" = 1 ]; then echo "SUMMARY $(basename "$R") $(grep -a 'tool-eval rc' "$R/audit.log" | cut -c27-)" | tee -a "$R/summary.txt"; exit 0; fi
timeout 7200 lm_eval --model local-chat-completions \
  --model_args "base_url=$U/v1/chat/completions,model=qwen3.8-27b,tokenizer=$MODEL_DIR,num_concurrent=4,max_retries=1,tokenized_requests=False" \
  --tasks gsm8k --num_fewshot 5 --limit 200 --apply_chat_template \
  --gen_kwargs '{"temperature":0,"chat_template_kwargs":{"enable_thinking":false,"preserve_thinking":true}}' --log_samples \
  --output_path "$R/gsm-served" > "$R/gsm-served.log" 2>&1
log "gsm8k rc=$? $(grep -aE '^\|gsm8k.*flexible' "$R/gsm-served.log" | head -1)"
# think-off check: no sample may carry a reasoning block
log "gsm8k samples with <think>: $(cat "$R"/gsm-served/*/samples_gsm8k_*.jsonl 2>/dev/null | grep -c '<think>')"
echo "SUMMARY $(basename "$R") $(grep -a 'tool-eval rc\|gsm8k rc' "$R/audit.log" | cut -c27- | tr '\n' ' ')" | tee -a "$R/summary.txt"
