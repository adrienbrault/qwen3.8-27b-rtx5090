#!/usr/bin/env bash
# R177 (2026-09-04, user: the README's configuration matrix lacks the served daily): the R142 battery on the served daily
# (0.29 nvfp4 route, SEQS 16 since R176) so the "Other configurations" table gets a same-instrument column. Runs ON the
# daily :8020 (traffic only, no GPU lock, no bounce): waits until the daily is up at max-num-seqs 16 (the miniswe-r174
# campaign's restore boots it), settles, then: decode_ss prose-c1 / code-c1 x3 (1024 tok), code-c8 x2, code-c16 x2 (r142f
# cell, 512 tok), ctx8k, deep30k, deep100k (prefill from TTFT), tool-eval x4, GSM8K T=0 x120, needle flood-2, acceptance.
# Unit: sudo systemd-run --unit=r177-matrix --collect -p User=adrienbrault -p RuntimeMaxSec=86400 bash /srv/qwen5090/r177-matrix.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-04-r177-matrix; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
U=http://127.0.0.1:8020
V=/srv/qwen5090/venv-lmeval/bin
log "=== R177 waiting for the daily at SEQS 16 on :8020 (miniswe-r174 restores it when the campaign ends) ==="
while :; do
  if curl -sf -m 5 $U/health >/dev/null && sudo docker inspect vllm-27b --format '{{json .Args}}' 2>/dev/null | grep -qE '"--max-num-seqs","16"|--max-num-seqs 16( |\\)'; then break; fi
  sleep 120
done
log "daily up at SEQS 16; settling 120 s before the battery"; sleep 120
curl -s -m 5 $U/metrics | grep -aoE '^vllm:cache_config_info.*' | grep -oE 'num_gpu_blocks="[0-9]+"|block_size="[0-9]+"' | tr '\n' ' ' | sed 's/^/[daily] /' | tee -a "$R/audit.log"; echo | tee -a "$R/audit.log"
sudo docker inspect vllm-27b --format '{{.Config.Image}}' | sed 's/^/[daily] image /' | tee -a "$R/audit.log"
p1(){ local tag=$1 name=$2; shift 2
  python3 /srv/qwen5090/probes/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$tag-$name.jsonl" > "$R/probe-$tag-$name.out" 2> "$R/probe-$tag-$name.err"
  grep -a RESULT "$R/probe-$tag-$name.out" | sed "s/^/[$tag $name] /" | cut -c1-260 | tee -a "$R/audit.log" || echo "[$tag $name] PROBE FAILED: $(grep -a . "$R/probe-$tag-$name.out" "$R/probe-$tag-$name.err" 2>/dev/null | tail -1 | cut -c1-140)" | tee -a "$R/audit.log"
}
tag=daily16
p1 "$tag" prose-c1 --conc 1 --tokens 1024 --runs 3 --kind prose
p1 "$tag" code-c1  --conc 1 --tokens 1024 --runs 3 --kind code
p1 "$tag" code-c8  --conc 8 --tokens 1024 --runs 2 --kind code
p1 "$tag" code-c16 --conc 16 --tokens 512 --runs 2 --kind code
p1 "$tag" ctx8k    --conc 1 --tokens 256 --runs 2 --kind prose --ctx 8000
p1 "$tag" deep30k  --conc 1 --tokens 512 --runs 2 --kind prose --ctx 30000
p1 "$tag" deep100k --conc 1 --tokens 512 --runs 2 --kind prose --ctx 100000
( cd "$HOME" && tool-eval-bench --base-url $U/v1 --model qwen3.8-27b --temperature 0.6 --top-p 0.95 --top-k 20 --trials 4 --parallel 8 --json-file "$R/tooleval-$tag.json" > "$R/tooleval-$tag.log" 2>&1 )
python3 /srv/qwen5090/probes/tooleval_summary.py "$R/tooleval-$tag.json" "$tag" 2>&1 | tee -a "$R/audit.log"
( cd "$R" && $V/lm_eval --model local-chat-completions --model_args "model=qwen3.8-27b,base_url=$U/v1/chat/completions,num_concurrent=16,max_retries=3,tokenized_requests=False,max_gen_toks=8192" --tasks gsm8k_cot_zeroshot --limit 120 --apply_chat_template --gen_kwargs "temperature=0.0" --output_path "$R/lmeval-$tag" --log_samples > "$R/lmeval-$tag.log" 2>&1 )
grep -aE "exact_match" "$R/lmeval-$tag.log" | head -1 | sed "s/^/[$tag gsm] /" | tee -a "$R/audit.log"
python3 /srv/qwen5090/probes/needle_tier.py --url $U --mode full --base-tokens 40000 --flood 2 --flood-tokens 30000 --max-tokens 2000 > "$R/needle-$tag.out" 2>&1
echo "[$tag] needles true=$(grep -ac '"needles_correct": true' "$R/needle-$tag.out") false=$(grep -ac '"needles_correct": false' "$R/needle-$tag.out" || echo 0)" | tee -a "$R/audit.log"
curl -s -m 5 $U/metrics | grep -aE "spec_decode_num_(accepted|draft)_tokens_total|num_preemptions_total" | sed "s/^/[$tag] /" | cut -c1-160 | tee -a "$R/audit.log"
log "engine error lines: $(sudo docker logs vllm-27b 2>&1 | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError')"
log "=== R177 DONE ==="
