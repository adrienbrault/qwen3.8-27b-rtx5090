#!/usr/bin/env bash
# R159c — finish the ladder against the STILL-RUNNING :8029 engine from r159-conc-b (no third boot / daily outage).
# decode_ss reported "no steady-state window (ss=0)" at c32/c64 with --tokens 512: prefill admission is staggered
# and the first streams finish before the last ones start, so all c streams never generate at once. Longer outputs
# (2048 tokens) give the overlap. Also redoes prose c8/c16 (the b unit was stopped mid-prose). Then settle + daily restore.
#   sudo systemd-run --unit=r159c-live --collect -p User=adrienbrault -p RuntimeMaxSec=3000 bash /srv/qwen5090/r159c-live.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-02-r159-conc-b
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
U=http://127.0.0.1:8029
settle(){ timeout 120 sudo docker rm -f vllm-27b vllm-exp vllm-eval >/dev/null 2>&1
  for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep 45; }
metrics(){ curl -s -m 5 $U/metrics | grep -aE "^vllm:spec_decode_num_(accepted|draft)_tokens_total" | sed "s/^/[$1] /" | tee -a "$R/audit.log"; }
ds(){ local kind=$1 tag=$2; shift 2
  python3 /srv/qwen5090/probes/decode_ss.py --url $U --model qwen3.8-27b --kind $kind "$@" --out "$R/decode-$kind-$tag.json" > "$R/decode-$kind-$tag.out" 2>&1
  grep -a "RESULT\|error" "$R/decode-$kind-$tag.out" | sed "s/^/[decode_ss $kind $tag] /" | cut -c1-230 | tee -a "$R/audit.log"
}
log "=== R159c start (live :8029 engine) ==="
if curl -s -m 5 -o /dev/null -w "%{http_code}" $U/health | grep -q 200; then
  ds prose c8-16 --conc 8 16 --tokens 512 --runs 3
  ds code  c32-64 --conc 32 64 --tokens 2048 --runs 2
  ds prose c32-64 --conc 32 64 --tokens 2048 --runs 2
  metrics after-c
  sudo docker logs vllm-exp > "$R/engine.log" 2>&1 || true
  grep -ac "memory allocation failed with OOM" "$R/engine.log" | sed "s/^/[engine] allocator-OOM-retries: /" | tee -a "$R/audit.log"
  grep -aiE "error" "$R/engine.log" | grep -av "allocation failed" | wc -l | sed "s/^/[engine] other error-lines: /" | tee -a "$R/audit.log"
else log "[R159c] :8029 not healthy — nothing to measure"; fi
settle
bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY UP|FAILED|KV pool|attempt|RESTORED" | cut -c1-160 | tee -a "$R/audit.log"
log "=== R159c DONE ==="
