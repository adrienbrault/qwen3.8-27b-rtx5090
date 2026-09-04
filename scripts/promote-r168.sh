#!/usr/bin/env bash
# R174 (2026-09-04): PROMOTE the 0.29 nvfp4 candidate to the daily (user "Promote, go" on flan/r168-DECISION.md). One shot,
# GPU-exclusive, queue-registered. Tears the fp8 daily down, wipes the native-l2 namespaces (fresh tier per generation: the
# 279 GB there are fp8-format blocks the nvfp4 engine cannot reuse), boots the NEW launch-daily.sh, smoke-tests (health, one
# chat completion, metrics, a short code c1 decode probe) and on ANY failure rolls back to the frozen fp8 launcher so the daily
# is never left down. The launcher files must already be in place (launch-daily.sh = 0.29 nvfp4, launch-daily-redhat-fp8-0902.sh).
# Unit: sudo systemd-run --unit=r174-promote --collect -p User=adrienbrault -p RuntimeMaxSec=3600 -p TimeoutStopSec=900 -E GPU_QUEUE_NAME=r174-promote bash /srv/qwen5090/promote-r168.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-04-r174-promote; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
grep -q "0.29 nvfp4 DAILY" /srv/qwen5090/launch-daily.sh || { log "ABORT: launch-daily.sh is not the 0.29 nvfp4 launcher"; exit 3; }
[ -x /srv/qwen5090/launch-daily-redhat-fp8-0902.sh ] || { log "ABORT: rollback launcher missing"; exit 3; }
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock; flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
log "=== R174 promotion start (lock held) ==="
U=http://127.0.0.1:8020; L2=/srv/qwen5090/native-l2
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-45}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
rollback(){ log "ROLLBACK: $1"; teardown
  if env -i HOME="$HOME" USER="$USER" PATH="$PATH" bash /srv/qwen5090/launch-daily-redhat-fp8-0902.sh > "$R/boot-rollback.log" 2>&1; then log "ROLLBACK UP: $(tail -1 "$R/boot-rollback.log" | cut -c1-200)"
  else log "ROLLBACK FAILED — the daily is DOWN: $(grep -aE 'FAILED' "$R/boot-rollback.log" | tail -1 | cut -c1-200)"; fi
  log "=== R174 FAILED (fp8 daily restored: see above) ==="; exit 1; }
trap 'log "### SIGTERM ###"; rollback SIGTERM' TERM
log "taking the fp8 daily down (v0.28 fp8, pool $(curl -s -m 3 $U/metrics | grep -aoE '^vllm:cache_config_info.*' | grep -oE 'num_gpu_blocks="[0-9]+"' | head -1))"; teardown
log "wiping native-l2 namespaces: $(du -sh "$L2" 2>/dev/null | cut -f1) used before"
sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync
log "native-l2 after wipe: $(df -h "$L2" | tail -1 | awk '{print $3" used, "$4" free"}')"
if env -i HOME="$HOME" USER="$USER" PATH="$PATH" bash /srv/qwen5090/launch-daily.sh > "$R/boot-daily.log" 2>&1; then log "BOOT: $(tail -1 "$R/boot-daily.log" | cut -c1-500)"
else log "boot FAILED: $(grep -aE 'FAILED' "$R/boot-daily.log" | tail -2 | tr '\n' ' ' | cut -c1-400)"; rollback "launch-daily.sh failed"; fi
curl -sf -m 5 $U/health >/dev/null || rollback "health check"
OUT=$(curl -sf -m 180 $U/v1/chat/completions -H 'content-type: application/json' -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Reply with the single word PROMOTED and nothing else."}],"max_tokens":256}')
echo "$OUT" > "$R/smoke-chat.json"
echo "$OUT" | grep -q PROMOTED || rollback "chat smoke test did not answer PROMOTED (see smoke-chat.json)"
log "smoke chat OK: $(echo "$OUT" | python3 -c 'import json,sys;j=json.load(sys.stdin);m=j["choices"][0]["message"];print(repr((m.get("content") or "")[:60]),j.get("usage"))' 2>&1 | cut -c1-200)"
curl -s $U/metrics | grep -aE "^vllm:(cache_config_info|spec_decode_num_drafts_total|spec_decode_num_accepted_tokens_total)" | cut -c1-200 | sed 's/^/[metrics] /' | tee -a "$R/audit.log"
log "free VRAM after smoke: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader | tr '\n' ' ')"
python3 /srv/qwen5090/probes/decode_ss.py --url $U --model qwen3.8-27b --conc 1 --kind code --tokens 512 --runs 2 --out "$R/decode-code-c1.json" > "$R/decode-code-c1.out" 2>&1
log "[dss code c1] $(tail -1 "$R/decode-code-c1.out" | cut -c1-200)"
curl -sf -m 5 $U/health >/dev/null || rollback "health check after the decode probe"
log "engine error lines: $(sudo docker logs vllm-27b 2>&1 | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError')"
log "=== R174 DONE — daily = 0.29 nvfp4 (S image, ns9, pool pinned), rollback launch-daily-redhat-fp8-0902.sh ==="
