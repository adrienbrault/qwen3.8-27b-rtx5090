#!/usr/bin/env bash
# r27-l1a-curves (2026-10-17) — flan/r27-as-curves.sh re-run on the L1a daily (launch-daily.sh = repo 34b91b77: A_S + the L1a
# draft-plan layer, VLLM_SM12X_DRAFT_UB_PLAN=1). User 2026-10-17 "L1a sure promote"; promotion pushes = re-measured charts.
# Method, instruments, shapes, runs and seed scheme identical to r27-as-curves (and R675), so the curves compare point for
# point; only the asserted launcher md5, names and seed prefixes change. GLM daily stopped/restored as there. GPU ~35 min.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/$(date -u +%F-%H%M%S)-r27-l1a-curves; mkdir -p "$R"
U=http://127.0.0.1:8020; PR=/srv/qwen5090/probes
WANT_MD5=6347110bbd23a1ecc00a414b9d627fce
log(){ echo "$(date -Is) [r27-l1a-curves] $*" | tee -a "$R/audit.log"; }
echo "SUMMARY $(basename "$R") results $R" >> "$R/summary.txt"
export GPU_QUEUE_NAME=r27-l1a-curves
. /srv/qwen5090/lib/gpu-queue.sh
vram_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' ' '; }
gpus_empty(){ for i in $(seq 36); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}')" = 0 ] && return 0; sleep 5; done; return 1; }
teardown27(){ sudo docker ps -a --format '{{.Names}}' | grep -qx vllm-27b || return 0
  sudo docker logs vllm-27b > "$R/engine-vllm-27b-$(date +%H%M%S).log" 2>&1; sudo docker rm -f vllm-27b >/dev/null 2>&1; gpus_empty
  # docker rm -f leaves vLLM's 16 GB /dev/shm CPU-offload region behind; the next 27B boot then fails "Insufficient space
  # in /dev/shm" (2026-10-16 20:53, the j35 L2a screen right after this unit).
  sudo /srv/qwen5090/shm-sweep.sh quiet >> "$R/shm-sweep.log" 2>&1 || true; }
BOOTED=0
finish(){ teardown27; bash /srv/qwen5090/daily-power.sh stock >/dev/null 2>&1
  rm -f "$GPU_QUEUE_MARK"
  if [ "$BOOTED" = 1 ]; then
    if [ -n "$(gpu_queue_others)" ]; then log "GPU queue continues ($(gpu_queue_others)); GLM restore left to the next unit"
    elif timeout -k 20 1380 bash /srv/qwen5090/r858/restart-glm-daily.sh > "$R/glm-restore.log" 2>&1 \
         && curl -fsS -m 5 http://127.0.0.1:8029/health >/dev/null; then log "GLM daily restored on :8029"
    else log "GLM RESTORE FAILED (glm-restore.log)"; fi
  fi
  log "=== r27-l1a-curves $1 ==="; echo "SUMMARY $(basename "$R") $1" >> "$R/summary.txt"; }
trap 'log "signal"; finish ABORTED; exit 4' TERM INT HUP
for f in /srv/qwen5090/launch-daily.sh $PR/decode_ss.py $PR/kv_capacity_probe.py /srv/qwen5090/daily-power.sh /srv/qwen5090/r858/restart-glm-daily.sh; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
[ "$(md5sum < /srv/qwen5090/launch-daily.sh | cut -d' ' -f1)" = "$WANT_MD5" ] || { log "ABORT: launch-daily.sh md5 != $WANT_MD5 (not the L1a launcher)"; exit 3; }

gpu_lock
log "lock held"
BOOTED=1
sudo docker logs glm53 > "$R/glm-final.log" 2>&1 || true
sudo docker rm -f glm53 glm53-preflight >/dev/null 2>&1 || true
gpus_empty || log "WARN: GPU memory still in use after GLM stop: $(vram_free)"
UP=0
for try in 1 2 3; do
  log "boot attempt $try/3: launch-daily.sh"
  if env -i HOME="$HOME" USER="$USER" PATH="$PATH" bash /srv/qwen5090/launch-daily.sh > "$R/boot-$try.log" 2>&1 && curl -sf -m 5 $U/health >/dev/null; then UP=1; break; fi
  log "attempt $try FAILED: $(grep -aoE 'CUDA error: [a-z ]+|FAILED: [^,]{0,90}|ABORT[^,]{0,120}' "$R/boot-$try.log" | tail -1)"; teardown27
done
[ "$UP" = 1 ] || { finish NO-BOOT; exit 3; }
grep -a 'DAILY UP' "$R/boot-$try.log" | tail -1 | cut -c1-300 | tee -a "$R/audit.log"
bash /srv/qwen5090/daily-power.sh stock 2>&1 | grep -aiE '^WARN' | tee -a "$R/audit.log"
POOL=$(curl -s -m 5 $U/metrics | grep -aoE 'kv_cache_size_tokens="[0-9]+"' | grep -oE '[0-9]+' | head -1)
log "UP: pool ${POOL:-?} tokens, free $(vram_free), power $(nvidia-smi --query-gpu=index,power.limit --format=csv,noheader | tr '\n' ' ')"
log "image $(sudo docker ps --format '{{.Image}}' -f name=^vllm-27b$)"

NONCE=$(( $(date +%s) % 100000 ))
dec(){ local name=$1; shift
  python3 $PR/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$name.jsonl" > "$R/probe-$name.out" 2> "$R/probe-$name.err"
  grep -a RESULT "$R/probe-$name.out" | sed "s/^/[$name] /" | cut -c1-230 | tee -a "$R/audit.log" >/dev/null
  grep -aq RESULT "$R/probe-$name.out" || log "[$name] PROBE FAILED: $(tail -2 "$R/probe-$name.err" | tr '\n' ' ' | cut -c1-200)"; }
log "decode c1..c16, code and prose"
for kind in code prose; do dec "$kind" --kind $kind --conc 1 2 4 6 8 12 16 --tokens 1024 --runs 3 --seed-prefix "r27l-$NONCE-"; done
log "cold prefill, 3 fresh prompts per length"
for ctx in 8000 30000 60000 120000 200000 240000; do for i in 1 2 3; do
  seed=$(( (NONCE * 7 + ctx / 1000 * 10 + i) % 100000 ))
  python3 $PR/kv_capacity_probe.py --url $U --ctx $ctx --conc 1 --tokens 1 --seed $seed >> "$R/prefill.jsonl" 2>> "$R/prefill.err"
  log "  prefill ctx $ctx #$i: $(tail -1 "$R/prefill.jsonl" | cut -c1-200)"
done; done
log "decode at depth, one stream"
for ctx in 30000 60000 120000 200000; do for kind in code prose; do
  dec "$kind-c1-$((ctx / 1000))k" --kind $kind --conc 1 --ctx $ctx --tokens 1024 --runs 2 --seed-prefix "r27ld-$NONCE-"
done; done
log "engine errors: $(sudo docker logs vllm-27b 2>&1 | grep -ac 'ERROR')  preemptions: $(curl -s -m 5 $U/metrics | grep -aE '^vllm:num_preemptions_total' | awk '{print $2}' | head -1)  xid: $(sudo journalctl -k --since '-2h' --no-pager | grep -c 'NVRM: Xid')"
finish DONE
