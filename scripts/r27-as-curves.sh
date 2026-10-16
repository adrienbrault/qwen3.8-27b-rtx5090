#!/usr/bin/env bash
# r27-as-curves (2026-10-16) — R675's decode and prefill curves re-measured on the promoted A_S daily (launch-daily.sh =
# repo 06faf891: vLLM v0.31.0 + FlashInfer 0.7.0.post1, split-KV override, V2 runner, triton mamba + FlashInfer GDN
# prefill; same checkpoint, NVFP4 KV pin, MTP ns3, pcie_ipc, BSS). User 2026-10-16: "you pushed 27b readme without
# updating charts?!" — the public README's decode-scaling-16 solid lines and prefill.svg were still R675 (v0.29 launcher).
# Method identical to flan/r675-27b-curves.sh (same instruments, shapes, runs, seeds scheme, stock power) so the two
# curves are comparable point for point. Differences from R675: the GLM daily (:8029, glm53) holds the GPUs, so it is
# stopped (logs archived) and restored with r858/restart-glm-daily.sh unless another unit is queued; launcher md5 asserted.
# GPU ~35 min. Queue-chained (registers before the flock; the r27 j35 chain waits for an empty queue).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/$(date -u +%F-%H%M%S)-r27-as-curves; mkdir -p "$R"
U=http://127.0.0.1:8020; PR=/srv/qwen5090/probes
WANT_MD5=63e147ebd6b70f22f7f83c984b4382e3
log(){ echo "$(date -Is) [r27-as-curves] $*" | tee -a "$R/audit.log"; }
echo "SUMMARY $(basename "$R") results $R" >> "$R/summary.txt"
export GPU_QUEUE_NAME=r27-as-curves
. /srv/qwen5090/lib/gpu-queue.sh
vram_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' ' '; }
gpus_empty(){ for i in $(seq 36); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}')" = 0 ] && return 0; sleep 5; done; return 1; }
teardown27(){ sudo docker ps -a --format '{{.Names}}' | grep -qx vllm-27b || return 0
  sudo docker logs vllm-27b > "$R/engine-vllm-27b-$(date +%H%M%S).log" 2>&1; sudo docker rm -f vllm-27b >/dev/null 2>&1; gpus_empty; }
BOOTED=0
finish(){ teardown27; bash /srv/qwen5090/daily-power.sh stock >/dev/null 2>&1
  rm -f "$GPU_QUEUE_MARK"
  if [ "$BOOTED" = 1 ]; then
    if [ -n "$(gpu_queue_others)" ]; then log "GPU queue continues ($(gpu_queue_others)); GLM restore left to the next unit"
    elif timeout -k 20 1380 bash /srv/qwen5090/r858/restart-glm-daily.sh > "$R/glm-restore.log" 2>&1 \
         && curl -fsS -m 5 http://127.0.0.1:8029/health >/dev/null; then log "GLM daily restored on :8029"
    else log "GLM RESTORE FAILED (glm-restore.log)"; fi
  fi
  log "=== r27-as-curves $1 ==="; echo "SUMMARY $(basename "$R") $1" >> "$R/summary.txt"; }
trap 'log "signal"; finish ABORTED; exit 4' TERM INT HUP
for f in /srv/qwen5090/launch-daily.sh $PR/decode_ss.py $PR/kv_capacity_probe.py /srv/qwen5090/daily-power.sh /srv/qwen5090/r858/restart-glm-daily.sh; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
[ "$(md5sum < /srv/qwen5090/launch-daily.sh | cut -d' ' -f1)" = "$WANT_MD5" ] || { log "ABORT: launch-daily.sh md5 != $WANT_MD5 (not the A_S launcher)"; exit 3; }

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
for kind in code prose; do dec "$kind" --kind $kind --conc 1 2 4 6 8 12 16 --tokens 1024 --runs 3 --seed-prefix "r27c-$NONCE-"; done
log "cold prefill, 3 fresh prompts per length"
for ctx in 8000 30000 60000 120000 200000 240000; do for i in 1 2 3; do
  seed=$(( (NONCE * 7 + ctx / 1000 * 10 + i) % 100000 ))
  python3 $PR/kv_capacity_probe.py --url $U --ctx $ctx --conc 1 --tokens 1 --seed $seed >> "$R/prefill.jsonl" 2>> "$R/prefill.err"
  log "  prefill ctx $ctx #$i: $(tail -1 "$R/prefill.jsonl" | cut -c1-200)"
done; done
log "decode at depth, one stream"
for ctx in 30000 60000 120000 200000; do for kind in code prose; do
  dec "$kind-c1-$((ctx / 1000))k" --kind $kind --conc 1 --ctx $ctx --tokens 1024 --runs 2 --seed-prefix "r27cd-$NONCE-"
done; done
log "engine errors: $(sudo docker logs vllm-27b 2>&1 | grep -ac 'ERROR')  preemptions: $(curl -s -m 5 $U/metrics | grep -aE '^vllm:num_preemptions_total' | awk '{print $2}' | head -1)  xid: $(sudo journalctl -k --since '-2h' --no-pager | grep -c 'NVRM: Xid')"
finish DONE
