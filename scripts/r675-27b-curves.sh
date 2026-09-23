#!/usr/bin/env bash
# R675 — decode and prefill curves of the served 27B daily (launch-daily.sh: nvidia NVFP4 checkpoint, NVFP4 KV at the
# 14.86 GB pin, MTP ns3, pcie_ipc all-reduce, batch-sharded sampling; R231/R234), measured on ONE boot for the public
# repo's figures (user 2026-09-23: "graphs for decode/prefill like the other public repo"). The 27B repo has decode
# points at 1/8/16 streams from R234 and a cold-prefill row from R183, which predates the served checkpoint. A figure
# needs one boot and one instrument per curve, so this measures them fresh.
# INSTRUMENTS (the 27B repo's own):
#   decode   probes/decode_ss.py: steady-state aggregate over the samples where all c streams run, 1,024 forced
#            tokens, 3 runs per shape, code and prose, c = 1 2 4 6 8 12 16 (16 = the served sequence limit).
#   prefill  probes/kv_capacity_probe.py --conc 1 --tokens 1: one cold request, prompt tokens counted by the server,
#            latency to the single output token = time to first token; 3 prompts per length at 8k/30k/60k/120k/200k/
#            240k filler, each with a fresh seed (clock nonce) so neither the GPU prefix cache nor the host/disk tiers
#            can serve it.
#   depth    decode_ss.py --conc 1 --ctx N at N = 30k/60k/120k/200k, code and prose, 2 runs, fresh seed prefix.
# Power: stock limits while measuring (every published 27B number is at stock; launch-daily.sh re-caps to 400 W).
# The Flash-Next daily (:8022) is stopped for the run and restored by the queue afterwards.
# GPU ~35 min. Queue-chained.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-23-r675-27b-curves; mkdir -p "$R"
U=http://127.0.0.1:8020; PR=/srv/qwen5090/probes
LIVE=/srv/qwen5090/launch-flashnext.sh
log(){ echo "$(date -Is) [r675] $*" | tee -a "$R/audit.log"; }
export GPU_QUEUE_NAME=r675-27b-curves
. /srv/qwen5090/lib/gpu-queue.sh
. /srv/qwen5090/lib/serve-ctl.sh
SCTL_LOG="$R/audit.log"
teardown27(){ sudo docker ps -a --format '{{.Names}}' | grep -qx vllm-27b || return 0
  sudo docker logs vllm-27b > "$R/engine-vllm-27b-$(date +%H%M%S).log" 2>&1; sudo docker rm -f vllm-27b >/dev/null 2>&1
  for i in $(seq 36); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}')" = 0 ] && break; sleep 5; done; }
finish(){ teardown27; bash /srv/qwen5090/daily-power.sh stock >/dev/null 2>&1
  log "power after: $(nvidia-smi --query-gpu=index,power.limit --format=csv,noheader | tr '\n' ' ')"
  finish_restore "$LIVE"; rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R675 $1 ==="; }
trap 'log "signal"; finish ABORTED; exit 4' TERM INT HUP
for f in /srv/qwen5090/launch-daily.sh $PR/decode_ss.py $PR/kv_capacity_probe.py /srv/qwen5090/daily-power.sh; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done

gpu_lock
log "lock held; served at entry: $(served_id || echo none)"
served_stop; BOOTED=1
for i in $(seq 36); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}')" = 0 ] && break; sleep 5; done
UP=0
for try in 1 2 3; do   # R233: the warmup flake is a lottery at this pin; draw again, never change the pin
  log "boot attempt $try/3: launch-daily.sh"
  if env -i HOME="$HOME" USER="$USER" PATH="$PATH" bash /srv/qwen5090/launch-daily.sh > "$R/boot-$try.log" 2>&1 && curl -sf -m 5 $U/health >/dev/null; then UP=1; break; fi
  log "attempt $try FAILED: $(grep -aoE 'CUDA error: [a-z ]+|FAILED: [^,]{0,90}' "$R/boot-$try.log" | tail -1)"; teardown27
done
[ "$UP" = 1 ] || { finish NO-BOOT; exit 3; }
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
for kind in code prose; do dec "$kind" --kind $kind --conc 1 2 4 6 8 12 16 --tokens 1024 --runs 3 --seed-prefix "r675-$NONCE-"; done
log "cold prefill, 3 fresh prompts per length"
for ctx in 8000 30000 60000 120000 200000 240000; do for i in 1 2 3; do
  seed=$(( (NONCE * 7 + ctx / 1000 * 10 + i) % 100000 ))
  python3 $PR/kv_capacity_probe.py --url $U --ctx $ctx --conc 1 --tokens 1 --seed $seed >> "$R/prefill.jsonl" 2>> "$R/prefill.err"
  log "  prefill ctx $ctx #$i: $(tail -1 "$R/prefill.jsonl" | cut -c1-200)"
done; done
log "decode at depth, one stream"
for ctx in 30000 60000 120000 200000; do for kind in code prose; do
  dec "$kind-c1-$((ctx / 1000))k" --kind $kind --conc 1 --ctx $ctx --tokens 1024 --runs 2 --seed-prefix "r675d-$NONCE-"
done; done
log "engine errors: $(sudo docker logs vllm-27b 2>&1 | grep -ac 'ERROR')  preemptions: $(curl -s -m 5 $U/metrics | grep -aE '^vllm:num_preemptions_total' | awk '{print $2}' | head -1)"
finish DONE
