#!/usr/bin/env bash
# r27-pin-ladder (2026-10-20; user: "Do it. Might as well optimize for myself, and for others"): R233's pin ladder re-run on
# today's served stack (vLLM 0.31 + A_S + L1a + BA4 + no-block-drop, launch-daily.sh e680228b). R233 (2026-09-09, v0.29)
# capped the SEQS-16 KV pin at 14.86 GB/GPU on BOOT RELIABILITY, not VRAM: 15.90 GB booted 2/3, 16.90 GB 1/3 (the R191
# warmup "CUDA error: invalid argument"), both holding >= 1,371 MiB free through the stress. The daily leaves ~3.9 GB free
# per card. Same method as R233: tier wiped before every boot, per-boot seeds, five 120K + one 250K cold stress with the
# cold-prefill assert, free VRAM sampled every 2 s; EXP :8029 with the daily's PCIE_IPC=1 BSS=1 SEQS=16, stock power.
# Rule, written before the data: a pin qualifies iff 3/3 boots clean, every stress verified cold, and >= 1,000 MiB free
# through the stress on the worst boot. 15.90 and 16.90 run 3 boots each; 17.90 runs 3 more only if 16.90 qualified.
# The pick is the largest qualifying pin; promotion (and BA4 re-admission at the new pin, if its reports bind the pin)
# is a separate step. GPU ~45 min (~65 with 17.90).
#   ssh flan 'sudo systemd-run --unit=r27-pin-ladder --collect -p User="$(id -un)" -p RuntimeMaxSec=43200 -p TimeoutStopSec=900 -E GPU_QUEUE_NAME=r27-pin-ladder bash /srv/qwen5090/r27-pin-ladder.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/$(date -u +%F-%H%M%S)-r27-pin-ladder; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
U=http://127.0.0.1:8029; PR=/srv/qwen5090/probes; L2=/srv/qwen5090/eval-l2
[ "$(md5sum < /srv/qwen5090/launch-daily.sh | cut -c1-8)" = e680228b ] || { log "ABORT: launch-daily.sh is not the no-block-drop daily"; exit 3; }
cp /srv/qwen5090/launch-daily.sh "$R/"
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-45}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; teardown
  bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"
  log "=== r27-pin-ladder $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock"; flock 9; }
HAVE_LOCK=1
log "=== r27-pin-ladder start: 15.90 / 16.90 (17.90 if 16.90 qualifies), 3 boots each, verified-cold stress ==="
bash /srv/qwen5090/daily-power.sh stock 2>&1 | grep -aiE '^WARN' | sed 's/^/[power] /' | tee -a "$R/audit.log"
teardown
sample(){ while :; do nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1 >> "$1"; sleep 2; done; }
run(){ local tag=$1 kvb=$2 s1=$3 s2=$4
  log "--- $tag: pin $kvb ($(awk -v b=$kvb 'BEGIN{printf "%.2f", b/1073741824}') GiB), seeds $s1/$s2, tier wiped ---"
  sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + 2>/dev/null; sync
  sudo docker rm -f vllm-exp >/dev/null 2>&1
  if ! ( env -i PATH="$PATH" HOME="$HOME" USER="$(id -un)" EXP=1 SEQS=16 BSS=1 PCIE_IPC=1 KV_BYTES="$kvb" POOL_MIN=100000 POOL_MAX=1900000 \
        bash /srv/qwen5090/launch-daily.sh ) > "$R/boot-$tag.log" 2>&1; then
    local why; why=$(grep -aoE "CUDA error: [a-z ]+|FAILED: [^,]{0,80}" "$R/boot-$tag.log" | tail -1)
    log "[$tag] BOOT FAILED: ${why:-unknown}"; echo "$tag $kvb BOOTFAIL 0 0 0" >> "$R/ladder.txt"; teardown; return 1; fi
  local pool free; pool=$(tail -1 "$R/boot-$tag.log" | grep -aoE "Pool [0-9]+" | grep -oE "[0-9]+")
  free=$(tail -1 "$R/boot-$tag.log" | grep -aoE "min free VRAM [0-9]+" | grep -oE "[0-9]+")
  log "[$tag] BOOT OK  pool=${pool:-?}  free after pre-warm=${free:-?} MiB"
  : > "$R/vram-$tag.txt"; sample "$R/vram-$tag.txt" & local SP=$!
  local five one; five=$(python3 $PR/kv_capacity_probe.py --url $U --ctx 120000 --conc 5 --tokens 3000 --ignore-eos --seed "$s1" 2>&1 | tail -1)
  log "[$tag five120k] $(echo "$five" | cut -c1-260)"
  one=$(python3 $PR/kv_capacity_probe.py --url $U --ctx 250000 --conc 1 --tokens 200 --ignore-eos --seed "$s2" 2>&1 | tail -1)
  log "[$tag one250k] $(echo "$one" | cut -c1-260)"
  kill $SP 2>/dev/null; wait $SP 2>/dev/null
  # COLD-STRESS ASSERT: R232's arm C read 63,162 tok/s on this request because the tier served it. Cold is ~4,200.
  local rate; rate=$(echo "$one" | python3 -c 'import sys,json,re
s=sys.stdin.read()
m=re.search(r"\[\[([0-9.]+), ([0-9]+)\]\]", s)
print(round(int(m.group(2))/float(m.group(1))) if m else 0)' 2>/dev/null)
  local minfree; minfree=$(sort -n "$R/vram-$tag.txt" | head -1)
  if [ "${rate:-0}" -gt 10000 ]; then
    log "[$tag] COLD-STRESS ASSERT FAILED: 250K prefill read ${rate} tok/s (cold is ~4,200) — this boot was tier-served, headroom NOT comparable"
    echo "$tag $kvb ${pool:-0} ${free:-0} ${minfree:-0} CONTAMINATED-$rate" >> "$R/ladder.txt"; teardown; return 2; fi
  log "[$tag] cold verified (${rate} tok/s)  free DURING stress: min=${minfree:-?} MiB over $(wc -l < "$R/vram-$tag.txt") samples"
  echo "$tag $kvb ${pool:-0} ${free:-0} ${minfree:-0} cold-$rate" >> "$R/ladder.txt"
  teardown; return 0; }
N=$(( ($(date +%s%N) / 1000) % 90000 ))
qual(){ awk -v p=$1 '$2==p && $6 ~ /^cold/ && $5>=1000 {c++} END{print c+0}' "$R/ladder.txt"; }
for i in 1 2 3; do run "P1590-$i" 15900000000 $((N+i)) $((N+100+i)); done
for i in 1 2 3; do run "P1690-$i" 16900000000 $((N+200+i)) $((N+300+i)); done
if [ "$(qual 16900000000)" = 3 ]; then for i in 1 2 3; do run "P1790-$i" 17900000000 $((N+400+i)) $((N+500+i)); done
else log "16.90 did not qualify: 17.90 skipped"; fi
log "=== ladder (tag pin pool free_prewarm free_in_stress state) ==="; cat "$R/ladder.txt" 2>/dev/null | tee -a "$R/audit.log"
for p in 15900000000 16900000000 17900000000; do grep -q " $p " "$R/ladder.txt" 2>/dev/null || continue
  fail=$(awk -v p=$p '$2==p && $3=="BOOTFAIL" {c++} END{print c+0}' "$R/ladder.txt")
  worst=$(awk -v p=$p '$2==p && $6 ~ /^cold/ {print $5}' "$R/ladder.txt" | sort -n | head -1)
  log "PIN $p: $(qual $p)/3 qualifying boots, $fail flakes, worst in-stress free=${worst:-n/a} MiB"
  [ "$(qual $p)" = 3 ] && echo "$p" > "$R/best.txt"
done
log "VERDICT largest qualifying pin: $(cat "$R/best.txt" 2>/dev/null || echo 'none above 14.86 (keep 14.86)')"
grep -aE "^2026|BOOT|free DURING|COLD-STRESS|PIN |ladder" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
finish "DONE"
echo "SUMMARY $(basename "$R") $(grep -a VERDICT "$R/audit.log" | tail -1 | cut -d" " -f2-)" >> "$R/summary.txt"
