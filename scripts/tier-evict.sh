#!/usr/bin/env bash
# flan KV disk-tier evictor — the eviction vLLM's fs tier does not have.
#
# FACTS (verified in vLLM source, v0.28.0 and main):
#   * vllm/v1/kv_offload/tiering/fs/manager.py: the fs secondary tier has NO eviction, capacity
#     limit or TTL; only the CPU primary tier has `eviction_policy`. RFC #38260 delegates eviction
#     to each tier; llm-d docs: "must be handled by ... an external controller" (their PVC Evictor).
#   * lookup() = os.path.exists -> a deleted block is a MISS; every primary store re-cascades to the
#     fs tier -> deleted blocks are re-written on next use; writes are atomic (.tmp + os.replace).
#   * BUT kv_connector/v1/offloading/worker.py (v0.28.0) has `assert transfer_result.success`
#     ("we currently do not support job failures"): a file deleted AFTER the scheduler's lookup and
#     BEFORE the worker's read CRASHES THE ENGINE. So live deletion is only safe when no request can
#     be in that lookup->load window, i.e. when the engine is IDLE. This script therefore:
#       - gates every deletion batch on the daily's /metrics: running == 0 AND waiting == 0;
#       - deletes in small batches, re-checking idleness before each (window: milliseconds);
#       - skips anything accessed in the last HOT_MIN minutes (atime; tier mounted strictatime,lazytime);
#       - never deletes directories; removes orphan .tmp files older than 10 min.
#     Race-FREE eviction happens at boot in launch-daily-v0280.sh (engine down). This runtime
#     sweeper exists so the tier never reaches 100% between boots (a runtime ENOSPC on store is the
#     same unsupported-failure path).
# Policy (PVC Evictor defaults): trigger >= HIGH% used, delete coldest-first until <= LOW%.
set -uo pipefail
MNT=${MNT:-/srv/qwen5090/native-l2}; DAILY=${DAILY:-http://127.0.0.1:8020}
HIGH=${HIGH:-85}; LOW=${LOW:-70}; HOT_MIN=${HOT_MIN:-30}; BATCH=${BATCH:-200}
LOG=/srv/qwen5090/results/tier-evict.log; log(){ echo "$(date -Is) $*" | tee -a "$LOG"; }
used(){ df --output=pcent "$MNT" | tail -1 | tr -dc 0-9; }
idle(){  # true when NO engine can be inside a lookup->load window. Unreachable daily => no engine using the tier (experiments run NO_TIER on :8029).
  local m; m=$(curl -s --max-time 2 "$DAILY/metrics" 2>/dev/null) || return 0
  [ -z "$m" ] && return 0
  local r w; r=$(echo "$m" | awk '/^vllm:num_requests_running/{print $2; exit}'); w=$(echo "$m" | awk '/^vllm:num_requests_waiting/{print $2; exit}')
  [ "${r%.*}" = 0 ] && [ "${w%.*}" = 0 ]
}
mountpoint -q "$MNT" || { log "ERROR: $MNT not mounted"; exit 1; }
U=$(used); [ "$U" -lt "$HIGH" ] && { echo "$(date -Is) ok: ${U}% used" >> "$LOG"; exit 0; }
# strictatime is the kernel default and is NOT shown in /proc/mounts; only noatime/relatime are. So: atime is real iff neither flag is present.
if grep -E "^[^ ]+ $MNT " /proc/mounts | grep -qE "noatime|relatime"; then KEY=T; log "WARN: $MNT mounted noatime/relatime — evicting by mtime (FIFO), not true LRU"; else KEY=A; fi
log "eviction: ${U}% used >= ${HIGH}% — target <= ${LOW}% (idle-gated, batches of $BATCH)"
sudo find "$MNT" -type f -name '*.tmp' -mmin +10 -delete 2>/dev/null
n=0; bytes=0; skipped=0
mapfile -t CAND < <(sudo find "$MNT" -type f -name '*.bin' -${KEY,,}min +"$HOT_MIN" -printf "%${KEY}@ %s %p\n" 2>/dev/null | sort -n)
i=0; total=${#CAND[@]}
while [ "$i" -lt "$total" ]; do
  [ "$(used)" -le "$LOW" ] && break
  if ! idle; then skipped=$((skipped+1)); [ "$skipped" -ge 20 ] && { log "engine busy for the whole run — leaving $(used)% used; next timer tick retries"; break; }; sleep 3; continue; fi
  for ((j=i; j<i+BATCH && j<total; j++)); do
    read -r _ size path <<<"${CAND[$j]}"; sudo rm -f -- "$path" && { n=$((n+1)); bytes=$((bytes+size)); }
  done
  i=$((i+BATCH))
done
log "done: deleted $n files ($((bytes/1024/1024/1024)) GiB), now $(used)% used, busy-waits=$skipped"
[ "$(used)" -le "$LOW" ] || [ "$skipped" -ge 20 ] || log "WARN: still $(used)% used — everything left is hot (<${HOT_MIN} min); traffic exceeds tier size"
