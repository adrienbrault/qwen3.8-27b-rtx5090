#!/usr/bin/env bash
# R205 (2026-09-06, memo item 1 "restore MTP prefix caching and offload hits"): build the MTP cache-fix image on the pcie-mtp image (0148).
# CPU-only, niced, no GPU lock (the daily keeps serving while it builds).
#   <mtppcie>-mtpcache = patches-v0290/Dockerfile.mtp-cache (0152 #52807, 0154 #54637 excerpt, 0155 #52771 port, 0156 #54288)
# Unit: sudo systemd-run --unit=r205-build --collect -p User=adrienbrault -p RuntimeMaxSec=7200 -p Nice=19 -p IOSchedulingClass=idle bash /srv/qwen5090/build-r205-mtp-cache.sh
set -uo pipefail
R=/srv/qwen5090/results/2026-09-06-r205-mtp-cache-image; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
D=/srv/qwen5090/patches-v0290
DAILY=$(sed -nE 's/^DAILY_IMG=([^ ]+).*/\1/p' /srv/qwen5090/launch-daily.sh | head -1)
BASE=$DAILY-mtppcie
[ -n "$DAILY" ] && sudo docker image inspect "$BASE" >/dev/null 2>&1 || { log "FAILED: pcie-mtp image not found ($BASE)"; exit 1; }
sudo docker run --rm --entrypoint cat "$BASE" /opt/prs-markers/0148 2>/dev/null | grep -q 0148 || { log "FAILED: $BASE lacks the 0148 marker"; exit 1; }
for f in 0152-pr52807-offload-recurrent-load-boundary-v0290.diff 0154-pr54637-mamba-groups-not-eagle-v0290.diff 0155-pr52771-offload-hits-under-mtp-v0290.diff 0156-pr54288-offload-final-token-v0290.diff touched-0152-0156.txt Dockerfile.mtp-cache; do [ -f "$D/$f" ] || { log "FAILED: $D/$f missing"; exit 1; }; done
build(){ local tag=$1 df=$2 attempt rc; shift 2
  if [ "${FORCE:-0}" != 1 ] && sudo docker image inspect "$tag" >/dev/null 2>&1; then log "--- build $tag: image exists, skipped"; return 0; fi
  for attempt in 1 2 3; do
    log "--- build $tag ($df) attempt $attempt"
    ( cd "$D" && sudo nice -n 19 ionice -c3 docker build -f "$df" --build-arg BASE="$BASE" "$@" -t "$tag" . ) > "$R/build-${tag##*:}.log" 2>&1; rc=$?
    grep -aE "APPLIED|patching file|Hunk|FAILED|Error|error:|failed to export layer" "$R/build-${tag##*:}.log" | tail -14 | cut -c1-200 | sed "s/^/[${tag##*:}] /" | tee -a "$R/audit.log"
    [ $rc -eq 0 ] && { log "build $tag OK: $(sudo docker image inspect "$tag" --format '{{.Id}}' | cut -c8-19)"; return 0; }
    if grep -aq "failed to export layer\|failed to commit: rename" "$R/build-${tag##*:}.log"; then log "build $tag: containerd layer-export race, retrying in 60 s"; sleep 60; continue; fi
    break
  done
  log "FAILED: build $tag rc=$rc (log $R/build-${tag##*:}.log)"; return 1; }
log "=== R205 build start on $BASE ==="
build "$BASE-mtpcache" Dockerfile.mtp-cache || exit 1
log "=== R205 build DONE: $BASE-mtpcache ==="
