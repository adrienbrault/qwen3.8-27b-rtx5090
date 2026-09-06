#!/usr/bin/env bash
# R205d (2026-09-06, MTP prefix-cache reopen part 2, patch 0158): build the eagle-shift image on the R205 mtpcache image.
# CPU-only, niced, no GPU lock (the GPU stays with whatever unit holds it).
#   <mtpcache>-eagleshift = patches-v0290/Dockerfile.mtp-eagle-shift (0158 = Mamba groups retain the eagle-backed-off replay boundary, marker /opt/prs-markers/0158)
# Unit: sudo systemd-run --unit=r205d-build --collect -p User=adrienbrault -p RuntimeMaxSec=7200 -p Nice=19 -p IOSchedulingClass=idle bash /srv/qwen5090/build-r205d-eagle-shift.sh
set -uo pipefail
R=/srv/qwen5090/results/2026-09-06-r205d-eagleshift-image; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
D=/srv/qwen5090/patches-v0290
DAILY=$(sed -nE 's/^DAILY_IMG=([^ ]+).*/\1/p' /srv/qwen5090/launch-daily.sh | head -1); BASE=$DAILY-mtppcie-mtpcache
[ -n "$DAILY" ] && sudo docker image inspect "$BASE" >/dev/null 2>&1 || { log "FAILED: mtpcache image not found ($BASE)"; exit 1; }
for f in 0158-eagle-drop-mamba-replay-boundary-v0290.diff Dockerfile.mtp-eagle-shift; do [ -f "$D/$f" ] || { log "FAILED: $D/$f missing"; exit 1; }; done
TAG=$BASE-eagleshift
if [ "${FORCE:-0}" != 1 ] && sudo docker image inspect "$TAG" >/dev/null 2>&1; then log "--- build $TAG: image exists, skipped"; exit 0; fi
log "=== R205d build start on $BASE ==="
for attempt in 1 2 3; do
  log "--- build $TAG attempt $attempt"
  ( cd "$D" && sudo nice -n 19 ionice -c3 docker build -f Dockerfile.mtp-eagle-shift --build-arg BASE="$BASE" -t "$TAG" . ) > "$R/build.log" 2>&1; rc=$?
  grep -aE "APPLIED|patching file|Hunk|FAILED|Error|error:|failed to export layer" "$R/build.log" | tail -14 | cut -c1-200 | sed "s/^/[eagleshift] /" | tee -a "$R/audit.log"
  if [ $rc -eq 0 ]; then
    sudo docker run --rm --entrypoint cat "$TAG" /opt/prs-markers/0158 | grep -q 0158 || { log "FAILED: marker 0158 missing in $TAG"; exit 1; }
    log "=== R205d build DONE: $TAG $(sudo docker image inspect "$TAG" --format '{{.Id}}' | cut -c8-19) ==="; exit 0
  fi
  if grep -aq "failed to export layer\|failed to commit: rename" "$R/build.log"; then log "containerd layer-export race, retrying in 60 s"; sleep 60; continue; fi
  break
done
log "FAILED: build $TAG rc=$rc (log $R/build.log)"; exit 1
