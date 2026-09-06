#!/usr/bin/env bash
# R205b (2026-09-06, memo item "GDN metadata overhead", vllm#52297 port = patch 0157): build the hoisted-GDN-metadata image on the daily image.
# CPU-only, niced, no GPU lock (the GPU stays with whatever unit holds it).
#   <daily>-gdncm = patches-v0290/Dockerfile.gdn-common-metadata (0157 = #52297 port by codex, BRIEF31; pure refactor, no knob, marker /opt/prs-markers/0157)
# Unit: sudo systemd-run --unit=r205b-build --collect -p User=adrienbrault -p RuntimeMaxSec=7200 -p Nice=19 -p IOSchedulingClass=idle bash /srv/qwen5090/build-r205b-gdn-common-metadata.sh
set -uo pipefail
R=/srv/qwen5090/results/2026-09-06-r205b-gdncm-image; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
D=/srv/qwen5090/patches-v0290
BASE=$(sed -nE 's/^DAILY_IMG=([^ ]+).*/\1/p' /srv/qwen5090/launch-daily.sh | head -1)
[ -n "$BASE" ] && sudo docker image inspect "$BASE" >/dev/null 2>&1 || { log "FAILED: daily image not found ($BASE)"; exit 1; }
for f in 0157-gdn-common-metadata-once-v0290.diff Dockerfile.gdn-common-metadata; do [ -f "$D/$f" ] || { log "FAILED: $D/$f missing"; exit 1; }; done
TAG=$BASE-gdncm
if [ "${FORCE:-0}" != 1 ] && sudo docker image inspect "$TAG" >/dev/null 2>&1; then log "--- build $TAG: image exists, skipped"; exit 0; fi
log "=== R205b build start on $BASE ==="
for attempt in 1 2 3; do
  log "--- build $TAG attempt $attempt"
  ( cd "$D" && sudo nice -n 19 ionice -c3 docker build -f Dockerfile.gdn-common-metadata --build-arg BASE="$BASE" -t "$TAG" . ) > "$R/build.log" 2>&1; rc=$?
  grep -aE "APPLIED|patching file|Hunk|FAILED|Error|error:|failed to export layer" "$R/build.log" | tail -14 | cut -c1-200 | sed "s/^/[gdncm] /" | tee -a "$R/audit.log"
  if [ $rc -eq 0 ]; then
    sudo docker run --rm --entrypoint cat "$TAG" /opt/prs-markers/0157 | grep -q 0157 || { log "FAILED: marker 0157 missing in $TAG"; exit 1; }
    log "=== R205b build DONE: $TAG $(sudo docker image inspect "$TAG" --format '{{.Id}}' | cut -c8-19) ==="; exit 0
  fi
  if grep -aq "failed to export layer\|failed to commit: rename" "$R/build.log"; then log "containerd layer-export race, retrying in 60 s"; sleep 60; continue; fi
  break
done
log "FAILED: build $TAG rc=$rc (log $R/build.log)"; exit 1
