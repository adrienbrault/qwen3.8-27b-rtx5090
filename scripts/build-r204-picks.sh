#!/usr/bin/env bash
# R204 (2026-09-06, R202 §6 steps 3-4, user "ok go"): build the two cherry-pick images on the daily image. CPU-only, niced, no GPU lock
# (runs beside R203's rulers, which are prompt-logprob reads, not latency reads).
#   <daily>-picks = patches-v0290/Dockerfile.r202-picks  (0149 #55507, 0150 #54275, 0151 #54972, 0152 #52807)
#   <daily>-gdnfi = patches-v0290/Dockerfile.gdn-fi-prefill (0153 #50862 alone: new prefill kernel class, own numerics pair)
# Unit: sudo systemd-run --unit=r204-build --collect -p User=adrienbrault -p RuntimeMaxSec=7200 -p Nice=19 -p IOSchedulingClass=idle bash /srv/qwen5090/build-r204-picks.sh
set -uo pipefail
R=/srv/qwen5090/results/2026-09-06-r204-picks-image; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
D=/srv/qwen5090/patches-v0290
BASE=$(sed -nE 's/^DAILY_IMG=([^ ]+).*/\1/p' /srv/qwen5090/launch-daily.sh | head -1)
[ -n "$BASE" ] && sudo docker image inspect "$BASE" >/dev/null 2>&1 || { log "FAILED: daily image not found ($BASE)"; exit 1; }
for f in 0149-pr55507-mamba-align-seed-v0290.diff 0150-pr54275-kv-suggestion-clamp-v0290.diff 0151-pr54972-uva-empty-cache-v0290.diff 0152-pr52807-offload-recurrent-load-boundary-v0290.diff 0153-pr50862-gdn-fi-prefill-sm12x-v0290.diff touched-0149-0152.txt Dockerfile.r202-picks Dockerfile.gdn-fi-prefill; do [ -f "$D/$f" ] || { log "FAILED: $D/$f missing"; exit 1; }; done
build(){ local tag=$1 df=$2 attempt rc; shift 2
  if [ "${FORCE:-0}" != 1 ] && sudo docker image inspect "$tag" >/dev/null 2>&1; then log "--- build $tag: image exists, skipped"; return 0; fi
  for attempt in 1 2 3; do
    log "--- build $tag ($df) attempt $attempt"
    ( cd "$D" && sudo nice -n 19 ionice -c3 docker build -f "$df" --build-arg BASE="$BASE" "$@" -t "$tag" . ) > "$R/build-${tag##*:}.log" 2>&1; rc=$?
    grep -aE "APPLIED|patching file|Hunk|FAILED|Error|error:|failed to export layer" "$R/build-${tag##*:}.log" | tail -12 | cut -c1-200 | sed "s/^/[${tag##*:}] /" | tee -a "$R/audit.log"
    [ $rc -eq 0 ] && { log "build $tag OK: $(sudo docker image inspect "$tag" --format '{{.Id}}' | cut -c8-19)"; return 0; }
    if grep -aq "failed to export layer\|failed to commit: rename" "$R/build-${tag##*:}.log"; then log "build $tag: containerd layer-export race, retrying in 60 s"; sleep 60; continue; fi
    break
  done
  log "FAILED: build $tag rc=$rc (log $R/build-${tag##*:}.log)"; return 1; }
log "=== R204 build start on $BASE ==="
build "$BASE-picks" Dockerfile.r202-picks || exit 1
build "$BASE-gdnfi" Dockerfile.gdn-fi-prefill || exit 1
log "=== R204 build DONE: $BASE-picks, $BASE-gdnfi ==="
