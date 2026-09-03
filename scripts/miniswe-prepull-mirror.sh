#!/usr/bin/env bash
# Pre-pull SWE-bench Verified instance images for a running miniswe-full.sh campaign through Google's public
# Docker Hub pull-through cache (mirror.gcr.io — its own upstream IP, so flan's anonymous Hub quota of
# ~100 manifest requests/h is not consumed; a fresh 4 GB image took 9 s on 2026-09-02). Each image is
# tagged with its docker.io name so mini-SWE-agent and the swebench harness find it locally and never
# pull. Stays a window AHEAD of the campaign (by preds.json count) and respects a root-fs free-space floor,
# so at most ~AHEAD images (~4 GB each) sit on disk beyond what the campaign itself holds.
# No daemon change (registry-mirrors would need a dockerd restart = every container down mid-campaign).
#   usage: miniswe-prepull-mirror.sh [RESULTS_DIR]     env: AHEAD=100 MIN_FREE_G=250 MIRROR=mirror.gcr.io
#   unit:  sudo systemd-run --unit=miniswe-prepull --collect -p User=adrienbrault bash /srv/qwen5090/miniswe-prepull-mirror.sh
set -uo pipefail
R=${1:-/srv/qwen5090/results/2026-09-02-miniswe-rh}
AHEAD=${AHEAD:-100}; MIN_FREE_G=${MIN_FREE_G:-250}; MIRROR=${MIRROR:-mirror.gcr.io}
log(){ echo "$(date -Is) [prepull-mirror] $*" | tee -a "$R/prepull-mirror.log"; }
img(){ echo "swebench/sweb.eval.x86_64.${1//__/_1776_}:latest" | tr 'A-Z' 'a-z'; }
done_count(){ python3 -c "import json,sys; print(len(json.load(open('$R/out/preds.json'))))" 2>/dev/null || echo 0; }
[ -s "$R/instances.txt" ] || { log "no instances.txt yet"; exit 1; }
N=$(wc -l < "$R/instances.txt"); log "start: $N instances, ahead=$AHEAD, floor ${MIN_FREE_G}G, mirror $MIRROR"
i=0; pulled=0
while read -r iid; do
  i=$((i+1)); im=$(img "$iid")
  sudo docker image inspect "docker.io/$im" >/dev/null 2>&1 && continue
  while :; do
    d=$(done_count); free=$(( $(df -k --output=avail / | tail -1 | tr -dc 0-9) / 1048576 ))
    (( i - d <= AHEAD )) && (( free >= MIN_FREE_G )) && break
    sleep 120
  done
  t=0
  until sudo docker pull -q "$MIRROR/$im" >/dev/null 2>"$R/prepull-mirror.err"; do
    t=$((t+1)); log "pull error #$t on $iid: $(head -c 160 "$R/prepull-mirror.err" | tr '\n' ' ')"; [ $t -ge 5 ] && break; sleep 60
  done
  [ $t -ge 5 ] && continue
  sudo docker tag "$MIRROR/$im" "docker.io/$im" && sudo docker rmi "$MIRROR/$im" >/dev/null 2>&1
  pulled=$((pulled+1)); (( pulled % 20 == 0 )) && log "pulled $pulled (at index $i, campaign done $(done_count), free $(df -h / | awk 'NR==2{print $4}'))"
done < "$R/instances.txt"
log "DONE: pulled $pulled new images"
