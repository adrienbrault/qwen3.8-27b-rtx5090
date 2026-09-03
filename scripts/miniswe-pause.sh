#!/usr/bin/env bash
# Pause the mini-SWE-agent campaign and bring the daily back (user: "pause eval, bring back daily").
# Run as a unit so it survives the ssh session:
#   sudo systemd-run --unit=daily-pause-restore --collect -p User=adrienbrault bash /srv/qwen5090/miniswe-pause.sh
# Steps: stop the prepull helper, SIGKILL the runner cgroup (its TERM trap would be SIGKILLed by systemd's
# 90 s stop timeout before finish() completes), record the pause in audit.log, remove agent + scorer
# containers, capture the eval engine log BEFORE removing it (lesson of 2026-09-02), wait for idle GPUs,
# settle 60 s (R162: enough on a healthy box; daily-restore-retry falls back to 300 s on a failed boot), then daily-restore-retry.
# Resume: sudo systemd-run --unit=miniswe --collect -p User=adrienbrault -p RuntimeMaxSec=100000 \
#           -E START=<chunk start> bash /srv/qwen5090/miniswe-full.sh 500
set -u
R=${R:-/srv/qwen5090/results/2026-09-02-miniswe-rh}
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
sudo systemctl stop miniswe-prepull 2>/dev/null
sudo systemctl kill -s KILL miniswe 2>/dev/null; sleep 3
log "### PAUSED by user: runner killed, preds=$(python3 -c "import json;print(len(json.load(open('$R/out/preds.json'))))" 2>/dev/null || echo '?') ###"
ids=$(sudo docker ps -q --filter name=minisweagent-); [ -n "$ids" ] && sudo docker rm -f $ids >/dev/null
ids=$(sudo docker ps -aq --filter name=sweb.eval); [ -n "$ids" ] && sudo docker rm -f $ids >/dev/null
sudo docker logs vllm-eval > "$R/engine-pause-$(date +%H%M%S).log" 2>&1
sudo docker rm -f vllm-eval >/dev/null 2>&1
for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr "\n" " "); case "$u" in *[1-9][0-9][0-9][0-9]*) sleep 5;; *) break;; esac; done
log "gpus idle ($u MiB); settling 60s before daily restore"
sleep 60
bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt" | cut -c1-160 | tee -a "$R/audit.log"
