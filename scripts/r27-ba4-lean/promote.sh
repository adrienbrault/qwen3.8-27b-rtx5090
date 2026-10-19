#!/usr/bin/env bash
# BA4 daily promotion (user 2026-10-20 "go"): install the BA4 admission inputs and the BA4 daily launcher, keep the L1a
# launcher as rollback, then verify with one EXP boot of the installed launcher (:8029, same image/flags as the daily path)
# that BA4 reports 96/96 rank-layers active. GLM is restored afterwards (the 27B daily is not serving while GLM holds the GPUs).
#   ssh <serving host> 'sudo systemd-run --unit=r27-ba4-promote-$(date -u +%H%M%S) -p RuntimeMaxSec=3600 bash /srv/qwen5090/ba4-lean/promote.sh'
#   (copy scripts/serve-v031-ba4-daily.sh as launch-daily.sh and scripts/serve-v031-l1a-daily.sh as launch-daily-l1a.sh
#   to /srv/qwen5090/ba4-lean/ first; the serving host's copies differ from these in their header comments)
set -euo pipefail
export HOME=${HOME:-/root} USER=${USER:-root}
export PATH=${OPERATOR_BIN:-$HOME/.local/bin}:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
S=/srv/qwen5090/results/2026-10-19-r27-audition-screen-070155-3532152
H=/srv/qwen5090/ba4-lean
R=/srv/qwen5090/results/$(date -u +%F-%H%M%S)-ba4-promote
mkdir -p "$R"; exec > >(tee -a "$R/promote.log") 2>&1
GPU_QUEUE_NAME=ba4-promote . /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock; flock 9
finish() {
  rc=$?
  timeout 120 docker rm -f vllm-exp >/dev/null 2>&1 || true
  flock -u 9
  [ -n "$(gpu_queue_others)" ] || bash /srv/qwen5090/r858/restart-glm-daily.sh || echo "SUMMARY ba4-promote RESTORE-FAILED glm daily down"
  echo "SUMMARY ba4-promote $([ $rc = 0 ] && echo DONE || echo "FAILED rc=$rc") results=$R" | tee -a "$R/summary.txt"
}
trap finish EXIT
# 1. inputs the BA4 layer admits itself from (the screen's producer reports + binding; same GPUs, same image)
if [ ! -f /srv/qwen5090/ba4-inputs/binding.json ]; then
  mkdir -p /srv/qwen5090/ba4-inputs.new; cp -a "$S/preflight/ba4-inputs/." /srv/qwen5090/ba4-inputs.new/
  mv /srv/qwen5090/ba4-inputs.new /srv/qwen5090/ba4-inputs
fi
(cd /srv/qwen5090/ba4-inputs && sha256sum *) | tee "$R/ba4-inputs.sha256"
diff <(cd "$S/preflight/ba4-inputs" && sha256sum *) "$R/ba4-inputs.sha256"
# 2. launchers: rollback first, then the daily (.new + mv: never overwrite a script in place)
cp /srv/qwen5090/launch-daily.sh "$R/launch-daily.before.sh"
cmp "$R/launch-daily.before.sh" "$H/launch-daily-l1a.sh"   # the live launcher must be exactly the L1a one we keep as rollback
cp "$H/launch-daily-l1a.sh" /srv/qwen5090/launch-daily-l1a.sh.new && mv /srv/qwen5090/launch-daily-l1a.sh.new /srv/qwen5090/launch-daily-l1a.sh
cp "$H/launch-daily.sh" /srv/qwen5090/launch-daily.sh.new && mv /srv/qwen5090/launch-daily.sh.new /srv/qwen5090/launch-daily.sh
md5sum /srv/qwen5090/launch-daily.sh /srv/qwen5090/launch-daily-l1a.sh
# 3. verification boot of the installed launcher (EXP=1 takes the daily image and forces nothing else)
docker stop -t 80 glm53 > "$R/glm-stop.log" 2>&1 || true
sudo find /srv/qwen5090/eval-l2 -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} +
env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 bash /srv/qwen5090/launch-daily.sh > "$R/boot.log" 2>&1 || { tail -20 "$R/boot.log"; exit 1; }
tail -1 "$R/boot.log"
grep -q "+ BA4 96/96 active" "$R/boot.log" || { echo "BA4 not 96/96 active"; exit 1; }
python3 "$H/toolsmoke.py" http://127.0.0.1:8029 "$R/tools.json"
python3 -c 'import json,sys; t=json.load(open(sys.argv[1])); sys.exit(0 if t["passed"]==t["total"] else 1)' "$R/tools.json"
docker logs vllm-exp > "$R/engine.log" 2>&1
