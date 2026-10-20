#!/usr/bin/env bash
# no-block-drop daily promotion (user 2026-10-20 "Why would we not promote?!"): install the launcher that adds
# "disable_eagle_block_drop":true to the MTP speculative config, keep the BA4 launcher as rollback, then verify with one EXP
# boot of the installed launcher: the flag on the container, BA4 96/96, tool smoke 5/5, and 32K warm revisits identical to
# cold (probes/warm_identity.py, 3 prompts). GLM restored afterwards (the 27B daily is not serving while GLM holds the GPUs).
# RUN: sudo systemd-run --unit=r27-nbd-promote --collect -p RuntimeMaxSec=3600 -p Environment=HOME=$HOME bash /srv/qwen5090/nbd-promote/promote.sh
#   (copy scripts/serve-v031-nbd-daily.sh as launch-daily.sh and scripts/serve-v031-ba4-daily.sh as launch-daily-ba4.sh to
#   /srv/qwen5090/nbd-promote/ first; scripts/r27-ba4-lean/toolsmoke.py is installed as /srv/qwen5090/ba4-lean/toolsmoke.py)
set -euo pipefail
export HOME=${HOME:?} USER=${USER:-$(id -un)}
export PATH=$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
D=/srv/qwen5090; H=$D/nbd-promote
R=$D/results/$(date -u +%F-%H%M%S)-nbd-promote
mkdir -p "$R"; exec > >(tee -a "$R/promote.log") 2>&1
GPU_QUEUE_NAME=nbd-promote . $D/lib/gpu-queue.sh
exec 9>$D/gpu-exclusive.lock; flock 9
finish() {
  rc=$?
  sudo docker logs vllm-exp > "$R/engine.log" 2>&1 || true
  timeout 120 sudo docker rm -f vllm-exp >/dev/null 2>&1 || true
  rm -f "$GPU_QUEUE_MARK"; flock -u 9
  [ -n "$(gpu_queue_others)" ] || bash $D/r858/restart-glm-daily.sh > "$R/glm-restore.log" 2>&1 || echo "SUMMARY nbd-promote RESTORE-FAILED glm daily down"
  echo "SUMMARY nbd-promote $([ $rc = 0 ] && echo DONE || echo "FAILED rc=$rc") results=$R" | tee -a "$R/summary.txt"
}
trap finish EXIT
cp $D/launch-daily.sh "$R/launch-daily.before.sh"
cmp "$R/launch-daily.before.sh" "$H/launch-daily-ba4.sh"   # the live launcher must be exactly the BA4 one we keep as rollback
cp "$H/launch-daily-ba4.sh" $D/launch-daily-ba4.sh.new && mv $D/launch-daily-ba4.sh.new $D/launch-daily-ba4.sh
cp "$H/launch-daily.sh" $D/launch-daily.sh.new && mv $D/launch-daily.sh.new $D/launch-daily.sh
md5sum $D/launch-daily.sh $D/launch-daily-ba4.sh
sudo docker stop -t 80 glm53 > "$R/glm-stop.log" 2>&1 || true
sudo find $D/eval-l2 -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} +
env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 BSS=1 bash $D/launch-daily.sh > "$R/boot.log" 2>&1 || { tail -20 "$R/boot.log"; exit 1; }
tail -1 "$R/boot.log"
grep -q "no-block-drop" "$R/boot.log" || { echo "no-block-drop not in the boot label"; exit 1; }
grep -q "+ BA4 96/96 active" "$R/boot.log" || { echo "BA4 not 96/96 active"; exit 1; }
python3 $D/ba4-lean/toolsmoke.py http://127.0.0.1:8029 "$R/tools.json"
python3 -c 'import json,sys; t=json.load(open(sys.argv[1])); sys.exit(0 if t["passed"]==t["total"] else 1)' "$R/tools.json"
python3 $D/probes/warm_identity.py --url http://127.0.0.1:8029 --model qwen3.8-27b --ctx 32000 --n 3 --max-tokens 256 --seed 991 | tee "$R/ident.out"
python3 -c 'import json,sys; j=json.loads(open(sys.argv[1]).read().split("RESULT ",1)[1]); print("identity", j["identical"], "/", j["n"]); sys.exit(0 if j["identical"]==j["n"] else 1)' "$R/ident.out"
