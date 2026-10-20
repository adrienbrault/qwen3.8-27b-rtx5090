#!/usr/bin/env bash
# r27 flag screen (2026-10-20): two zero-code vLLM v0.31.0 speculative-config options, screened on the served BA4 launcher
#   (scripts/serve-v031-ba4-daily.sh installed as /srv/qwen5090/launch-daily.sh). Write-up: bench/results/r27-no-block-drop.md.
#   Installed as /srv/qwen5090/flags-screen/screen.sh with scripts/r27-flags-decide.py beside it as decide.py; the probes
#   scripts/decode_ss.py, scripts/reask.py and scripts/warm_identity.py under /srv/qwen5090/probes.
#   C = control (the daily launcher, EXP=1 :8029, BSS=1 as the daily)
#   P = C + SPEC_EXTRA '"draft_sample_method":"probabilistic"'  (drafter samples q, rejection sampler accepts min(1,p/q))
#   E = C + SPEC_EXTRA '"disable_eagle_block_drop":true'        (no re-prefill of the last full-attention block on a prefix hit)
# Boot order C P E P C E (each arm twice, interleaved), one boot ~3 min, measurements ~4 min per boot, GPU ~45 min.
# Per boot: decode_ss.py c1 + c3, code + prose, 3 runs, T 0.6 (the served sampling and every published decode number);
#           reask.py completions, 3 sends, 6K and 32K, gap 2 s (warm-revisit TTFT); seeds unique per boot.
# Rules, written before the data (decide.py):
#   P PASS iff accept/draft rises >= 1.5 pp on both kinds at c1, and per-stream t/s (median of its two boots) is >= C's
#          at every (kind, conc) cell minus 2 %, and >= +1 % at c1 on both kinds.
#   E PASS iff warm-send TTFT (median of sends 2-3, both boots) drops >= 0.08 s at 32K and does not rise at 6K, and
#          decode per-stream t/s is within 2 % of C in every cell.
# RUN: sudo systemd-run --unit=r27-flags-screen --collect -p RuntimeMaxSec=43200 -p TimeoutStopSec=900 -p Environment=HOME=$HOME bash /srv/qwen5090/flags-screen/screen.sh
set -uo pipefail
export HOME=${HOME:?} USER=${USER:-$(id -un)}
export PATH=$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
H=/srv/qwen5090/flags-screen; PR=/srv/qwen5090/probes; U=http://127.0.0.1:8029
R=/srv/qwen5090/results/$(date -u +%F-%H%M%S)-r27-flags-screen; mkdir -p "$R"
log(){ echo "$(date -Is) [r27-flags-screen] $*" | tee -a "$R/audit.log"; }
echo "SUMMARY $(basename "$R") results $R" >> "$R/summary.txt"
WANT_MD5=7cb1a981b0d7db5a54dd8e03c52e1b09
[ "$(md5sum < /srv/qwen5090/launch-daily.sh | cut -d' ' -f1)" = "$WANT_MD5" ] || { log "ABORT: launch-daily.sh is not the BA4 daily"; exit 3; }
cp /srv/qwen5090/launch-daily.sh "$R/launch-daily.sh"
export GPU_QUEUE_NAME=r27-flags-screen
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock; flock 9
teardown(){ sudo docker ps -a --format '{{.Names}}' | grep -qx vllm-exp || return 0
  sudo docker logs vllm-exp > "$R/engine-$1.log" 2>&1; sudo docker rm -f vllm-exp >/dev/null 2>&1
  sudo /srv/qwen5090/shm-sweep.sh quiet >> "$R/shm-sweep.log" 2>&1 || true; sleep 5; }
finish(){ rc=$?; teardown end; rm -f "$GPU_QUEUE_MARK"; flock -u 9
  if [ -z "$(gpu_queue_others)" ]; then timeout -k 20 1380 bash /srv/qwen5090/r858/restart-glm-daily.sh > "$R/glm-restore.log" 2>&1 \
    && log "GLM restored" || log "GLM RESTORE FAILED"; fi
  echo "SUMMARY $(basename "$R") $([ $rc = 0 ] && echo DONE || echo "FAILED rc=$rc")" | tee -a "$R/summary.txt"; }
trap finish EXIT
sudo docker stop -t 80 glm53 > "$R/glm-stop.log" 2>&1 || true
# Round 2 (2026-10-20, after round 1's decode cells carried ~±10 % boot noise): ARMS="C P C P C P" RUNS=10 REASK=0
# decides P alone with three boots per arm and ten runs per cell; decide.py then reports only the arms present.
# Round 3 (2026-10-20, E's correctness): ARMS="C E C E" DECODE=0 REASK=0 IDENT=1 runs probes/warm_identity.py (6K x 10, 32K x 6
#   prompts, each sent cold then warm, greedy, 256 tokens). Rule, written before the data: E PASS iff its warm==cold identical
#   count over both boots is >= C's minus 1 at each ctx, and its warm TTFT stays below C's.
ARMS=${ARMS:-"C P E P C E"}; RUNS=${RUNS:-3}; REASK=${REASK:-1}; DECODE=${DECODE:-1}; IDENT=${IDENT:-0}
log "arms: $ARMS, runs per cell $RUNS, reask $REASK"
i=0
for arm in $ARMS; do
  i=$((i+1)); tag="$i-$arm"
  case $arm in C) SX="";; P) SX='"draft_sample_method":"probabilistic"';; E) SX='"disable_eagle_block_drop":true';; esac
  sudo find /srv/qwen5090/eval-l2 -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} +
  log "boot $tag SPEC_EXTRA=${SX:-none}"
  if ! env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 BSS=1 SPEC_EXTRA="$SX" bash /srv/qwen5090/launch-daily.sh > "$R/boot-$tag.log" 2>&1; then
    log "boot $tag FAILED: $(tail -1 "$R/boot-$tag.log")"; teardown "$tag"; exit 1; fi
  log "UP $tag: $(grep -ao 'BA4 [0-9]*/96 active' "$R/boot-$tag.log" | tail -1); spec: $(grep -ao 'speculative_config=[^ ]*' "$R/boot-$tag.log" | head -1)"
  sudo docker logs vllm-exp 2>&1 | grep -aoE '"?(draft_sample_method|disable_eagle_block_drop)"?[=:] ?[^,} ]*' | sort -u | sed "s/^/[$tag] engine: /" | tee -a "$R/audit.log"
  [ "$IDENT" = 1 ] && for ctxn in 6000:10 32000:6; do
    python3 $PR/warm_identity.py --url $U --model qwen3.8-27b --ctx ${ctxn%:*} --n ${ctxn#*:} --max-tokens 256 --seed $((i*7)) \
      > "$R/ident-$tag-${ctxn%:*}.out" 2>&1 || log "ident $tag ${ctxn%:*} rc=$?"
    log "[$tag] $(grep -h RESULT "$R/ident-$tag-${ctxn%:*}.out" | cut -c1-240)"
  done
  [ "$DECODE" = 1 ] && for kind in code prose; do
    python3 $PR/decode_ss.py --url $U --model qwen3.8-27b --kind $kind --conc 1 3 --tokens 1024 --runs $RUNS \
      --seed-prefix "fs$i$arm" --out "$R/decode-$tag-$kind.jsonl" > "$R/probe-$tag-$kind.out" 2>&1 || log "decode $tag $kind rc=$?"
  done
  [ "$REASK" = 1 ] && for ctx in 6000 32000; do
    python3 $PR/reask.py --url $U --model qwen3.8-27b --api completions --ctx $ctx --max-tokens 8 --gap 2 --sends 3 \
      --seed $((RANDOM*7 + i)) > "$R/reask-$tag-$ctx.out" 2>&1 || log "reask $tag $ctx rc=$?"
  done
  grep -h RESULT "$R"/probe-$tag-*.out "$R"/reask-$tag-*.out | cut -c1-260 | sed "s/^/[$tag] /" >> "$R/audit.log"
  teardown "$tag"
done
python3 $H/decide.py "$R" | tee "$R/verdict.txt"
echo "SUMMARY $(basename "$R") $(grep -a '^VERDICT' "$R/verdict.txt" | tr '\n' ' ')" >> "$R/summary.txt"
