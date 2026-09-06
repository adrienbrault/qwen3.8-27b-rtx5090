#!/usr/bin/env bash
# R205 (2026-09-06, memo item 1 "restore MTP prefix caching and offload hits"): does the MTP head get its prefix-cache and tier hits back with the
# upstream fixes? R198 measured MTP ns3 on this hybrid at 0 prefix hits through 320K queries and 0 of 4 tier-served re-asks (27 s / 60 s full
# recomputes) because no KV group is annotated as the drafter's and the fallback flags every group — Mamba groups [0,1,2] included — as EAGLE/MTP.
# Image = <mtppcie>-mtpcache (build-r205-mtp-cache.sh): 0152 #52807, 0154 #54637 excerpt (coordinator fallback exempts Mamba groups), 0155 #52771
# port (offload hits under MTP), 0156 #54288. Same route as R198 (EXP=1 :8029, SEQS 16, PCIE_IPC=1, BSS=1, SPEC_METHOD=mtp SPEC_NS=3,
# VLLM_SM12X_PCIE_IPC_MTP=1, FORCE_FIRST, eval-l2 wiped) so the R198 sheet is the control, plus the numbers a promotion sheet needs:
#   boot (13.98 → 13.5) with the new warning text + the offloading "non-draft" info line; layout; /metrics census around a 120K cold + revisit
#   (prefix_hits must leave 0, revisit ≪ 19 s); warm-revisit gpu-warm + tier-flood (R204 PICKS arguments); warm_equal (T=0 cold-vs-warm
#   completion equality on 3×6K prompts, GPU-warm and after a flood — the stale-checkpoint check the needle answers are too coarse for);
#   needle gate (tier_served must be 4/4; R198 0/4); decode_ss code-c8 / prose-c1 / code-c16 / prose-30K-c1 (R197 M3p: 1,568 / 158 / 2,699 / 157;
#   DFlash ns7 daily R204: 1,680 / 158 / 2,489 / R197 151); dense + agentic rulers vs bf16 (MTP vs bf16 is not on record) + per-doc vs R203 ON;
#   error lines. R202 M5 rides along: MemAvailable before/after the daily teardown and after the candidate teardown (driver-fork pool retention).
# Verdict lives on the sheet; promotion (MTP-3 vs DFlash-7: c1 code −15 %, c8/c16 +10–18 %, 30K +10 %, pool +24 %, tier hits ?) is the user's call.
#   unit: sudo systemd-run --unit=r205-mtp-cache --collect -p User=adrienbrault -p RuntimeMaxSec=43200 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r205-mtp-cache bash /srv/qwen5090/r205-mtp-cache.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-06-r205-mtp-cache; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily.sh; L2=/srv/qwen5090/eval-l2; PR=/srv/qwen5090/probes
BF16_DIR=/srv/qwen5090/results/2026-09-01-r156-bf16-ladder; BF16_REF=$BF16_DIR/dump-a1-dense.jsonl; LADDER_CORPUS=/srv/qwen5090/r156-corpus.jsonl
R203=/srv/qwen5090/results/2026-09-06-r203-spec-ladder
DAILY=$(sed -nE 's/^DAILY_IMG=([^ ]+).*/\1/p' "$CAND" | head -1); IMG=$DAILY-mtppcie-mtpcache
PIN=13980000000; PIN_FALLBACK=13500000000
[ -n "$DAILY" ] || { log "ABORT: cannot read DAILY_IMG"; exit 3; }
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: image $IMG missing (build-r205-mtp-cache.sh)"; exit 3; }
for m in 0148 0152 0154 0155 0156; do sudo docker run --rm --entrypoint cat "$IMG" /opt/prs-markers/$m 2>/dev/null | grep -q "$m" || { log "ABORT: $IMG lacks the $m marker"; exit 3; }; done
[ -f "$BF16_REF" ] && [ -f "$LADDER_CORPUS" ] && [ -f "$BF16_DIR/dump-a1-agentic.jsonl" ] && [ -f "$BF16_DIR/agentic-ids.jsonl" ] || { log "ABORT: reference dumps/corpus missing"; exit 3; }
grep -qF 'SPEC_METHOD_=${SPEC_METHOD:-dflash}' "$CAND" || { log "ABORT: launch-daily.sh SPEC_METHOD wiring missing"; exit 3; }
for t in fidelity_ladder.py fidelity_compare.py agentic_ref.py ladder_doc_compare.py decode_ss.py kv_capacity_probe.py needle_gate.sh warm-revisit.py warm_equal.py; do [ -f "$PR/$t" ] || { log "ABORT: $PR/$t missing"; exit 3; }; done
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
memavail(){ log "[M5 MemAvailable $1] $(awk '/MemAvailable/{printf "%.2f GiB", $2/1048576}' /proc/meminfo)  engines: $(sudo docker ps --format '{{.Names}}' | grep -E '^vllm-' | tr '\n' ' ')"; }
finish(){ teardown; memavail after-candidate-teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R205 $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R205 start (lock held): $IMG — MTP ns3 on pcie_ipc with the prefix-cache/offload fixes (0152 0154 0155 0156); control = R198 (0 hits) ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; }
ELOG(){ sudo docker logs vllm-exp 2>&1; }
errs(){ ELOG | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError\|JointFailure'; }
memavail daily-up
teardown; memavail after-daily-teardown
wipe_l2
boot_once(){ local tag=$1 kv=$2 rc
  env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 KV_BYTES=$kv PCIE_IPC=1 BSS=1 CAND_IMG=$IMG SPEC_METHOD=mtp SPEC_NS=3 EXTRA_ENV_APPEND="-e VLLM_TRITON_FORCE_FIRST_CONFIG=1 -e VLLM_SM12X_PCIE_IPC_MTP=1" bash $CAND > "$R/boot-$tag-$kv.log" 2>&1; rc=$?
  if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
    ELOG > "$R/engine-boot-$tag.log"
    log "[$tag] BOOT OK pin=$kv pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9) min_free=$(grep -aoE 'min free VRAM [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9)MiB block=$(grep -aoE 'Setting attention block size to [0-9]+' "$R/engine-boot-$tag.log" | head -1 | tr -dc 0-9) aot_loaded=$(grep -ac 'Directly load AOT' "$R/engine-boot-$tag.log") artifacts=$(grep -aoE 'torch_aot_compile/[0-9a-f]{12}' "$R/engine-boot-$tag.log" | sort -u | cut -d/ -f2 | tr '\n' ',')"
    return 0; fi
  log "[$tag] boot pin=$kv FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-$tag-$kv.log" | tail -1 | cut -c1-220)"
  ELOG 2>/dev/null | grep -aiE "error|exception|Bug C|headroom" | head -5 | cut -c1-220 | sed "s/^/[$tag boot-err] /" | tee -a "$R/audit.log"
  ELOG > "$R/engine-bootfail-$tag.log" 2>/dev/null; teardown; return 1; }
if boot_once M3c $PIN; then T=M3c; elif boot_once M3cb $PIN_FALLBACK; then T=M3cb; log "[M3c] needed the 13.5 fallback"; else log "[M3c] BOOT FAILED at both pins"; finish ABORTED; exit 1; fi
log "[warn] $(grep -aoE "could be identified as the draft model's[^\"]{0,200}" "$R/engine-boot-$T.log" | head -1 | cut -c1-260)"
log "[offload] $(grep -aoE 'KV offloading: [^\"]{0,160}' "$R/engine-boot-$T.log" | sort -u | head -3 | tr '\n' '|' | cut -c1-300)"
log "[mtp-pcie] $(grep -aoE 'SM12X PCIe IPC: MTP drafter admitted[^\"]{0,80}' "$R/engine-boot-$T.log" | head -1)  num_spec=$(grep -aoE 'num_spec_tokens=[0-9]+' "$R/engine-boot-$T.log" | head -1)"
layout(){ log "[$1 layout] $(curl -s -m 5 $U/metrics | grep -aoE '^vllm:cache_config_info.*' | grep -oE 'block_size="[0-9]+"|num_gpu_blocks="[0-9]+"|kv_cache_size_tokens="[0-9]+"' | tr '\n' ' ') min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)MiB"; }
snap(){ curl -s -m 5 $U/metrics | awk -v t="$1" '/^vllm:prefix_cache_(queries|hits)_total/ {split($1,a,"{"); q[a[1]]+=$NF} /^vllm:kv_offload_total_bytes_total.*CPU_to_GPU/ {up+=$NF} /^vllm:kv_offload_total_bytes_total.*GPU_to_CPU/ {down+=$NF} /^vllm:num_preemptions_total/ {pre=$NF} END {printf "[metrics %s] prefix_queries=%d prefix_hits=%d tier_GPU_to_CPU=%.3f GB tier_CPU_to_GPU=%.3f GB preemptions=%s\n", t, q["vllm:prefix_cache_queries_total"], q["vllm:prefix_cache_hits_total"], down/1e9, up/1e9, pre}' | tee -a "$R/audit.log"; }
cap(){ log "[cap $1] $(python3 $PR/kv_capacity_probe.py --url $U "${@:2}" 2>&1 | tail -1 | cut -c1-330)"; }
revisit(){ local name=$1; shift 1
  python3 $PR/warm-revisit.py --url $U --model qwen3.8-27b --ctx 32000 "$@" > "$R/revisit-$name.log" 2>&1
  log "[revisit $name] $(grep -a RESULT "$R/revisit-$name.log" | cut -c1-330)"; }
equal(){ local name=$1; shift 1
  python3 $PR/warm_equal.py --url $U --model qwen3.8-27b --n 3 --ctx 6000 --max-tokens 48 "$@" > "$R/warm-equal-$name.log" 2>&1
  log "[warm-equal $name] $(tail -1 "$R/warm-equal-$name.log") $(grep -a RESULT "$R/warm-equal-$name.log" | cut -c1-420)"; }
p1(){ local name=$1; shift 1
  python3 $PR/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$name.jsonl" > "$R/probe-$name.out" 2> "$R/probe-$name.err"
  if grep -aq RESULT "$R/probe-$name.out"; then grep -a RESULT "$R/probe-$name.out" | sed "s/^/[$name] /" | cut -c1-260 | tee -a "$R/audit.log"; else log "[$name] PROBE FAILED: $(tail -1 "$R/probe-$name.err" | cut -c1-160)"; fi; }
ruler_dense(){ local T=$1
  timeout 3600 python3 $PR/fidelity_ladder.py --url $U --model qwen3.8-27b --corpus "$LADDER_CORPUS" --out "$R/dump-$T-dense.jsonl" --logprobs 20 --mode dense --resume > "$R/score-$T-dense.out" 2>&1
  curl -sf -m 5 $U/health >/dev/null || { log "ENGINE DIED during the dense ruler: $(sudo dmesg -T | grep -a Xid | tail -1 | cut -c1-160)"; return 1; }
  log "[$T dense] docs failed=$(grep -ac '^\[warn\]' "$R/score-$T-dense.out") records=$(wc -l < "$R/dump-$T-dense.jsonl")"
  python3 $PR/fidelity_compare.py --ref "$BF16_REF" --arm "$R/dump-$T-dense.jsonl" --label "$T" --json "$R/fidelity-$T-dense.json" 2>&1 | tee "$R/fidelity-$T-dense.txt" | grep -aE "overall top-1|corpus PPL|truncated KL" | cut -c1-200 | sed "s/^/[$T dense vs bf16] /" | tee -a "$R/audit.log"
  [ -s "$R203/dump-ON-dense.jsonl" ] && python3 $PR/ladder_doc_compare.py --a "$R/dump-$T-dense.jsonl" --b "$R203/dump-ON-dense.jsonl" --label-a $T --label-b R203ON 2>&1 | tee "$R/doc-compare-dense-$T-vs-R203ON.txt" | grep -aE "^corpus|^per-doc|^positions|^DOC-COMPARE" | cut -c1-260 | sed "s/^/[$T dense per-doc vs R203 ON] /" | tee -a "$R/audit.log"; }
ruler_agentic(){ local T=$1
  timeout 3600 python3 $PR/agentic_ref.py score --url $U --model qwen3.8-27b --ids "$BF16_DIR/agentic-ids.jsonl" --out "$R/dump-$T-agentic.jsonl" > "$R/score-$T-agentic.out" 2>&1
  curl -sf -m 5 $U/health >/dev/null || { log "ENGINE DIED during the agentic ruler"; return 1; }
  python3 $PR/fidelity_compare.py --ref "$BF16_DIR/dump-a1-agentic.jsonl" --arm "$R/dump-$T-agentic.jsonl" --label "AGENTIC-$T" --json "$R/bf16-$T-agentic.json" 2>&1 | tee "$R/bf16-$T-agentic.txt" | grep -aE "overall top-1|corpus PPL" | cut -c1-200 | sed "s/^/[$T agentic vs bf16] /" | tee -a "$R/audit.log"
  [ -s "$R203/dump-ON-agentic.jsonl" ] && python3 $PR/ladder_doc_compare.py --a "$R/dump-$T-agentic.jsonl" --b "$R203/dump-ON-agentic.jsonl" --label-a $T --label-b R203ON 2>&1 | tee "$R/doc-compare-agentic-$T-vs-R203ON.txt" | grep -aE "^corpus|^per-doc|^positions|^DOC-COMPARE" | cut -c1-260 | sed "s/^/[$T agentic per-doc vs R203 ON] /" | tee -a "$R/audit.log"; }
sleep 20; layout M3c; snap boot
cap short1 --ctx 0 --conc 1 --tokens 400 --ignore-eos --seed 31
snap after-short
cap ctx120k-cold --ctx 120000 --conc 1 --tokens 200 --ignore-eos --seed 32
snap after-120k-cold
cap ctx120k-revisit --ctx 120000 --conc 1 --tokens 200 --ignore-eos --seed 32
snap after-120k-revisit
cap five100k --ctx 120000 --conc 5 --tokens 3000 --ignore-eos --seed 33
revisit gpu-warm
equal gpu-warm
revisit tier-flood --flood 12 --flood-ctx 90000
equal tier-flood --flood 12 --flood-ctx 90000
snap after-revisits
log "needle gate start (131K + 220K cold, then 12x90K flood evicts them, then two re-asks each through the tier; R198: tier_served 0/4)"
U=$U bash $PR/needle_gate.sh mtpcache "$R" > "$R/needle-gate.log" 2>&1; rc=$?
log "needle gate rc=$rc: $(grep -aE 'SUMMARY' "$R/needle-gate.log" | tail -1 | cut -c1-330)"
grep -aE "^\[needle-gate" "$R/needle-gate.log" | grep -aE "depth=" | cut -c1-200 | sed "s/^/[needle] /" | tee -a "$R/audit.log"
snap after-needles
p1 code-c8 --conc 8 --tokens 1024 --runs 2 --kind code
p1 prose-c1 --conc 1 --tokens 1024 --runs 2 --kind prose
p1 code-c16 --conc 16 --tokens 1024 --runs 1 --kind code
p1 prose-30k-c1 --conc 1 --tokens 1024 --runs 2 --kind prose --ctx 30000
ruler_dense M3c && ruler_agentic M3c
log "[M3c] engine error lines: $(errs)  preemptions: $(curl -s -m 5 $U/metrics | grep -a '^vllm:num_preemptions_total' | awk '{print $NF}')  tier: $(curl -s -m 5 $U/metrics | grep -aE '^vllm:kv_offload_tiering_(fs|chunk)' | awk '{s=s" "$1"="$2} END{print s}' | cut -c1-220)  eval-l2 $(du -sh $L2 2>/dev/null | cut -f1)"
ELOG > "$R/engine-M3c-final.log"
grep -aE "BOOT OK|boot pin|BOOT FAILED|warn\]|offload\]|mtp-pcie|layout|metrics|cap |revisit|warm-equal|needle|RESULT|PROBE FAILED|vs bf16|per-doc|error lines|ENGINE DIED|M5|FAILED|DAILY|===" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
finish DONE
