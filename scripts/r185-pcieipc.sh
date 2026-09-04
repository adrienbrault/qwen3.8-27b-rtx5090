#!/usr/bin/env bash
# R185 (2026-09-05): FlashInfer's pcie_ipc all-reduce (PR #4393, main df8b5c1) vendored into the served image by patch 0138
# (package pcie_ipc_ar21, image ...-fi0616-pcieipc, opt-in VLLM_SM12X_PCIE_IPC_AR=1). R184 measured the kernel 36% faster than
# vLLM's custom all-reduce at 10 rows and 24% at 160 on free cards; the ceiling for the whole decode step is 2.5-5.5% at c1 and
# ~8% at c8/c16. Helpers are r183b-kernels.sh's (same rulers, same probes, same launch-daily EXP passthrough).
#   0. ar-bench under the GPU lock: nccl vs flashinfer-main pcie vs the vendored package (vend) at the decode rows. GATE: vend
#      graph-replay error 0 vs NCCL and vend within 15% of pcie at every row; a numerics failure stops the run (no engine boot on
#      a kernel that returns wrong sums).
#   1. PCIE-on: the pcieipc image with the knob, decode c1/c8/c16 + ttft 8k/120k + dfid ctx0/30K + dense/agentic rulers vs bf16.
#      The engine log must say "PCIe IPC all-reduce enabled" and list PCIE_IPC ahead of CUSTOM; if it says "disabled: <reason>",
#      the arm is measured as the image-parity control instead and the reason is the finding.
#   2. PCIE-off: same image, knob unset, decode + dfid ctx0 — the layer alone must be neutral against the r183b replicate band.
#   unit: sudo systemd-run --unit=r185-pcieipc --collect -p User=adrienbrault -p RuntimeMaxSec=10800 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r185-pcieipc bash /srv/qwen5090/r185-pcieipc.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-05-r185-pcieipc; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily.sh; L2=/srv/qwen5090/eval-l2; PR=/srv/qwen5090/probes
BF16_DIR=/srv/qwen5090/results/2026-09-01-r156-bf16-ladder; BF16_REF=$BF16_DIR/dump-a1-dense.jsonl; LADDER_CORPUS=/srv/qwen5090/r156-corpus.jsonl
FD=/srv/qwen5090/results/2026-08-23-fidelity; DREF=/srv/qwen5090/results/2026-09-04-r173c-bf16-decode
PINS="13980000000 13500000000"
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: image $IMG missing"; exit 3; }
[ -f "$BF16_REF" ] && [ -f "$LADDER_CORPUS" ] && [ -f "$BF16_DIR/dump-a1-agentic.jsonl" ] && [ -f "$FD/corpus.jsonl" ] && [ -f "$DREF/dec-bf16-ctx30000.jsonl" ] && [ -f "$DREF/dec-bf16-ctx0.jsonl" ] && [ -f "$R/pcie-tune-free.json" ] || { log "ABORT: references missing"; exit 3; }
grep -q "R183 EXP-only passthrough" "$CAND" || { log "ABORT: launch-daily.sh lacks the R183 EXP passthrough"; exit 3; }
grep -q '"vend"' /srv/qwen5090/probes/ar_bench.py || { log "ABORT: ar_bench.py lacks the vend backend"; exit 3; }
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval ar-bench; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R185 $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R185 start (lock held): pcie_ipc all-reduce vendored (0138) — ar-bench gate, then PCIE-on / PCIE-off arms on $IMG ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; }
ELOG(){ sudo docker logs vllm-exp 2>&1; }
errs(){ ELOG | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError\|JointFailure'; }
layout(){ curl -s -m 5 $U/metrics | grep -aoE '^vllm:cache_config_info.*' | grep -oE 'block_size="[0-9]+"|num_gpu_blocks="[0-9]+"|kv_cache_size_tokens="[0-9]+"' | tr '\n' ' '; }
boot_arm(){ local tag=$1 pins=$2 kv rc; shift 2
  for kv in $pins; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 KV_BYTES=$kv CAND_IMG=$IMG "$@" bash $CAND > "$R/boot-$tag-$kv.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
      log "[$tag] BOOT OK pin=$kv env='$*' pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9) min_free=$(grep -aoE 'min free VRAM [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9)MiB $(layout)"
      ELOG > "$R/engine-boot-$tag.log"
      log "[$tag allreduce] $(grep -aoE "Using \[[^]]*\] all-reduce backends[^.]*" "$R/engine-boot-$tag.log" | head -1)"
      grep -aE "PCIe IPC" "$R/engine-boot-$tag.log" | sed -E 's/^.*(PCIe IPC)/\1/' | sort -u | head -8 | cut -c1-220 | sed "s/^/[$tag pcie] /" | tee -a "$R/audit.log"
      return 0; fi
    log "[$tag] boot attempt pin=$kv FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-$tag-$kv.log" | tail -1 | cut -c1-220)"
    ELOG 2>/dev/null | grep -aiE "error|exception|unsupported|not supported|PCIe IPC|JointFailure" | head -6 | cut -c1-220 | sed "s/^/[$tag boot-err] /" | tee -a "$R/audit.log"
    teardown
  done; return 1; }
cap(){ log "[$1 cap $2] $(python3 $PR/kv_capacity_probe.py --url $U "${@:3}" 2>&1 | tail -1 | cut -c1-330)"; }
p1(){ local tag=$1 name=$2; shift 2
  python3 $PR/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$tag-$name.jsonl" > "$R/probe-$tag-$name.out" 2> "$R/probe-$tag-$name.err"
  if grep -aq RESULT "$R/probe-$tag-$name.out"; then grep -a RESULT "$R/probe-$tag-$name.out" | sed "s/^/[$tag $name] /" | cut -c1-260 | tee -a "$R/audit.log"
  else log "[$tag $name] PROBE FAILED: $(grep -a . "$R/probe-$tag-$name.out" "$R/probe-$tag-$name.err" 2>/dev/null | tail -1 | cut -c1-140)"; fi; }
ruler_dense(){ local T=$1
  timeout 3600 python3 $PR/fidelity_ladder.py --url $U --model qwen3.8-27b --corpus "$LADDER_CORPUS" --out "$R/dump-$T-dense.jsonl" --logprobs 20 --mode dense > "$R/score-$T-dense.out" 2>&1
  log "[$T bf16 dense] $(tail -1 "$R/score-$T-dense.out" | cut -c1-160)"
  python3 $PR/fidelity_compare.py --ref "$BF16_REF" --arm "$R/dump-$T-dense.jsonl" --label "$T" --json "$R/bf16-$T.json" 2>&1 | tee "$R/bf16-$T.txt" | grep -aE "overall top-1|corpus PPL|truncated KL" | cut -c1-200 | sed "s/^/[$T vs bf16 dense] /" | tee -a "$R/audit.log"; }
ruler_agentic(){ local T=$1
  timeout 3600 python3 $PR/agentic_ref.py score --url $U --model qwen3.8-27b --ids "$BF16_DIR/agentic-ids.jsonl" --out "$R/dump-$T-agentic.jsonl" > "$R/score-$T-agentic.out" 2>&1
  python3 $PR/fidelity_compare.py --ref "$BF16_DIR/dump-a1-agentic.jsonl" --arm "$R/dump-$T-agentic.jsonl" --label "AGENTIC-$T" --json "$R/bf16-$T-agentic.json" 2>&1 | tee "$R/bf16-$T-agentic.txt" | grep -aE "overall top-1|corpus PPL" | cut -c1-200 | sed "s/^/[$T vs bf16 agentic] /" | tee -a "$R/audit.log"; }
dfid(){ local T=$1 ctx=$2
  python3 $PR/decode_fidelity.py run --url $U --corpus "$FD/corpus.jsonl" --out "$R/dec-$T-ctx$ctx.jsonl" --chunks 20 --ctx "$ctx" --tokens 256 > "$R/dec-$T-ctx$ctx.out" 2>&1
  log "[$T decode ctx$ctx vs bf16] $(python3 $PR/decode_fidelity.py compare "$DREF/dec-bf16-ctx$ctx.jsonl" "$R/dec-$T-ctx$ctx.jsonl" 2>&1 | tail -1 | cut -c1-300)"; }
bat_decode(){ local T=$1
  p1 $T code-c1 --conc 1 --tokens 1024 --runs 2 --kind code
  p1 $T prose-c1 --conc 1 --tokens 1024 --runs 2 --kind prose
  p1 $T code-c8 --conc 8 --tokens 1024 --runs 2 --kind code
  p1 $T code-c16 --conc 16 --tokens 1024 --runs 1 --kind code
  log "[$T errors after c16] $(errs)"; }

teardown
# ---------- 0. ar-bench gate: vendored package vs flashinfer main, free cards ----------
IMG=$IMG BACKENDS=nccl,pcie,vend ROWS=1,8,10,16,20,40,80,160,320 R=$R TUNE=pcie-tune-free.json bash /srv/qwen5090/ar-bench.sh bench > "$R/arbench-gate.out" 2>&1
grep -aE "^ +[0-9]+ +[0-9]+ \||rows +bytes|MISMATCH|graph capture failed|OOM|vend configs|skipped" "$R/arbench-gate.out" | cut -c1-200 | sed 's/^/[arbench] /' | tee -a "$R/audit.log"
GATE=$(python3 - "$R" <<'EOF'
import glob, json, os, sys
fs = sorted(glob.glob(os.path.join(sys.argv[1], "arbench-bench-*.json")), key=os.path.getmtime)
if not fs: print("FAIL no json"); sys.exit()
d = json.load(open(fs[-1])); n = d["notes"]; bad = []
for k in ("mismatch", "graph_mismatch", "graph_fail", "oom"):
    for e in n.get(k, []):
        if e[0] == "vend": bad.append(f"{k}:{e[1]}")
slow = []
for r, cell in d["graph"].items():
    v, p = cell.get("vend"), cell.get("pcie")
    if v is None and int(r) <= 320: bad.append(f"no-vend-graph:{r}")
    if v is not None and p is not None and v > 1.15 * p: slow.append(f"{r}:{v:.1f}vs{p:.1f}")
err = n.get("graph_maxerr", {}).get("vend", {})
print(("FAIL " + " ".join(bad)) if bad else ("WARN slow " + " ".join(slow) if slow else "PASS"), "maxerr=" + str(max(err.values()) if err else "n/a"), fs[-1].split("/")[-1])
EOF
)
log "[arbench gate] $GATE"
case "$GATE" in FAIL*) log "GATE FAILED: the vendored kernel is not the measured kernel — no engine boot"; finish ABORTED; exit 1;; esac

# ---------- 1. PCIE-on ----------
wipe_l2
if boot_arm PCIE-on "$PINS" EXTRA_ENV_APPEND="-e VLLM_SM12X_PCIE_IPC_AR=1"; then
  if grep -aq "PCIe IPC all-reduce enabled" "$R/engine-boot-PCIE-on.log" && grep -aoE "Using \[[^]]*\] all-reduce backends" "$R/engine-boot-PCIE-on.log" | head -1 | grep -q PCIE_IPC; then
    log "[PCIE-on] path ENABLED"; ON=1
  else log "[PCIE-on] path NOT enabled — measured as the image-parity control (see [PCIE-on pcie] lines)"; ON=0; fi
  sleep 20
  bat_decode PCIE-on
  cap PCIE-on ttft8k --ctx 8000 --conc 1 --tokens 8 --seed 41
  cap PCIE-on ttft120k --ctx 120000 --conc 1 --tokens 8 --seed 43
  dfid PCIE-on 0; ruler_dense PCIE-on
  if [ "$ON" = 1 ]; then p1 PCIE-on deep30k --conc 1 --tokens 1024 --runs 2 --kind prose --ctx 30000; dfid PCIE-on 30000; ruler_agentic PCIE-on; fi
  log "[PCIE-on engine error-lines] $(errs)"; ELOG | grep -aE "PCIe IPC|JointFailure|pcie_ipc" | grep -av "enabled:\|prepared resolved" | sort | uniq -c | sort -rn | head -6 | cut -c1-220 | sed 's/^/[PCIE-on pcie-tail] /' | tee -a "$R/audit.log"
  teardown
else log "[PCIE-on] BOOT FAILED on every pin"; ON=0; fi

# ---------- 2. PCIE-off: same image, knob unset ----------
if [ "$ON" = 1 ]; then
  wipe_l2
  if boot_arm PCIE-off "$PINS"; then sleep 20; bat_decode PCIE-off; dfid PCIE-off 0; log "[PCIE-off engine error-lines] $(errs)"; teardown
  else log "[PCIE-off] BOOT FAILED on every pin"; fi
fi
grep -aE "arbench|BOOT OK|BOOT FAILED|ENABLED|NOT enabled|pcie\]|allreduce\]|cap |RESULT|PROBE FAILED|error-lines|errors after|vs bf16|decode ctx|pcie-tail|GATE" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
finish DONE
