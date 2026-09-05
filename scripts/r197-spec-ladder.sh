#!/usr/bin/env bash
# R197 (2026-09-05, user: "re measure the best number of dflash2 tokens to predict, we're at 9, let's try again 6/7/8" + "try swapping
# dflash2 with MTP, also try 3/4"): speculative-length ladder on the daily route. The last ns read (R173/R173b/R173c, 2026-09-04) put ns7
# at +12.5 % c8 paired and −7 % prose c1, then RETRACTED it on decode numerics (ns7 2× farther from bf16 at 30K: 0.0105 vs 0.0052). R193d
# has since shown a per-boot numerics draw of that very size (~0.004–0.006 at 30K, the runtime Triton autotune), so the retraction rests
# on a reading the protocol now voids. num_speculative_tokens is NOT a compile factor (SpeculativeConfig.compute_hash: method, aux
# hidden states, draft-model hash, layer ids only), so every DFlash arm loads ONE artifact and, with VLLM_TRITON_FORCE_FIRST_CONFIG=1,
# the R193d/R193e protocol holds across the whole ladder: a ctx0/30K difference between D6/D7/D8 and D9 is the draft length's own
# numerics (verify batch q=ns+1), nothing else. The MTP arms (the checkpoint's own Qwen3.5 MTP head, 15 mtp.* tensors in the RedHat
# safetensors; R157 measured MTP ns4 on nvfp4 KV: c1 −41 %, c8 +8–20 % vs DFlash2-fp8, 2026-09-02) compile their own artifact
# (uses_aux_hidden_states differs), so M3 vs M4 is same-artifact and M vs D / M vs bf16 carries the fresh-artifact caveat (R190c).
# Arms, one boot each on the daily route (EXP=1 :8029, SEQS 16, PCIE_IPC=1, BSS=1, pin 13.98 → 13.5 GB, nvfp4 KV, image = daily):
#   D9 (control, first) → D6 → D7 → D8 → D9b (boot-to-boot floor: same artifact + forced first config ⇒ bitwise, else protocol failure)
#   → M3 → M4 (SPEC_METHOD=mtp, launch-daily.sh R197 passthrough, no /draft mount).
# Per arm: pool/min_free/artifact; decode_fidelity ctx0 + 30K vs bf16 (r173c) and later vs D9; decode_ss code c1 ×3, prose c1 ×2,
# prose 30K ×2, code c8 ×2, code c16 ×2. Steps/s = t/s ÷ (1 + ns × accept_per_draft) with the ARM's ns (the README convention used 9).
# No tool-eval: spec decoding is distribution-lossless; the T>0 instrument's ±1–2 spread cannot rank draft lengths. ~2 h. Restores the
# daily at the end unless another unit is queued (r196 re-queued behind this one, user: "move the new model audition after these").
#   unit: sudo systemd-run --unit=r197-spec-ladder --collect -p User=adrienbrault -p RuntimeMaxSec=43200 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r197-spec-ladder bash -c '. /srv/qwen5090/lib/gpu-queue.sh; exec bash /srv/qwen5090/r197-spec-ladder.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-05-r197-spec-ladder; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
IMG=${IMG:-vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc-bsshash}
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily.sh; L2=/srv/qwen5090/eval-l2; PR=/srv/qwen5090/probes
FD=/srv/qwen5090/results/2026-08-23-fidelity; DREF=/srv/qwen5090/results/2026-09-04-r173c-bf16-decode
PINS="13980000000 13500000000"
PIN_ENV="EXTRA_ENV_APPEND=-e VLLM_TRITON_FORCE_FIRST_CONFIG=1${EXTRA_ARM_ENV:+ $EXTRA_ARM_ENV}"
# r193d P1/P2 + r193e P3/B3 artifact (forced first config, BSS invisible to the hash since 0147)
EXPECT_HASHES=7ff095820d8f,859caf58c368,e70c5bc3970c,
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: image $IMG missing"; exit 3; }
[ -f "$FD/corpus.jsonl" ] && [ -f "$DREF/dec-bf16-ctx30000.jsonl" ] && [ -f "$DREF/dec-bf16-ctx0.jsonl" ] || { log "ABORT: reference files missing"; exit 3; }
grep -q "R197 EXP-only passthrough" "$CAND" || { log "ABORT: launch-daily.sh lacks the R197 SPEC_METHOD passthrough"; exit 3; }
grep -qF 'SPEC_METHOD_=${SPEC_METHOD:-dflash}' "$CAND" || { log "ABORT: launch-daily.sh SPEC_METHOD wiring missing"; exit 3; }
for t in decode_ss.py decode_fidelity.py; do [ -f "$PR/$t" ] || { log "ABORT: $PR/$t missing"; exit 3; }; done
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R197 $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R197 start (lock held): speculative-length ladder on $IMG (PCIE_IPC=1 BSS=1 forced-first-config) — D9 D6 D7 D8 D9b M3 M4 ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; }
ELOG(){ sudo docker logs vllm-exp 2>&1; }
errs(){ ELOG | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError\|JointFailure'; }
# boot_arm TAG METHOD NS
boot_arm(){ local tag=$1 method=$2 ns=$3 kv rc saved loaded hashes
  for kv in $PINS; do
    env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 KV_BYTES=$kv PCIE_IPC=${PCIE_IPC_ARM:-1} BSS=1 CAND_IMG=$IMG SPEC_METHOD=$method SPEC_NS=$ns "$PIN_ENV" bash $CAND > "$R/boot-$tag-$kv.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
      ELOG > "$R/engine-boot-$tag.log"
      saved=$(grep -ac "saved AOT compiled function" "$R/engine-boot-$tag.log"); loaded=$(grep -ac "Directly load AOT" "$R/engine-boot-$tag.log")
      hashes=$(grep -aoE "torch_aot_compile/[0-9a-f]{12}" "$R/engine-boot-$tag.log" | cut -d/ -f2 | sort -u | tr '\n' ',')
      log "[$tag] BOOT OK method=$method ns=$ns pin=$kv pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9) min_free=$(grep -aoE 'min free VRAM [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9)MiB pcie=$(grep -ac 'PCIe IPC all-reduce enabled' "$R/engine-boot-$tag.log") bss_lines=$(grep -ac 'Batch-sharded sampling enabled' "$R/engine-boot-$tag.log") spec=$(grep -aoE 'num_speculative_tokens[^,]{0,6}' "$R/engine-boot-$tag.log" | head -1) aot_saved=$saved aot_loaded=$loaded compile_hashes=$hashes $([ "$hashes" = "$EXPECT_HASHES" ] && echo "= r193d P1 artifact" || echo "≠ r193d P1 artifact")"
      echo "$tag $method $ns $hashes $saved $loaded" >> "$R/arms.txt"
      return 0; fi
    log "[$tag] boot attempt pin=$kv FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-$tag-$kv.log" | tail -1 | cut -c1-220)"
    ELOG 2>/dev/null | grep -aiE "error|exception|speculat|mtp|draft" | grep -av "INFO" | head -6 | cut -c1-220 | sed "s/^/[$tag boot-err] /" | tee -a "$R/audit.log"
    teardown
  done; return 1; }
p1(){ local tag=$1 name=$2; shift 2
  python3 $PR/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$tag-$name.jsonl" > "$R/probe-$tag-$name.out" 2> "$R/probe-$tag-$name.err"
  if grep -aq RESULT "$R/probe-$tag-$name.out"; then grep -a RESULT "$R/probe-$tag-$name.out" | sed "s/^/[$tag $name] /" | cut -c1-300 | tee -a "$R/audit.log"
  else log "[$tag $name] PROBE FAILED: $(grep -a . "$R/probe-$tag-$name.out" "$R/probe-$tag-$name.err" 2>/dev/null | tail -1 | cut -c1-140)"; fi; }
dfid(){ local T=$1 ctx=$2
  python3 $PR/decode_fidelity.py run --url $U --corpus "$FD/corpus.jsonl" --out "$R/dec-$T-ctx$ctx.jsonl" --chunks 20 --ctx "$ctx" --tokens 256 > "$R/dec-$T-ctx$ctx.out" 2>&1
  log "[$T decode ctx$ctx vs bf16] $(python3 $PR/decode_fidelity.py compare "$DREF/dec-bf16-ctx$ctx.jsonl" "$R/dec-$T-ctx$ctx.jsonl" 2>&1 | tail -1 | cut -c1-300)"; }
# arm TAG METHOD NS
arm(){ local tag=$1 method=$2 ns=$3
  teardown; wipe_l2
  if boot_arm "$tag" "$method" "$ns"; then
    sleep 20
    dfid "$tag" 0
    dfid "$tag" 30000
    p1 $tag code-c1 --conc 1 --tokens 1024 --runs 3 --kind code
    p1 $tag prose-c1 --conc 1 --tokens 1024 --runs 2 --kind prose
    p1 $tag prose-c1-30k --conc 1 --tokens 1024 --runs 2 --kind prose --ctx 30000
    p1 $tag code-c8 --conc 8 --tokens 1024 --runs 2 --kind code
    p1 $tag prose-c8 --conc 8 --tokens 1024 --runs 2 --kind prose
    p1 $tag code-c16 --conc 16 --tokens 1024 --runs 2 --kind code
    log "[$tag engine error-lines] $(errs)  preemptions: $(curl -s -m 5 $U/metrics | grep -a '^vllm:num_preemptions_total' | awk '{print $NF}')"
  else log "[$tag] BOOT FAILED on every pin"; fi; }
# ARMS override (R197 rerun 2026-09-05 12:2x UTC): the first D6 boot failed on the launcher's ns9-only 1,584-token block assert (ns6 sizes the
# block to 1,536; launcher now asserts the exact value for dflash ns9 only). A rerun unit re-does the failed arm(s) into the SAME results dir,
# then the analysis below re-runs over every arm present; audit.log keeps both passes.
#   rerun unit: sudo systemd-run --unit=r197-d6-rerun --collect -p User=adrienbrault -p RuntimeMaxSec=43200 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r197-d6-rerun -E ARMS="D6 dflash 6" bash -c '. /srv/qwen5090/lib/gpu-queue.sh; while systemctl is-active -q r197-spec-ladder; do sleep 30; done; exec bash /srv/qwen5090/r197-spec-ladder.sh'
# 12:3x UTC restart (user: "please measure prose c8"): prose-c8 added to every arm; the first pass (D9 complete, D7 mid-boot) was stopped and
# the remaining arms re-queued as unit r197-spec-ladder2 with ARMS="D6 dflash 6;D7 dflash 7;D8 dflash 8;D9b dflash 9;M3 mtp 3;M4 mtp 4".
# D9's prose-c8 reading is D9b's (same configuration, same artifact); D9 itself keeps its first-pass rows.
# 13:03 UTC: M3 booted (pool 1,309,368) but the launcher's 1.1M EXP pool ceiling rejected it; launcher band is now spec-aware.
# 13:15 UTC: M4 then failed the pcie_ipc assert: patch 0138's all-reduce supports only the sequential DFlash drafter ("PCIe IPC all-reduce
# disabled: rank prerequisites [('only the sequential DFlash drafter is supported', ...)]"), so every MTP arm runs the CUSTOM all-reduce. Unit
# r197-spec-ladder4 = PCIE_IPC_ARM=0 ARMS="M3 mtp 3;M4 mtp 4;M4b mtp 4;D9c dflash 9": M4b = MTP boot-to-boot floor, D9c = ns9 DFlash without
# pcie_ipc = the same-day all-reduce pairing (M-vs-D9c isolates the spec method; M-vs-D9 is what a daily swap would actually deliver).
#   unit: sudo systemd-run --unit=r197-spec-ladder4 --collect -p User=adrienbrault -p RuntimeMaxSec=43200 -p TimeoutStopSec=900 -E GPU_QUEUE_NAME=r197-spec-ladder4 \
#         -E PCIE_IPC_ARM=0 -E ARMS="M3 mtp 3;M4 mtp 4;M4b mtp 4;D9c dflash 9" bash -c '. /srv/qwen5090/lib/gpu-queue.sh; while systemctl is-active -q r197-spec-ladder2 || systemctl is-active -q r197-spec-ladder3; do sleep 30; done; exec bash /srv/qwen5090/r197-spec-ladder.sh'
# 13:5x UTC (user: "make an image where MTP works"): patch 0148 (codex, BRIEF30) admits the MTP drafter to the 0138 PCIe IPC all-reduce behind
#   VLLM_SM12X_PCIE_IPC_MTP=1; image ...-pcieipc-bsshash-mtppcie. IMG and EXTRA_ARM_ENV are now overridable so ladder5 runs the MTP arms on it with
#   PCIE_IPC_ARM=1; M3p/M4p vs M3/M4 (CUSTOM) must be bitwise when the hash sets match and loaded (R185: pcie_ipc ≡ CUSTOM on every paired ruler).
#   unit: sudo systemd-run --unit=r197-spec-ladder5 --collect -p User=adrienbrault -p RuntimeMaxSec=43200 -p TimeoutStopSec=900 -E GPU_QUEUE_NAME=r197-spec-ladder5 \
#         -E IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc-bsshash-mtppcie -E EXTRA_ARM_ENV="-e VLLM_SM12X_PCIE_IPC_MTP=1" -E ARMS="M3p mtp 3;M4p mtp 4" \
#         bash -c '. /srv/qwen5090/lib/gpu-queue.sh; while systemctl is-active -q r197-spec-ladder4; do sleep 30; done; exec bash /srv/qwen5090/r197-spec-ladder.sh'
# 13:5x UTC (user: "also try d6", D6 already measured in ladder2 → read as MTP ns6): unit r197-spec-ladder4b, ARMS="M6 mtp 6", PCIE_IPC_ARM=0, behind ladder4.
# 13:0x UTC (user: "also try dflash 10/11"): unit r197-spec-ladder3 with ARMS="D10 dflash 10;D11 dflash 11" queued behind ladder2, same results dir.
ARMS="${ARMS:-D9 dflash 9;D6 dflash 6;D7 dflash 7;D8 dflash 8;D9b dflash 9;M3 mtp 3;M4 mtp 4}"
log "arms: $ARMS"
echo "$ARMS" | tr ';' '\n' | while read -r t m n; do [ -n "$t" ] && arm "$t" "$m" "$n"; done
# numerics: every arm vs D9 (same artifact for the D arms = the draft length's own numerics; M arms = fresh artifact, caveat)
for T in D6 D7 D8 D9b D10 D11 D9c M3 M4 M4b M6 M3p M4p M6p; do for ctx in 0 30000; do
  [ -f "$R/dec-D9-ctx$ctx.jsonl" ] && [ -f "$R/dec-$T-ctx$ctx.jsonl" ] && log "[$T vs D9 decode ctx$ctx] $(python3 $PR/decode_fidelity.py compare "$R/dec-D9-ctx$ctx.jsonl" "$R/dec-$T-ctx$ctx.jsonl" 2>&1 | tail -1 | cut -c1-300)"
done; done
# pcie_ipc MTP arms (ladder5, image mtppcie) vs their CUSTOM twins: bitwise expected when the hash sets match and loaded
for P in M3 M4 M6; do for ctx in 0 30000; do [ -f "$R/dec-$P-ctx$ctx.jsonl" ] && [ -f "$R/dec-${P}p-ctx$ctx.jsonl" ] && log "[${P}p vs $P decode ctx$ctx] $(python3 $PR/decode_fidelity.py compare "$R/dec-$P-ctx$ctx.jsonl" "$R/dec-${P}p-ctx$ctx.jsonl" 2>&1 | tail -1 | cut -c1-300)"; done; done
for ctx in 0 30000; do [ -f "$R/dec-M4-ctx$ctx.jsonl" ] && [ -f "$R/dec-M4b-ctx$ctx.jsonl" ] && log "[M4b vs M4 decode ctx$ctx] $(python3 $PR/decode_fidelity.py compare "$R/dec-M4-ctx$ctx.jsonl" "$R/dec-M4b-ctx$ctx.jsonl" 2>&1 | tail -1 | cut -c1-300)"; [ -f "$R/dec-M3-ctx$ctx.jsonl" ] && [ -f "$R/dec-M4-ctx$ctx.jsonl" ] && log "[M4 vs M3 decode ctx$ctx] $(python3 $PR/decode_fidelity.py compare "$R/dec-M3-ctx$ctx.jsonl" "$R/dec-M4-ctx$ctx.jsonl" 2>&1 | tail -1 | cut -c1-300)"; done
# artifact check: all D arms one hash set, D6/D7/D8/D9b loaded (saved=0); M3 saved or loaded, M4 loaded
awk '{print $1, $2, $3, "hashes="$4, "saved="$5, "loaded="$6}' "$R/arms.txt" | sed 's/^/[artifact] /' | tee -a "$R/audit.log"
DH=$(awk '$2=="dflash"{print $4}' "$R/arms.txt" | sort -u | wc -l); DS=$(awk '$2=="dflash" && $1!="D9" && $5>0' "$R/arms.txt" | wc -l)
[ "$DH" = 1 ] && [ "$DS" = 0 ] && log "SAME-ARTIFACT: every DFlash arm loaded one compile artifact — D-vs-D9 deltas are the draft length's numerics" || log "CONFOUNDED: DFlash arms span $DH hash sets / $DS arms after D9 saved an artifact — D-vs-D9 numerics void, speed stands"
# steps/s table: t/s ÷ (1 + ns × accept), ns per arm
python3 - "$R/audit.log" > "$R/steps.txt" <<'PY'
import json,re,sys
ns={'D9':9,'D6':6,'D7':7,'D8':8,'D9b':9,'M3':3,'M4':4}; rows={}
for l in open(sys.argv[1]):
    m=re.match(r'\[(\S+) (\S+)\] RESULT (\{.*\})',l.strip())
    if not m: continue
    tag,name,js=m.groups()
    try: d=json.loads(js)
    except Exception: continue
    a=d.get('accept_per_draft_median'); t=d.get('ss_agg_tps_median')
    if a is None or t is None: continue
    rows[(tag,name)]=(t,a,t/(1+ns[tag]*a))
names=['code-c1','prose-c1','prose-c1-30k','code-c8','prose-c8','code-c16']
print('arm   ns  '+'  '.join(f'{n:>26}' for n in names)+'   (t/s @accept -> steps/s)')
for tag in ns:
    cells=[]
    for n in names:
        r=rows.get((tag,n)); cells.append(f'{r[0]:7.1f} @{r[1]:.3f} -> {r[2]:6.1f}' if r else ' '*26)
    print(f'{tag:<5} {ns[tag]:>2}  '+'  '.join(f'{c:>26}' for c in cells))
PY
sed 's/^/[steps] /' "$R/steps.txt" | tee -a "$R/audit.log"
grep -aE "BOOT OK|BOOT FAILED|boot-err|RESULT|PROBE FAILED|decode ctx|vs bf16|error-lines|artifact\]|SAME-ARTIFACT|CONFOUNDED|steps\]" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
finish DONE
