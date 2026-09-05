#!/usr/bin/env bash
# R199 step 0 (2026-09-05, backlog "Qwopus3.8-27B-Flash audition", user request): fetch the three checkpoints. No GPU; runs beside the daily.
#   Jackrong/Qwopus3.8-27B-Flash            bf16 SFT fine-tune, 55.6 GB (the fidelity reference for its quants)
#   Shiftedx/Qwopus3.8-27B-Flash-NVFP4-MTP  ModelOpt NVFP4 W4A4, native MTP head kept, 19.7 GB (like-for-like arm vs the RedHat daily)
#   sojufx/Qwopus3.8-27B-Flash-NVFP4        ModelOpt mixed NVFP4/FP8, 23.8 GB (fidelity arm)
# Idempotent (resumes). Pins each repo to the revision seen on 2026-09-05 so a later push cannot change what was measured.
#   sudo systemd-run --unit=r199-fetch --collect -p User=adrienbrault -p RuntimeMaxSec=36000 bash /srv/qwen5090/r199-fetch-qwopus.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-05-r199-qwopus; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
log "=== R199 fetch start ==="; df -h /srv/qwen5090 | tail -1 | tee -a "$R/audit.log"
python3 - <<'PY' 2>&1 | grep -vE "^\s*$|it/s|Fetching" | tail -30 | tee -a "$R/audit.log"
from huggingface_hub import snapshot_download
for repo, rev, dest in [
    ("Jackrong/Qwopus3.8-27B-Flash",            "072b8c27", "/srv/qwen5090/models/qwopus-27b-flash-bf16"),
    ("Shiftedx/Qwopus3.8-27B-Flash-NVFP4-MTP",  "e46dcfbe", "/srv/qwen5090/models/qwopus-27b-flash-shiftedx-nvfp4"),
    ("sojufx/Qwopus3.8-27B-Flash-NVFP4",        "bdd5036f", "/srv/qwen5090/models/qwopus-27b-flash-sojufx-nvfp4"),
]:
    p = snapshot_download(repo_id=repo, revision=rev, local_dir=dest, max_workers=8)
    print("snapshot:", repo, rev, "->", p, flush=True)
PY
for d in qwopus-27b-flash-bf16 qwopus-27b-flash-shiftedx-nvfp4 qwopus-27b-flash-sojufx-nvfp4; do
  D=/srv/qwen5090/models/$d; n=$(ls "$D"/*.safetensors 2>/dev/null | wc -l); idx=$(python3 -c "import json;print(len(set(json.load(open('$D/model.safetensors.index.json'))['weight_map'].values())))" 2>/dev/null || echo "?")
  log "[$d] size=$(du -sh "$D" 2>/dev/null | cut -f1) safetensors=$n index_shards=$idx $([ "$n" = "$idx" ] && echo COMPLETE || echo INCOMPLETE)"
done
log "=== R199 fetch DONE ==="
