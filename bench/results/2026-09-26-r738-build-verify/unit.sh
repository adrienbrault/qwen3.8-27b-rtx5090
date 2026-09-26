#!/usr/bin/env bash
# R738 (2026-09-25): verify the public 27B repo's scripts/build-served-image.sh (issue #2) by building the served image from a
# clean `git archive` of the committed tree, with NO_CACHE=1, under its own repository name (vllm-qwen38-verify) so the
# served vllm-qwen38:* tags are never touched. Then compare the result with the served image file by file (every non-binary
# file in the patched packages, and the installed distribution versions); compiled .so files are expected to differ.
#
# Staging (operator, from the Mac):
#   git -C ~/Developer/ai/qwen3.8-27b-rtx5090 archive --format=tar HEAD | ssh flan 'rm -rf /srv/qwen5090/build-verify-27b/src && mkdir -p /srv/qwen5090/build-verify-27b/src && tar -x -C /srv/qwen5090/build-verify-27b/src'
#   echo <sha> | ssh flan 'cat > /srv/qwen5090/build-verify-27b/src/.commit'
# Launch: sudo systemd-run --unit=r738-build-verify-27b --collect -p RuntimeMaxSec=43200 -p Environment=HOME=$HOME_OF_THE_BUILD_USER /usr/bin/bash /srv/qwen5090/r738-build-verify-27b.sh
#
# Takes the GPU-exclusive lock WITHOUT registering in gpu-queue/, so the chain's last experiment still restores the daily:
# the build (CPU, niced) runs beside the daily and never beside a benchmark. The GPU is not used (CHECK=1 hides it).
# The base image is already local (the daily's own base), so its 8.65 GB pull is not in the timing.
set -uo pipefail
SRC=/srv/qwen5090/build-verify-27b/src
REPO=vllm-qwen38-verify
SERVED=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc-bsshash-mtppcie-mtpcache-eagleshift
VERIFY=$REPO:${SERVED#*:}
R=/srv/qwen5090/results/2026-09-25-r738-build-verify-27b
mkdir -p "$R"; exec > >(tee -a "$R/unit.log") 2>&1
log(){ echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
export HOME=${HOME:-$HOME_OF_THE_BUILD_USER}   # the docker-buildx plugin lives in ~/.docker/cli-plugins

[ -f "$SRC/scripts/build-served-image.sh" ] && [ -s "$SRC/.commit" ] || { log "FATAL: stage $SRC first (see header)"; exit 2; }
log "R738 source commit $(cat "$SRC/.commit")"
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "waiting for the GPU-exclusive lock (keeps the build off every benchmark)"; flock 9; }
log "lock held"
if docker image inspect "$VERIFY" >/dev/null 2>&1; then log "FATAL: $VERIFY already exists; remove the $REPO tags first"; exit 2; fi

snap(){ log "--- disk $1"; df -B1 --output=used,avail / | tail -1 | sed 's/^/df-bytes used,avail: /'; docker system df; docker buildx du 2>/dev/null | tail -2; }
snap before > "$R/disk-before.txt"; cat "$R/disk-before.txt"
t0=$(date +%s)
( cd "$SRC" && NO_CACHE=1 CHECK=1 IMAGE_REPO=$REPO LOG_DIR="$R/build-logs" nice -n 10 /usr/bin/time -v bash scripts/build-served-image.sh ) > "$R/build.log" 2>&1
rc=$?; t1=$(date +%s)
tail -40 "$R/build.log"
log "build rc=$rc wall $(( (t1 - t0) / 60 )) min $(( (t1 - t0) % 60 )) s"
snap after > "$R/disk-after.txt"; cat "$R/disk-after.txt"
[ $rc = 0 ] || { log "VERDICT: BUILD FAILED rc=$rc (build.log, build-logs/)"; exit 1; }

# Per-layer wall from the build log's own timestamps.
grep -E "^[0-9T:-]+Z (--- build|build .* OK)" "$R/build.log" > "$R/layer-times.txt" || true

MANIFEST='
import hashlib, importlib.metadata as md, os, sys
for d in sorted(md.distributions(), key=lambda d: (d.metadata["Name"] or "").lower()):
    print("dist", (d.metadata["Name"] or "?").lower(), d.version)
import vllm, flashinfer
sp = os.path.dirname(os.path.dirname(vllm.__file__))
roots = [os.path.join(sp, p) for p in ("vllm", "flashinfer", "pcie_ipc_ar21", "arctic_inference")] + ["/opt/prs-markers", "/opt/vllm-sm12x"]
for root in roots:
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(x for x in dn if x not in ("__pycache__", "build") and not x.startswith("CMakeFiles"))
        for f in sorted(fn):
            p = os.path.join(dp, f)
            if f.endswith((".so", ".pyc", ".o", ".a", ".cubin", ".fatbin")) or os.path.islink(p): continue
            print("file", p.replace(sp, "SP"), hashlib.md5(open(p, "rb").read()).hexdigest())
'
for tag in "$SERVED" "$VERIFY"; do
  n=$([ "$tag" = "$SERVED" ] && echo served || echo verify)
  docker run --rm --network=none --entrypoint python3 "$tag" -c "$MANIFEST" > "$R/manifest-$n.txt" 2> "$R/manifest-$n.err" || log "manifest $n rc=$?"
done
diff "$R/manifest-served.txt" "$R/manifest-verify.txt" > "$R/manifest.diff"; drc=$?
log "manifest lines served $(wc -l < "$R/manifest-served.txt") verify $(wc -l < "$R/manifest-verify.txt"); diff lines $(grep -c '^[<>]' "$R/manifest.diff")"
head -60 "$R/manifest.diff"
docker image ls --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' | grep -E "^($REPO|vllm-qwen38):" > "$R/images.txt"
if [ $drc = 0 ] && [ -s "$R/manifest-served.txt" ]; then log "VERDICT: BUILD OK, IDENTICAL to the served image in every non-binary file and distribution version"
else log "VERDICT: BUILD OK, DIFFERS from the served image (manifest.diff)"; fi
