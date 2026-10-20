#!/usr/bin/env bash
# Clean-room build of the public 27B repo's served image (vllm-qwen38:v0310-r27c-A_S-L1a-ubplan) on flan, in a separate
# docker-in-docker daemon with its own empty storage, so the build sees only the public tree and freshly pulled bases and
# can never retag the host's served images. 2026-10-19, user: "yes please improve" (one-command build + clean-room check).
#   ssh flan 'bash -s' < flan/public-27b-cleanroom.sh -- <public-tree.tgz on flan> [VLLM_SOURCE_SHA256]
# Without the sha argument it only fetches the pinned vLLM source archive, prints its sha256 and checks its csrc/ and
# cmake/patches/ against a git checkout of the v0.31.0 tag; with it, it runs scripts/build-v031-image.sh end to end.
# Results: /srv/qwen5090/results/public-27b-cleanroom-<stamp>/ (logs, build-logs/v0310/, content report).
# Environment (2026-10-20, BA4 publish while the user served the 27B daily: "don't touch flan gpu for now"):
#   BA4=1        also build the BA4 layer (BA4=1 scripts/build-v031-image.sh); the exported cleanroom tag is then the BA4 image,
#                and its inventory plus every vllm_ba4 and /opt/l1a-ba4 file is compared with the host's served BA4 image
#                (containers on runc, --network=none, no GPU device).
#   IDENTITY=0   skip the host identity probe (it runs on the NVIDIA runtime with --gpus all; CUDA_VISIBLE_DEVICES is empty,
#                but it still opens the driver).
#   CPUS=<n>     cap the docker-in-docker daemon (pulls, builds, the BA4 CPU tests) at n CPUs so a serving engine keeps cores.
#   GHCR_TAG=<ref>  tag the exported image under that name on the host (never pushes).
set -euo pipefail
[ "${1:-}" = -- ] && shift
SRC=${1:?public tree tarball on flan}; SHA=${2:-}
BA4=${BA4:-0}; IDENTITY=${IDENTITY:-1}; CPUS=${CPUS:-}; GHCR_TAG=${GHCR_TAG:-}
case "$BA4$IDENTITY" in [01][01]) ;; *) echo "BA4 and IDENTITY must be 0 or 1"; exit 2;; esac
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=/srv/qwen5090/results/public-27b-cleanroom-$STAMP
WORK=/srv/qwen5090/cleanroom-27b
DIND=cleanroom-27b-dind
mkdir -p "$OUT"; exec > >(tee "$OUT/run.log") 2>&1
echo "== $STAMP src=$SRC sha=${SHA:-<compute>} BA4=$BA4 IDENTITY=$IDENTITY CPUS=${CPUS:-all} GHCR_TAG=${GHCR_TAG:-none}"
sha256sum "$SRC" | tee "$OUT/source-tarball.sha256"
sudo -n docker rm -f "$DIND" >/dev/null 2>&1 || true
trap 'sudo -n docker rm -f "$DIND" >/dev/null 2>&1 || true; sudo -n chown -R "$(id -u):$(id -g)" "$OUT" 2>/dev/null || true' EXIT
sudo -n mkdir -p "$WORK/docker"; sudo -n rm -rf "$WORK/repo"; sudo -n mkdir -p "$WORK/repo"
sudo -n tar -xzf "$SRC" -C "$WORK/repo"
sudo -n docker pull -q docker:27-dind >/dev/null
sudo -n docker run -d --name "$DIND" --privileged -e DOCKER_TLS_CERTDIR= ${CPUS:+--cpus "$CPUS"} \
  -v "$WORK/docker:/var/lib/docker" -v "$WORK/repo:/repo" -v "$OUT:/out" docker:27-dind >/dev/null
for i in $(seq 1 60); do sudo -n docker exec "$DIND" docker info >/dev/null 2>&1 && break; sleep 2; done
sudo -n docker exec "$DIND" docker info --format 'inner daemon {{.ServerVersion}}, images {{.Images}}'
sudo -n docker exec "$DIND" apk add -q --no-cache bash python3 git curl coreutils tar patch diffutils
if [ -z "$SHA" ]; then
  sudo -n docker exec "$DIND" bash -euo pipefail -c '
    C=$(grep -oE "VLLM_COMMIT=[0-9a-f]{40}" /repo/scripts/build-v031-image.sh | head -1 | cut -d= -f2)
    echo "pinned commit $C"
    cd /tmp && curl -fsSL -o vllm.tgz "https://codeload.github.com/vllm-project/vllm/tar.gz/$C"
    sha256sum vllm.tgz | tee /out/vllm-source.sha256
    mkdir -p a && tar -xzf vllm.tgz -C a --strip-components=1
    git clone -q --depth 1 --branch v0.31.0 https://github.com/vllm-project/vllm.git g
    [ "$(git -C g rev-parse HEAD)" = "$C" ] && echo "tag v0.31.0 == $C"
    diff -r a/csrc g/csrc && diff -r a/cmake/patches g/cmake/patches && echo "archive csrc/ and cmake/patches/ == git tag"'
else
  rc=0
  sudo -n docker exec -e VLLM_SOURCE_SHA256="$SHA" -e NO_CACHE=1 -e BA4="$BA4" -w /repo "$DIND" bash scripts/build-v031-image.sh || rc=$?
  sudo -n docker exec "$DIND" sh -c 'cp -r /repo/build-logs /out/ 2>/dev/null || true'
  echo "build rc=$rc"
  if [ "$rc" = 0 ]; then
    # The inner daemon has no NVIDIA runtime, so the identity check (CHECK=1) runs on the host. Only a cleanroom tag is
    # exported: loading the launcher's tag would retag the host's served image.
    TAG=vllm-qwen38:v0310-r27c-A_S-L1a-ubplan; [ "$BA4" = 0 ] || TAG=vllm-qwen38:v0310-r27c-A_S-BA4-L1a-ubplan
    CR=$TAG-cleanroom-$STAMP
    sudo -n docker exec "$DIND" docker tag "$TAG" "$CR"
    sudo -n docker exec "$DIND" docker save "$CR" | sudo -n docker load
    sudo -n docker image inspect --format '{{.Id}}' "$CR" | tee "$OUT/cleanroom-image-id.txt"
    if [ "$IDENTITY" = 1 ]; then
      code=$(sed -n "/^IDENTITY_CODE=\$(cat <<'PY'\$/,/^PY\$/p" "$WORK/repo/scripts/build-v031-image.sh" | sed '1d;$d')
      sudo -n docker run --rm --network=none --runtime nvidia --gpus all -e CUDA_VISIBLE_DEVICES= --entrypoint python3 \
        "$CR" -c "$code" 2>&1 | tee "$OUT/identity.log"
      echo "identity rc=${PIPESTATUS[0]} image=$CR"
    else echo "identity probe skipped (IDENTITY=0)"; fi
    if [ "$BA4" = 1 ]; then
      # Content identity with the image the host serves, on runc: the build's own manifest tool over vllm, flashinfer and the
      # overlay, then sha256 of every vllm_ba4 and /opt/l1a-ba4 file except bytecode (the build's CPU tests write some).
      SERVED=vllm-qwen38:v0310-r27c-A_S-BA4-L1a-ubplan
      sudo -n docker image inspect --format '{{.Id}}' "$SERVED" | tee "$OUT/served-image-id.txt"
      for side in served:$SERVED cleanroom:$CR; do
        img=${side#*:}; name=${side%%:*}
        sudo -n docker run --rm -i --runtime runc --network=none -e NVIDIA_VISIBLE_DEVICES=void --entrypoint python3 "$img" - generate \
          < "$WORK/repo/scripts/v031_manifest.py" > "$OUT/$name-ba4-manifest.txt"
        sudo -n docker run --rm --runtime runc --network=none -e NVIDIA_VISIBLE_DEVICES=void --entrypoint bash "$img" -c \
          'cd / && find usr/local/lib/python3.12/dist-packages/vllm_ba4 opt/l1a-ba4 -type f ! -name "*.pyc" -print0 | sort -z | xargs -0 sha256sum' \
          > "$OUT/$name-ba4-files.txt"
      done
      crc=0
      python3 "$WORK/repo/scripts/v031_manifest.py" compare "$OUT/served-ba4-manifest.txt" "$OUT/cleanroom-ba4-manifest.txt" \
        > "$OUT/served-vs-cleanroom-report.txt" 2>&1 || crc=$?
      echo "inventory compare rc=$crc ($(wc -l < "$OUT/served-ba4-manifest.txt") / $(wc -l < "$OUT/cleanroom-ba4-manifest.txt") entries)"
      if diff "$OUT/served-ba4-files.txt" "$OUT/cleanroom-ba4-files.txt" > "$OUT/ba4-files.diff"; then
        echo "BA4 package and /opt/l1a-ba4 files identical ($(wc -l < "$OUT/served-ba4-files.txt") files)"
      else echo "BA4 files DIFFER: $OUT/ba4-files.diff"; fi
    fi
    if [ -n "$GHCR_TAG" ]; then sudo -n docker tag "$CR" "$GHCR_TAG"; echo "tagged $GHCR_TAG = $(sudo -n docker image inspect --format '{{.Id}}' "$GHCR_TAG")"; fi
  fi
fi
sudo -n docker rm -f "$DIND" >/dev/null
sudo -n chown -R "$(id -u):$(id -g)" "$OUT"
echo "== done $OUT (inner storage kept at $WORK/docker until the image is published; remove with sudo rm -rf $WORK)"
