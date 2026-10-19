#!/usr/bin/env bash
# Clean-room build of the public 27B repo's served image (vllm-qwen38:v0310-r27c-A_S-L1a-ubplan) on flan, in a separate
# docker-in-docker daemon with its own empty storage, so the build sees only the public tree and freshly pulled bases and
# can never retag the host's served images. 2026-10-19, user: "yes please improve" (one-command build + clean-room check).
#   ssh flan 'bash -s' < flan/public-27b-cleanroom.sh -- <public-tree.tgz on flan> [VLLM_SOURCE_SHA256]
# Without the sha argument it only fetches the pinned vLLM source archive, prints its sha256 and checks its csrc/ and
# cmake/patches/ against a git checkout of the v0.31.0 tag; with it, it runs scripts/build-v031-image.sh end to end.
# Results: /srv/qwen5090/results/public-27b-cleanroom-<stamp>/ (logs, build-logs/v0310/, content report).
set -euo pipefail
[ "${1:-}" = -- ] && shift
SRC=${1:?public tree tarball on flan}; SHA=${2:-}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=/srv/qwen5090/results/public-27b-cleanroom-$STAMP
WORK=/srv/qwen5090/cleanroom-27b
DIND=cleanroom-27b-dind
mkdir -p "$OUT"; exec > >(tee "$OUT/run.log") 2>&1
echo "== $STAMP src=$SRC sha=${SHA:-<compute>}"
sudo -n docker rm -f "$DIND" >/dev/null 2>&1 || true
sudo -n mkdir -p "$WORK/docker"; sudo -n rm -rf "$WORK/repo"; sudo -n mkdir -p "$WORK/repo"
sudo -n tar -xzf "$SRC" -C "$WORK/repo"
sudo -n docker pull -q docker:27-dind >/dev/null
sudo -n docker run -d --name "$DIND" --privileged -e DOCKER_TLS_CERTDIR= \
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
  sudo -n docker exec -e VLLM_SOURCE_SHA256="$SHA" -e NO_CACHE=1 -w /repo "$DIND" bash scripts/build-v031-image.sh || rc=$?
  sudo -n docker exec "$DIND" sh -c 'cp -r /repo/build-logs /out/ 2>/dev/null || true'
  echo "build rc=$rc"
  if [ "$rc" = 0 ]; then
    # The inner daemon has no NVIDIA runtime, so the identity check (CHECK=1) runs on the host. Only a cleanroom tag is
    # exported: loading the launcher's tag would retag the host's served image.
    CR=vllm-qwen38:v0310-r27c-A_S-L1a-ubplan-cleanroom-$STAMP
    sudo -n docker exec "$DIND" docker tag vllm-qwen38:v0310-r27c-A_S-L1a-ubplan "$CR"
    sudo -n docker exec "$DIND" docker save "$CR" | sudo -n docker load
    sudo -n docker image inspect --format '{{.Id}}' "$CR" | tee "$OUT/cleanroom-image-id.txt"
    code=$(sed -n "/^IDENTITY_CODE=\$(cat <<'PY'\$/,/^PY\$/p" "$WORK/repo/scripts/build-v031-image.sh" | sed '1d;$d')
    sudo -n docker run --rm --network=none --runtime nvidia --gpus all -e CUDA_VISIBLE_DEVICES= --entrypoint python3 \
      "$CR" -c "$code" 2>&1 | tee "$OUT/identity.log"
    echo "identity rc=${PIPESTATUS[0]} image=$CR"
  fi
fi
sudo -n docker rm -f "$DIND" >/dev/null
sudo -n chown -R "$(id -u):$(id -g)" "$OUT"
echo "== done $OUT (inner storage kept at $WORK/docker until the image is published; remove with sudo rm -rf $WORK)"
