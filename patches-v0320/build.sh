#!/usr/bin/env bash
# Local CPU-only image builds. No NVIDIA runtime, device access, or host install.
set -euo pipefail
cd "$(dirname "$0")"
BASE=vllm/vllm-openai@sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9
CONTROL=vllm-qwen38:v0320-native-r1
CANDIDATE=vllm-qwen38:v0320-native-ba4-r1
mkdir -p out
docker image inspect "$BASE" >/dev/null 2>&1 || docker pull --platform linux/amd64 "$BASE"
[ "$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$BASE")" = linux/amd64 ]
docker build --platform linux/amd64 --network=none --pull=false --target native \
  --build-arg BASE_IMAGE="$BASE" -t "$CONTROL" . 2>&1 | tee out/build-native.log
CONTROL_ID=$(docker image inspect --format '{{.Id}}' "$CONTROL")
docker build --platform linux/amd64 --network=none --pull=false --target ba4 \
  --build-arg BASE_IMAGE="$BASE" --build-arg CONTROL_IMAGE_ID="$CONTROL_ID" \
  -t "$CANDIDATE" . 2>&1 | tee out/build-ba4.log
docker image inspect --format '{{.Id}} {{.Os}}/{{.Architecture}}' "$CONTROL" "$CANDIDATE" | tee out/images.txt
