#!/usr/bin/env bash
# CPU-only experiment-image build; the daily image/tag is never modified.
set -euo pipefail
cd -- "$(dirname -- "$0")"
BASE=vllm-qwen38:v0310-r27c-A_S-BA4-L1a-ubplan
IMAGE=$BASE-humming-r1
sudo docker image inspect "$BASE" --format '{{.Id}}'
# Reproduce against the actual installed Python class before applying the fix.
if sudo docker run --rm --network none -v "$PWD/test_deviceinfo.py:/tmp/test_deviceinfo.py:ro" \
  --entrypoint python3 "$BASE" /tmp/test_deviceinfo.py > deviceinfo-before.log 2>&1; then
  echo 'REFUSED: baseline did not reproduce'; exit 1
fi
grep -q 'RecursionError: maximum recursion depth exceeded' deviceinfo-before.log
sudo docker build --network none -t "$IMAGE" .
sudo docker image inspect "$IMAGE" --format '{{.Id}}'
