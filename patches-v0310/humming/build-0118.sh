#!/usr/bin/env bash
# CPU-only build of today's official Humming release, separate from the daily.
set -euo pipefail
cd -- "$(dirname -- "$0")"
sudo docker build -f Dockerfile.0118 -t vllm-qwen38:v0310-r27c-A_S-BA4-L1a-ubplan-humming-0118 .
