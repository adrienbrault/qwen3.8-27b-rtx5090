#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BRIEF_DIR=$(dirname -- "$SCRIPT_DIR")
DIFF="$SCRIPT_DIR/0137-fs-tier-eviction-v0290.diff"
VERIFY_ROOT=$(mktemp -d "$SCRIPT_DIR/work/verify-0137.XXXXXX")

copy_and_patch() {
  label=$1
  source_tree=$2
  target_tree="$VERIFY_ROOT/$label"
  mkdir -p "$target_tree"
  cp -aL "$source_tree/." "$target_tree/"
  echo "[$label] patch dry-run (--fuzz=0)"
  patch --dry-run --fuzz=0 -p1 -i "$DIFF" -d "$target_tree"
  echo "[$label] patch apply (--fuzz=0)"
  patch --fuzz=0 -p1 -i "$DIFF" -d "$target_tree"
}

copy_and_patch v029src "$BRIEF_DIR/TREES/v029src"
copy_and_patch v029real "$BRIEF_DIR/TREES/v029real"
copy_and_patch v028src "$BRIEF_DIR/TREES/v028src"

if [ -z "${PYTHON_BIN:-}" ]; then
  for candidate in python python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
      "$candidate" -c "import numpy, pytest, torch" >/dev/null 2>&1; then
      PYTHON_BIN=$candidate
      break
    fi
  done
fi

TEST_PREFIX=
if [ -z "${PYTHON_BIN:-}" ]; then
  LOCAL_PYTHON=/Volumes/Developer/ai/harness-llm-bench/.venv/bin/python
  LOCAL_TORCH=/Users/adrienbrault/.cache/uv/archive-v0/mpti-nMnnb5LI4Ab
  LOCAL_BOOTSTRAP="$SCRIPT_DIR/work/test-bootstrap"
  if [ ! -x "$LOCAL_PYTHON" ] || [ ! -d "$LOCAL_TORCH" ] ||
    [ ! -f "$LOCAL_BOOTSTRAP/sitecustomize.py" ]; then
    echo "No Python with pytest, numpy, and torch found." >&2
    echo "Set PYTHON_BIN to a vLLM development-environment Python." >&2
    exit 1
  fi
  PYTHON_BIN=$LOCAL_PYTHON
  TEST_PREFIX="$LOCAL_BOOTSTRAP:$LOCAL_TORCH"
fi

TOUCHED_FILES=(
  vllm/v1/kv_offload/tiering/fs/manager.py
  vllm/v1/kv_offload/tiering/fs/thread_pool.py
  vllm/v1/kv_offload/tiering/metrics.py
  vllm/v1/kv_offload/tiering/spec.py
  tests/v1/kv_offload/tiering/test_fs_tier_eviction.py
)

for label in v029src v029real v028src; do
  target_tree="$VERIFY_ROOT/$label"
  compile_paths=()
  for relative_path in "${TOUCHED_FILES[@]}"; do
    compile_paths+=("$target_tree/$relative_path")
  done
  echo "[$label] py_compile"
  "$PYTHON_BIN" -m py_compile "${compile_paths[@]}"

  echo "[$label] pytest"
  test_pythonpath="$target_tree"
  if [ -n "$TEST_PREFIX" ]; then
    test_pythonpath="$TEST_PREFIX:$test_pythonpath"
  fi
  if [ -n "${PYTHONPATH:-}" ]; then
    test_pythonpath="$test_pythonpath:$PYTHONPATH"
  fi
  env VLLM_TEST_ROOT="$target_tree" PYTHONPATH="$test_pythonpath" "$PYTHON_BIN" -m pytest --confcutdir="$target_tree/tests/v1/kv_offload/tiering" -q "$target_tree/tests/v1/kv_offload/tiering/test_fs_tier_eviction.py"
done

echo "verification work: $VERIFY_ROOT"
