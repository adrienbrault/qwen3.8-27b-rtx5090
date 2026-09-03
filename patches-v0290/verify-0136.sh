#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
tree=${1:-"$repo_root/TREES/v029real"}
patch_file="$script_dir/0136-opt-in-nvfp4-prefill-split-kv-v0290.diff"
target_rel="vllm/v1/attention/backends/flashinfer.py"

test -f "$tree/$target_rel"
patch --dry-run -p1 --fuzz=0 -d "$tree" < "$patch_file"

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/verify-0136.XXXXXX")
trap 'rm -rf -- "$work_dir"' EXIT
mkdir -p "$work_dir/$(dirname -- "$target_rel")"
cp "$tree/$target_rel" "$work_dir/$target_rel"
patch -p1 --fuzz=0 -d "$work_dir" < "$patch_file"
python3 -m py_compile "$work_dir/$target_rel"

python3 - "$work_dir/$target_rel" <<'PY'
import ast
import functools
import os
import pathlib
import sys
import types

source_path = pathlib.Path(sys.argv[1])
tree = ast.parse(source_path.read_text(encoding="utf-8"))
functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
helper_name = "_maybe_enable_sm12x_nvfp4_prefill_split_kv"
assert helper_name in functions
source = source_path.read_text(encoding="utf-8")
assert 'VLLM_SM12X_NVFP4_PREFILL_SPLIT_KV' in source
assert f'{helper_name}()' in source

class FakeLogger:
    def info_once(self, *args, **kwargs):
        pass

    def warning_once(self, *args, **kwargs):
        pass


# Execute just the added helper with a fake FlashInfer module: no torch/vLLM/GPU
# import is needed. Check both the safe default and explicit opt-in behavior.
helper_module = ast.Module(body=[functions[helper_name]], type_ignores=[])
ast.fix_missing_locations(helper_module)
namespace = {"cache": functools.cache, "logger": FakeLogger()}
exec(compile(helper_module, str(source_path), "exec"), namespace)
helper = namespace[helper_name]

fake_flashinfer = types.ModuleType("flashinfer")
fake_flashinfer.__path__ = []
fake_prefill = types.ModuleType("flashinfer.prefill")
original_guard = lambda _dtype: True
fake_prefill._nvfp4_kv_requires_disabled_split_kv = original_guard
fake_flashinfer.prefill = fake_prefill
sys.modules["flashinfer"] = fake_flashinfer
sys.modules["flashinfer.prefill"] = fake_prefill

os.environ.pop("VLLM_SM12X_NVFP4_PREFILL_SPLIT_KV", None)
helper()
assert fake_prefill._nvfp4_kv_requires_disabled_split_kv is original_guard

helper.cache_clear()
os.environ["VLLM_SM12X_NVFP4_PREFILL_SPLIT_KV"] = "1"
helper()
assert fake_prefill._nvfp4_kv_requires_disabled_split_kv(None) is False
print("0136 dry-run apply, syntax, safe-default, and opt-in hook checks passed")
PY
