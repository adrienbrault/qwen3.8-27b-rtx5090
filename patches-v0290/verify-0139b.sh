#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
python3 - <<'PY'
from pathlib import Path
import shutil
base = Path('.work22/base')
if base.exists():
    shutil.rmtree(base)
base.mkdir(parents=True)
shutil.copytree('src', base / 'vllm')
print('Copied src/ to .work22/base/vllm/', flush=True)
PY
cd .work22/base
patch --batch --forward --dry-run -p1 --fuzz=0 < "$ROOT/deliver/0139b-nvfp4-a16-allowlist-v0290.diff"
patch --batch --forward -p1 --fuzz=0 < "$ROOT/deliver/0139b-nvfp4-a16-allowlist-v0290.diff"
python3 - <<'PY'
from pathlib import Path
import importlib.util
import py_compile
import sys

patch = Path('../../deliver/0139b-nvfp4-a16-allowlist-v0290.diff')
# The scratch tree is .work22/base; deliver is two directories above it.
files = [line[6:] for line in patch.read_text().splitlines()
         if line.startswith('+++ b/')]
assert len(files) == 5
for name in files:
    py_compile.compile(name, doraise=True)
print(f'py_compile: PASS ({len(files)} touched files)')

# Load only this standard-library helper, bypassing all vllm package imports.
spec = importlib.util.spec_from_file_location('allowlist_under_test', 'vllm/nvfp4_marlin_allowlist.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
parse, match = helper.parse_allowlist, helper.matches_allowlist
name = 'model.language_model.layers.17.mlp.down_proj'
assert not match(parse(None), name)
assert not match(parse(''), name)
pattern = parse(r'layers\.(\d+)\.mlp\.down_proj$')
assert match(pattern, name)
assert not match(pattern, name.replace('down_proj', 'gate_up_proj'))
assert not match(pattern, None)
try:
    parse('[')
except ValueError as exc:
    assert "'['" in str(exc)
    assert 'VLLM_SM12X_NVFP4_MARLIN_LAYERS' in str(exc)
else:
    raise AssertionError('Invalid regex did not raise ValueError')
print('Matcher: PASS (unset, empty, down_proj, fused-name exclusion, missing name, invalid regex)')

arms = [
    (r'layers\.([0-9]|[1-4][0-9]|5[0-5])\.mlp\.down_proj$', 56),
    (r'layers\.([0-9]|[1-4][0-9]|5[0-5])\.mlp\.gate_up_proj$', 56),
    (r'layers\.([0-9]|1[0-8])\.mlp\.(gate_up|down)_proj$', 38),
    (r'layers\.([0-9]|[1-4][0-9]|5[0-5])\.mlp\.(gate_up|down)_proj$', 112),
]
for regex, expected in arms:
    count = sum(match(parse(regex), f'model.language_model.layers.{i}.mlp.{proj}_proj')
                for i in range(64) for proj in ('gate_up', 'down'))
    assert count == expected, (regex, count, expected)
assert 'vllm' not in sys.modules and 'torch' not in sys.modules
print('Arm matcher counts: PASS (56, 56, 38; all-NVFP4 control 112)')
# Evaluate only the two actual env getter ASTs, never import envs/vllm.
import ast
from types import SimpleNamespace
source = ast.parse(Path('vllm/envs.py').read_text())
keys = ('VLLM_SM12X_NVFP4_MARLIN_LAYERS', 'VLLM_SM12X_NVFP4_A16_KERNEL')
getters = {}
values = {}
for node in ast.walk(source):
    if isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value in keys:
                assert key.value not in getters
                getters[key.value] = eval(
                    compile(ast.Expression(value), '<env getter>', 'eval'),
                    {'os': SimpleNamespace(getenv=values.get)},
                )
assert set(getters) == set(keys)
regex_env, kernel_env = keys
assert getters[regex_env]() is None
assert getters[kernel_env]() == 'marlin'
assert helper.parse_a16_kernel(None) == 'marlin'
assert helper.parse_a16_kernel(getters[kernel_env]()) == 'marlin'
for choice in ('marlin', 'humming'):
    values[kernel_env] = choice
    assert type(getters[kernel_env]()) is str
    assert helper.parse_a16_kernel(getters[kernel_env]()) == choice
for bad in ('', 'Humming', ' humming', 'cutlass'):
    values[kernel_env] = bad
    try:
        helper.parse_a16_kernel(getters[kernel_env]())
    except ValueError as exc:
        assert kernel_env in str(exc) and repr(bad) in str(exc)
    else:
        raise AssertionError(f'Invalid kernel accepted: {bad!r}')
for regex in ('', r'layers\.(\d+)\.mlp\.down_proj$', '['):
    values[regex_env] = regex
    raw = getters[regex_env]()
    assert raw == (regex or None)
    assert raw is None or type(raw) is str
print('Kernel/env parsing: PASS (actual primitive getters, default, both choices, invalid values)')
assert 'vllm' not in sys.modules and 'torch' not in sys.modules
print('PASS: offline verification complete; no vllm/torch imports')
PY
