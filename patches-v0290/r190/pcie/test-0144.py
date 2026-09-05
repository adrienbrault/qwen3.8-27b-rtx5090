# Usage: python3 deliver/test-0144.py; dependency-free source/shape invariants only.
import ast
import os
from pathlib import Path
from unittest.mock import patch

root = Path(__file__).resolve().parent.parent / '.work/0144'
tree = ast.parse((root / 'vllm/envs.py').read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_pcie_fused_norm_enabled')
namespace = {'os': os}
exec(compile(ast.Module(body=[fn], type_ignores=[]), '<isolated env parser>', 'exec'), namespace)
parse = namespace[fn.name]
for value, ar, expect in [(None, None, False), ('0', None, False), ('1', '1', True),
                          ('1', None, ValueError), ('1', '0', ValueError),
                          ('true', '1', ValueError), ('', '1', ValueError), ('2', '1', ValueError)]:
    env = {}
    if value is not None:
        env['VLLM_SM12X_PCIE_IPC_AR_FUSED_NORM'] = value
    if ar is not None:
        env['VLLM_SM12X_PCIE_IPC_AR'] = ar
    with patch.dict(os.environ, env, clear=True):
        try:
            result = parse()
        except ValueError:
            assert expect is ValueError
        else:
            assert result is expect
# Every row/pack is owned exactly once; all scratch addressing fits its epoch.
for rows in [1, 10, 80, 127, 128, 129, 160, 320]:
    blocks = min(rows, 128)
    owned = [(row, t + j * 256) for b in range(blocks)
             for row in range(b, rows, blocks) for t in range(256)
             for j in range(3) if t + j * 256 < 640]
    assert len(owned) == len(set(owned)) == rows * 640
    for epoch in (0, 1):
        for rank in (0, 1):
            lo = epoch * 2 * (320 * 640) + rank * rows * 640
            hi = lo + rows * 640
            assert epoch * 2 * (320 * 640) <= lo < hi <= (epoch + 1) * 2 * (320 * 640)
assert 2048 * 5120 > 320 * 5120
print('PASS: strict knob grammar, row ownership and scratch epoch bounds')
