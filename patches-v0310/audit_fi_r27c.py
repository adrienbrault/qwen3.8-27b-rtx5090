"""Read-only audit of the supplied/installed FI source; execute no source code."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


FILES = (
    "flashinfer/gemm/gemm_base.py",
    "flashinfer/gdn_prefill.py",
    "flashinfer/gdn_kernels/delta_rule_dsl/varlen_helper.py",
    "flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_cp_sm120.py",
)


def audit(root):
    sources = {name: (root / name).read_text() for name in FILES}
    gemm = ast.parse(sources[FILES[0]])
    functions = {n.name: n for n in gemm.body if isinstance(n, ast.FunctionDef)}
    helper = functions["_gemm_workspace_at_least"]
    helper_source = ast.unparse(helper)
    assert "resize_" not in {n.attr for n in ast.walk(helper) if isinstance(n, ast.Attribute)}
    assert "is_current_stream_capturing" in helper_source
    assert "CudnnWorkspaceTooSmallInCaptureError" in helper_source
    assert "torch.empty" in helper_source
    calls = []
    for name, fn in functions.items():
        if name.startswith("execute_cudnn_gemm"):
            assert not any(isinstance(n, ast.Attribute) and n.attr == "resize_" for n in ast.walk(fn)), name
            assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                       and n.func.id == "_gemm_workspace_at_least" for n in ast.walk(fn)), name
            calls.append({"name": name, "line": fn.lineno})
    assert len(calls) == 8, calls
    prefill = sources[FILES[1]]
    assert "12: cp_delta_rule_dsl_sm120" in prefill
    assert 'use_cp == "auto"' in prefill
    cp = sources[FILES[3]]
    assert "cp_delta_rule_mn_precompute_dsl_sm120(" in cp
    return {"source_root": str(root),
            "sha256": {n: hashlib.sha256(s.encode()).hexdigest() for n, s in sources.items()},
            "workspace_helper_line": helper.lineno,
            "cuDNN_execute_helper_calls": calls,
            "shared_resize_mechanism": "addressed in supplied FI 0.7 source; capture growth fails closed",
            "sm120_cp_prefill": "still reachable under auto dispatch; not disabled by ReplaySSM decode",
            "R685": "mitigated at the leading workspace mechanism; crash fix NOT PROVEN",
            "first_GPU_gate": "exact R685 NONCE=11084 full-corpus reproduction and soak",
            "fallback": "--gdn-prefill-backend triton"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(audit(args.root), indent=2) + "\n"
    if args.output:
        args.output.write_text(result)
    print(result)
