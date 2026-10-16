"""Audit the installed pin and adapt FI#4553 without moving shared storage."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path


def guard(path: Path) -> dict:
    source = path.read_text()
    tree = ast.parse(source)
    changes = []
    functions = []
    for fn in tree.body:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not fn.name.startswith("execute_cudnn_gemm"):
            continue
        functions.append(fn.name)
        for node in ast.walk(fn):
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            if not isinstance(call.func, ast.Attribute) or call.func.attr != "resize_":
                continue
            if not isinstance(call.func.value, ast.Name) or len(call.args) != 1:
                raise RuntimeError(f"Review unsupported resize in {fn.name}:{node.lineno}")
            name = call.func.value.id
            if "workspace" not in name:
                raise RuntimeError(f"Review non-workspace resize in {fn.name}")
            size = ast.unparse(call.args[0])
            changes.append((node.lineno, node.end_lineno, name, size, fn.name))
    if not functions:
        raise RuntimeError("No cuDNN execution functions found; inspect this pin manually")
    lines = source.splitlines(keepends=True)
    for start, end, name, size, fn in sorted(changes, reverse=True):
        indent = lines[start - 1][:len(lines[start - 1]) - len(lines[start - 1].lstrip())]
        lines[start - 1:end] = [
            indent + f"{name} = torch.zeros({size}, dtype=torch.uint8, device={name}.device)\n"
        ]
    result = "".join(lines)
    ast.parse(result)
    # No other resize expression may survive inside these functions.
    for fn in ast.parse(result).body:
        if isinstance(fn, ast.FunctionDef) and fn.name in functions:
            if any(isinstance(n, ast.Attribute) and n.attr == "resize_" for n in ast.walk(fn)):
                raise RuntimeError(f"Unreviewed shared resize remains in {fn.name}")
    path.write_text(result)
    return {
        "path": str(path), "functions": functions,
        "replacements": [{"function": fn, "line": start, "buffer": name}
                         for start, end, name, size, fn in changes],
        "before_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "after_sha256": hashlib.sha256(result.encode()).hexdigest(),
        "remaining_execute_cudnn_resize": 0,
        "scope": "execute_cudnn_gemm* only; GPU crash cause remains unproven",
    }


if __name__ == "__main__":
    spec = importlib.util.find_spec("flashinfer")
    path = Path(spec.origin).parent / "gemm/gemm_base.py"
    report = guard(path)
    target = Path("/opt/prs-markers/0163-audit.json")
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
