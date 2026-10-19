#!/usr/bin/env python3
"""Fail-closed installation into the exact daily source; no GPU import."""

import hashlib
import importlib.util
import json
from pathlib import Path
import py_compile
import subprocess

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install(site):
    manifest = json.loads((HERE / "manifest.json").read_text())
    patch = HERE / "L1a_BA4.diff"
    assert manifest["base"] == "vllm-qwen38:v0310-r27c-A_S-L1a-ubplan"
    assert sha(patch) == manifest["patch_sha256"], "patch checksum"
    for rel, expected in manifest["before"].items():
        assert sha(site / rel) == expected, "base mismatch: " + rel
    for rel, expected in manifest["unchanged"].items():
        assert sha(site / rel) == expected, "daily L1a context mismatch: " + rel
    for rel in manifest["after"].keys() - manifest["before"].keys():
        assert not (site / rel).exists(), "preexisting helper: " + rel
    with patch.open("rb") as f:
        subprocess.run(["patch", "--batch", "--fuzz=0", "--no-backup-if-mismatch",
                        "-p1", "-d", str(site)], stdin=f, check=True)
    for rel, expected in manifest["after"].items():
        path = site / rel
        assert sha(path) == expected, "patched mismatch: " + rel
        # Invalidate timestamp-based bytecode even for a same-size file.
        for pyc in path.parent.glob("__pycache__/" + path.stem + ".*.pyc"):
            pyc.unlink()
        py_compile.compile(str(path), doraise=True)
    for rel, expected in manifest["unchanged"].items():
        assert sha(site / rel) == expected, "unchanged file modified: " + rel
    print("L1a_BA4 installed: pinned daily base, diff and every output hash verified")


if __name__ == "__main__":
    spec = importlib.util.find_spec("vllm")
    assert spec and spec.origin, "vllm installation missing"
    install(Path(spec.origin).resolve().parent.parent)
