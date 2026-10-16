"""Fail if a release image's patch inputs differ from the audited local tree."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

root = Path(importlib.util.find_spec("vllm").origin).parent.parent
manifest = json.loads(Path(sys.argv[1]).read_text())
for name, expected in manifest.items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"Source drift before patching: {name}: {actual} != {expected}")
print(f"SOURCE INPUTS VERIFIED: {len(manifest)} files")
