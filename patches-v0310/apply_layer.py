"""Apply one logical layer, compile touched Python, and write source markers."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import py_compile
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=Path)
parser.add_argument("ids", nargs="+")
parser.add_argument("--root", type=Path)
args = parser.parse_args()
root = args.root or Path(importlib.util.find_spec("vllm").origin).parent.parent
markers = Path("/opt/prs-markers")
markers.mkdir(exist_ok=True)
for ident in args.ids:
    patches = list(args.folder.glob(ident + "-*.diff"))
    if len(patches) != 1:
        raise RuntimeError(f"Expected one patch for {ident}: {patches}")
    patch = patches[0]
    touched = [line[6:] for line in patch.read_text().splitlines() if line.startswith("+++ b/")]
    for name in touched:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-d", str(root)],
                   input=patch.read_bytes(), check=True)
    hashes = {}
    for name in touched:
        path = root / name
        if name.endswith(".py"):
            py_compile.compile(str(path), doraise=True)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (markers / ident).write_text("APPLIED " + patch.name + "\n")
    (markers / (ident + ".json")).write_text(json.dumps(hashes, indent=2) + "\n")
    print("APPLIED", patch.name)
