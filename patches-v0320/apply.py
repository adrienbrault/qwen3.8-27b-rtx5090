#!/usr/bin/env python3
"""Install the pinned source series without importing torch or touching GPUs."""
import argparse
import hashlib
import json
from pathlib import Path
import py_compile
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def install(root, ba4=False, control_image=None):
    manifest = json.loads((HERE / 'manifest.json').read_text())
    selected = [p for p in manifest['patches'] if p['ba4'] == ba4]
    if ba4 and not re.fullmatch(r'sha256:[0-9a-f]{64}', control_image or ''):
        raise ValueError('BA4 integration requires the freshly built native control image ID')
    # Validate the complete patch payload before changing the target.
    for patch in selected:
        if sha(HERE / 'patches' / patch['name']) != patch['sha256']:
            raise ValueError('patch checksum mismatch: ' + patch['name'])
    touched = set()
    for patch in selected:
        for name, expected in patch['before'].items():
            if sha(root / name) != expected or (expected is None and (root / name).exists()):
                raise ValueError('input source mismatch: ' + name)
        command = ['patch', '--batch', '--fuzz=0', '--no-backup-if-mismatch',
                   '-p1', '-d', str(root), '-i', str(HERE / 'patches' / patch['name'])]
        subprocess.run(command[:1] + ['--dry-run'] + command[1:], check=True)
        subprocess.run(command, check=True)
        for name, expected in patch['after'].items():
            if sha(root / name) != expected:
                raise ValueError('output source mismatch: ' + name)
            touched.add(name)
    if ba4:
        evidence = root / 'vllm_ba4/evidence.py'
        text = evidence.read_text()
        if text.count("BASE_IMAGE = 'UNQUALIFIED_V032_CONTROL_IMAGE'") != 1:
            raise ValueError('unexpected BA4 control-image binding')
        evidence.write_text(text.replace('UNQUALIFIED_V032_CONTROL_IMAGE', control_image))
    else:
        dest = root / 'pcie_ipc_ar21'
        if dest.exists():
            raise ValueError('preexisting PCIe helper package')
        shutil.copytree(HERE / 'pcie_ipc_ar21', dest,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '_build', '_C.so'))
        touched.update(str(p.relative_to(root)) for p in dest.rglob('*.py'))
    for name in sorted(touched):
        path = root / name
        for pyc in path.parent.glob('__pycache__/' + path.stem + '.*.pyc'):
            pyc.unlink()
        py_compile.compile(str(path), doraise=True)
    return {'upstream': manifest['upstream'], 'ba4_integration': ba4,
            'control_image': control_image, 'files': {n: sha(root / n) for n in sorted(touched)}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--ba4', action='store_true')
    parser.add_argument('--control-image')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    receipt = install(args.root.resolve(), args.ba4, args.control_image)
    if args.receipt:
        args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print('Source hashes and Python compilation verified for', len(receipt['files']), 'files')
