#!/usr/bin/env python3
"""Collect and compare the served L1a file inventory without importing packages."""

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import sys


SOURCE_SUFFIXES = {".py", ".json", ".cu", ".cuh"}
PACKAGE_SUFFIXES = SOURCE_SUFFIXES | {".so"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_files(root):
    if not root.is_dir():
        raise ValueError(f"missing manifest root: {root}")
    files = []

    def walk_error(error):
        raise error

    for directory, _, names in os.walk(root, followlinks=False, onerror=walk_error):
        for name in names:
            path = Path(directory) / name
            if not path.is_symlink() and path.is_file():
                files.append(path)
    return files


def generate(site_root, overlay_root):
    # Match the evidence's two groups: package paths, then the overlay tree.
    packages = []
    for name in ("flashinfer", "vllm"):
        files = regular_files(site_root / name)
        selected = [path for path in files if path.suffix in PACKAGE_SUFFIXES]
        if not selected:
            raise ValueError(f"empty package manifest: {name}")
        packages.extend((path.relative_to(site_root).as_posix(), path) for path in selected)
    overlay = [("vllm-sm12x/" + path.relative_to(overlay_root).as_posix(), path)
               for path in regular_files(overlay_root)]
    if not overlay:
        raise ValueError("empty overlay manifest")
    for name, path in sorted(packages) + sorted(overlay):
        print(f"{sha256(path)}  {name}")


def read_manifest(path):
    entries = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ValueError(f"{path.name}:{number}: invalid sha256sum line")
        digest, name = match.groups()
        parts = PurePosixPath(name).parts
        if (len(parts) < 2 or parts[0] not in {"flashinfer", "vllm", "vllm-sm12x"}
                or ".." in parts or str(PurePosixPath(name)) != name):
            raise ValueError(f"{path.name}:{number}: invalid inventory path: {name}")
        if parts[0] != "vllm-sm12x" and PurePosixPath(name).suffix not in PACKAGE_SUFFIXES:
            raise ValueError(f"{path.name}:{number}: unexpected package suffix: {name}")
        if name in entries:
            raise ValueError(f"{path.name}:{number}: duplicate path: {name}")
        entries[name] = digest
    if not entries:
        raise ValueError(f"{path.name}: empty manifest")
    return entries


def compare(expected_path, actual_path):
    expected = read_manifest(expected_path)
    actual = read_manifest(actual_path)
    missing = sorted(expected.keys() - actual.keys())
    extra = sorted(actual.keys() - expected.keys())
    common = sorted(expected.keys() & actual.keys())
    groups = {name: [0, 0] for name in ("sources", "shared objects", "other overlay files")}
    for name in missing:
        print(f"MISSING {name}")
    for name in extra:
        print(f"EXTRA {name}")
    for name in common:
        suffix = PurePosixPath(name).suffix
        group = ("sources" if suffix in SOURCE_SUFFIXES else
                 "shared objects" if suffix == ".so" else "other overlay files")
        mismatch = expected[name] != actual[name]
        groups[group][int(mismatch)] += 1
        if mismatch:
            print(f"MISMATCH [{group}] {name} expected={expected[name]} actual={actual[name]}")
        elif group == "shared objects":
            print(f"MATCH [shared objects] {name}")
    print(f"inventory: expected={len(expected)} actual={len(actual)} missing={len(missing)} extra={len(extra)}")
    for group, (matches, mismatches) in groups.items():
        print(f"{group}: matches={matches} mismatches={mismatches}")
    failed = bool(missing or extra or groups["sources"][1])
    print("CONTENT FAIL" if failed else "CONTENT PASS (compiled and other overlay byte differences are informational)")
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("generate")
    collect.add_argument("--site-root", type=Path, default=Path("/usr/local/lib/python3.12/dist-packages"))
    collect.add_argument("--overlay-root", type=Path, default=Path("/opt/vllm-sm12x"))
    diff = commands.add_parser("compare")
    diff.add_argument("expected", type=Path)
    diff.add_argument("actual", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "generate":
            generate(args.site_root, args.overlay_root)
            return 0
        return compare(args.expected, args.actual)
    except (OSError, ValueError) as error:
        print(f"MANIFEST ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
