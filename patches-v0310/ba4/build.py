#!/usr/bin/env python3
"""Build only on the approved local parent. No arbitrary Docker build args."""
import argparse
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def build(tag, run=subprocess.run):
    manifest = json.loads((HERE / 'manifest.json').read_text())
    expected = manifest['base_image_id']
    parent = run(['docker', 'image', 'inspect', '--format', '{{.Id}}', manifest['base']],
                 check=True, capture_output=True, text=True).stdout.strip()
    if parent != expected:
        raise ValueError('approved parent image ID mismatch: ' + parent)
    lines = (HERE / 'Dockerfile').read_text().splitlines()
    # BuildKit resolves FROM sha256:<config id> as a repository name (moby/buildkit#2204), so the Dockerfile names the
    # tag; the tag is checked against the approved image ID above and the built image's layer chain is checked below.
    if [line for line in lines if line.startswith('FROM ')] != ['FROM ' + manifest['base']]:
        raise ValueError('Dockerfile parent must be the approved base tag')
    if any(line.startswith('ARG ') for line in lines):
        raise ValueError('parent/build argument overrides are forbidden')
    run(['docker', 'build', '--network=none', '--pull=false', '-t', tag,
         '-f', str(HERE / 'Dockerfile'), str(HERE)], check=True)
    def layers(ref):
        out = run(['docker', 'image', 'inspect', '--format', '{{json .RootFS.Layers}}', ref],
                  check=True, capture_output=True, text=True).stdout
        return json.loads(out)
    parent_layers, built_layers = layers(expected), layers(tag)
    if built_layers[:len(parent_layers)] != parent_layers or len(built_layers) <= len(parent_layers):
        raise ValueError('built image is not layered on the approved parent ' + expected)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', default='vllm-qwen38:v0310-r27c-A_S-BA4-L1a-ubplan')
    args = p.parse_args()
    build(args.tag)


if __name__ == '__main__':
    main()
