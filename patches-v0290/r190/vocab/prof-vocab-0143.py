#!/usr/bin/env python3
# Usage: python3 /path/to/deliver/prof-vocab-0143.py trace.json[.gz] (inside served container).
"""Count host phase annotations and correlate only directly launched GPU work.
Graph replay kernels intentionally remain unassigned; no timing-overlap guessing.
Reports whole trace, not prof_decode_split.py's post-prefill window.
"""
import collections
import gzip
import json
import re
import sys


def summarize(events):
    pattern = re.compile(r'^vocab_ag:([^:]+):(\d+)x(\d+):bytes=(\d+):dtype=(.+)$')
    tags = [(e, pattern.match(e.get('name', ''))) for e in events if e.get('ph') == 'X']
    tags = [(e, m) for e, m in tags if m]
    totals = collections.defaultdict(lambda: dict(calls=0, input_bytes=0, gpu_us=0, kernels=0))
    scopes = collections.defaultdict(list)
    for e, m in tags:
        phase = m[1]
        totals[phase]['calls'] += 1
        totals[phase]['input_bytes'] += int(m[4])
        scopes[(e.get('pid'), e.get('tid'))].append((e['ts'], e['ts'] + e['dur'], phase))
    correlation = {}
    for e in events:
        if e.get('cat') not in ('cuda_runtime', 'cuda_driver') or e.get('ph') != 'X':
            continue
        corr = e.get('args', {}).get('correlation')
        if corr is None: continue
        containing = [s for s in scopes[(e.get('pid'), e.get('tid'))]
                      if s[0] <= e['ts'] < s[1]]
        if containing:
            # Use shortest containing scope, never GPU timestamp overlap.
            correlation[corr] = min(containing, key=lambda s: s[1]-s[0])[2]
    unassigned = 0
    for e in events:
        if e.get('cat') != 'kernel' or e.get('ph') != 'X': continue
        if 'nccl' not in e.get('name', '').lower(): continue
        phase = correlation.get(e.get('args', {}).get('correlation'))
        if phase is None:
            unassigned += 1
        else:
            totals[phase]['gpu_us'] += e['dur']
            totals[phase]['kernels'] += 1
    return dict(totals), unassigned


if __name__ == '__main__':
    path = sys.argv[1]
    with (gzip.open(path, 'rt') if path.endswith('.gz') else open(path)) as f:
        data = json.load(f)
    totals, unknown = summarize(data.get('traceEvents', []) if isinstance(data, dict) else data)
    if not totals:
        raise SystemExit('No 0143 host annotations: enable tags and recapture; NVTX alone is not a torch trace annotation.')
    print(json.dumps(totals, indent=2, sort_keys=True))
    print(f'Unassigned NCCL kernels (includes graph replay and non-vocab): {unknown}')
    print('Host counts include warmup/capture if traced; no per-step or replay attribution is inferred.')
