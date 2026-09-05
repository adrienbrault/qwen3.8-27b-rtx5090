#!/usr/bin/env python3
"""Usage: python measure_acceptance.py --corpus heldout.jsonl --url http://localhost:8029 --exclusive --out acceptance.jsonl."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.request
from build_calib_corpus import request_json

PREFIX = 'vllm:spec_decode_'
COUNTERS = tuple(PREFIX + x + '_total' for x in ('num_drafts', 'num_draft_tokens', 'num_accepted_tokens'))
GAUGES = ('vllm:num_requests_running', 'vllm:num_requests_waiting')
SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+\d+)?$')
LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def parse_metrics(text, model):
    series = {}
    for line in text.splitlines():
        match = SAMPLE.fullmatch(line)
        if not match:
            continue
        name, raw, value = match.groups()
        if name not in COUNTERS + GAUGES and name != PREFIX + 'num_accepted_tokens_per_pos_total':
            continue
        labels = dict((k, json.loads('"' + v + '"')) for k, v in LABEL.findall(raw or ''))
        if labels.get('model_name') != model:
            continue
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f'Invalid metric {name}')
        key = (name, tuple(sorted(labels.items())))
        if key in series:
            raise ValueError('Duplicate Prometheus series')
        series[key] = number
    if not set(COUNTERS + GAUGES) <= {k[0] for k in series}:
        raise ValueError('Missing model-labelled metrics; check --metrics-model and speculative decoding')
    return series


def summarize(before, after):
    if before.keys() != after.keys():
        raise ValueError('Metric series changed; restart or label change invalidates run')
    totals = {}
    positions = {}
    for (name, labels), value in before.items():
        delta = after[(name, labels)] - value
        if name in GAUGES:
            continue
        if delta < 0:
            raise ValueError('Counter reset invalidates run')
        totals[name] = totals.get(name, 0) + delta
        if name.endswith('per_pos_total'):
            pos = dict(labels)['position']
            positions[pos] = positions.get(pos, 0) + delta
    drafts, proposed, accepted = [totals[k] for k in COUNTERS]
    if not 0 < drafts <= proposed or not 0 <= accepted <= proposed:
        raise ValueError('No drafts or inconsistent counters')
    return dict(request_draft_steps=drafts, proposed_tokens=proposed, accepted_tokens=accepted,
                accepted_tokens_per_request_step=accepted / drafts,
                mean_acceptance_length=1 + accepted / drafts,
                acceptance_fraction=accepted / proposed, proposals_per_request_step=proposed / drafts,
                per_position_survival={k: v / drafts for k, v in sorted(positions.items(), key=lambda x: int(x[0]))})


def snapshot(base, model):
    headers = {}
    if os.environ.get('OPENAI_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['OPENAI_API_KEY']
    with urllib.request.urlopen(urllib.request.Request(base + '/metrics', headers=headers), timeout=10) as response:
        return parse_metrics(response.read().decode(), model)


def settled(base, model, seconds):
    # No requests may enter this endpoint during the entire measurement.
    deadline = time.monotonic() + max(60, seconds * 4)
    previous = None
    since = time.monotonic()
    while time.monotonic() < deadline:
        current = snapshot(base, model)
        idle = all(v == 0 for (name, _), v in current.items() if name in GAUGES)
        if not idle or current != previous:
            since = time.monotonic()
        if idle and current == previous and time.monotonic() - since >= seconds:
            return current
        previous = current
        time.sleep(0.5)
    raise RuntimeError('Metrics failed to settle; server busy or counters still publishing')


def complete(base, model, row, tokens, seed):
    body = dict(model=model, messages=row['messages'], max_tokens=tokens, min_tokens=tokens,
                temperature=0.6, seed=seed, stream=False)
    if row.get('tools'):
        body['tools'] = row['tools']
    result = request_json(base + '/v1/chat/completions', body, os.environ.get('OPENAI_API_KEY'))
    if 'error' in result or not result.get('choices'):
        raise RuntimeError(f'Completion failed: {result}')
    return result.get('usage')


def self_test():
    lines = []
    for engine in (0, 1):
        for name in COUNTERS + GAUGES:
            v = 0 if name in GAUGES else 10
            lines.append(f'{name}{{model_name="m",engine="{engine}"}} {v} 1000')
    before = parse_metrics('\n'.join(lines), 'm')
    after = {k: v + (dict(zip(COUNTERS, (2, 18, 6))).get(k[0], 0)) for k, v in before.items()}
    result = summarize(before, after)
    assert result['accepted_tokens_per_request_step'] == 3
    assert result['request_draft_steps'] == 4
    assert result['acceptance_fraction'] == 1 / 3
    try:
        summarize(after, before)
    except ValueError:
        pass
    else:
        raise AssertionError('reset not detected')
    print('R190-28 METRICS dry-run PASS')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus')
    p.add_argument('--url', default='http://localhost:8029')
    p.add_argument('--model', default='qwen3.8-27b')
    p.add_argument('--metrics-model', help='Exact Prometheus model_name, if different from API alias')
    p.add_argument('--out')
    p.add_argument('--conc', type=int, default=1)
    p.add_argument('--per-kind', type=int, default=12)
    p.add_argument('--tokens', type=int, default=1024)
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--settle-seconds', type=float, default=12)
    p.add_argument('--exclusive', action='store_true', help='Assert endpoint reserved, no unrelated requests')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if min(a.conc, a.per_kind, a.tokens, a.runs) < 1 or a.settle_seconds <= 0:
        p.error('positive counts and settle-seconds required')
    if a.dry_run:
        self_test()
        return
    if not a.exclusive or not a.corpus or not a.out:
        p.error('--exclusive, --corpus, --out required; aggregate metrics cannot attribute shared traffic')
    rows = [json.loads(x) for x in Path(a.corpus).read_text().splitlines() if x.strip()]
    if any(r.get('dry_run') for r in rows):
        p.error('Test corpus cannot be measured')
    pools = {k: [r for r in rows if r['kind'] == k][:a.per_kind] for k in ('code', 'prose', 'tool')}
    if any(len(v) != a.per_kind for v in pools.values()):
        p.error('Insufficient held-out rows for each kind')
    base = a.url.rstrip('/').removesuffix('/v1')
    metrics_model = a.metrics_model or a.model
    with Path(a.out).open('x') as output:
        for run in range(a.runs):
            for kind, subset in pools.items():
                before = settled(base, metrics_model, a.settle_seconds)
                start = time.monotonic()
                with ThreadPoolExecutor(max_workers=a.conc) as pool:
                    futures = [pool.submit(complete, base, a.model, row, a.tokens, run * 1000 + i)
                               for i, row in enumerate(subset)]
                    usages = [f.result() for f in futures]
                elapsed = time.monotonic() - start
                after = settled(base, metrics_model, a.settle_seconds)
                result = dict(kind=kind, run=run, conc=a.conc, ids=[r['id'] for r in subset],
                              wall_seconds=elapsed, usage=usages, **summarize(before, after))
                line = json.dumps(result)
                print(line, flush=True)
                output.write(line + '\n')
                output.flush()


if __name__ == '__main__':
    main()
