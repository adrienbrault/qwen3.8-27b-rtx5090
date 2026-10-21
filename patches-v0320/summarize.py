#!/usr/bin/env python3
"""Describe a completed screen; no promotion decision or qualification claim."""
import argparse
import json
from pathlib import Path


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return xs[int((len(xs)-1)*q)] if xs else None


def load(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def describe(root):
    out = {'scope': 'whole-configuration screen; 0.31 BA4 versus 0.32 native BA4 off',
           'gpu_qualification_complete': False, 'arms': {}}
    for arm in ['A', 'B']:
        d = root / arm
        record = {'decode': []}
        for f in sorted(d.glob('decode-*.jsonl')):
            record['decode'].extend(x['summary'] for x in load(f))
        if (d/'agents.jsonl').exists():
            rows = load(d/'agents.jsonl')
            good = [x for x in rows if x['ok']]
            complete = len(rows) == 18 and len(good) == 18
            timings = {}
            for label, subset in [('cold', [x for x in good if x['step'] == 0]),
                                  ('returning', [x for x in good if x['step'] > 0]),
                                  ('all', good)]:
                timings[label] = {
                    'requests': len(subset),
                    'ttft_median_s': pct([x['ttft_s'] for x in subset], .5),
                    'ttft_p95_s': pct([x['ttft_s'] for x in subset], .95),
                    'decode_tps_median': pct([x.get('decode_tps') for x in subset], .5),
                    'wall_median_s': pct([x['wall_s'] for x in subset], .5),
                    'sse_gap_max_s': max((x.get('sse_gap_max_s') or 0 for x in subset), default=None),
                    'request_sse_gap_p95_median_s': pct([x.get('sse_gap_p95_s') for x in subset], .5)}
            if good:
                span = max(x['t_submit']+x['wall_s'] for x in good)-min(x['t_submit'] for x in good)
                timings['wall_span_s'] = span
                timings['wall_output_tps'] = sum(x['gen_tokens'] for x in good)/span
                timings['prompt_tokens_range'] = [min(x['prompt_tokens'] for x in good), max(x['prompt_tokens'] for x in good)]
                timings['cached_tokens'] = sum(x.get('cached_tokens') or 0 for x in good)
                timings['prompt_tokens'] = sum(x['prompt_tokens'] for x in good)
                timings['note'] = 'Wall output rate includes prefill, tool think time and ramp-up/down. SSE gaps are frame gaps; MTP may emit several tokens per frame.'
            record['agents'] = {'complete': complete, 'failed': len(rows)-len(good), **timings}
        out['arms'][arm] = record
    if (root/'scope.txt').exists():
        out['scope'] = (root/'scope.txt').read_text().strip()
    out['fidelity'] = {f.stem: json.loads(f.read_text()) for f in sorted(root.glob('fidelity-*.json'))}
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('results', type=Path)
    a = p.parse_args()
    print(json.dumps(describe(a.results), indent=2))
