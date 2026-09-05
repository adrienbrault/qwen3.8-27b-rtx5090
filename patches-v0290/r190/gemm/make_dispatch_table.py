#!/usr/bin/env python3
# Usage: python make_dispatch_table.py census.json > dispatch.json  # review before enabling
"""Conservative candidate table: isolated measured M only; unmeasured M use baseline."""
import argparse
import json
import math
import sys

SHARED = {'CutlassNvFp4LinearKernel', 'FlashInferCutlassNvFp4LinearKernel',
          'FlashInferB12xNvFp4LinearKernel', 'FlashInferCudnnNvFp4LinearKernel'}


def make_table(data, threshold=.05):
    baseline = data['baseline_nvfp4']
    if baseline not in SHARED:
        raise ValueError('baseline layout unsupported by dispatch; do not generate a table')
    rules=[]
    for shape, suffix in [('gate_up', 'gate_up_proj'), ('down', 'down_proj')]:
        rows = [r for r in data['records'] if r['shape']==shape and r['kernel'] in SHARED
                and isinstance(r['us'], (float,int)) and math.isfinite(r['us']) and r['us']>0]
        by_m={}
        for r in rows:
            cell=by_m.setdefault(r['M'], {})
            if r['kernel'] in cell:
                raise ValueError('duplicate measurements: aggregate runs explicitly before generation')
            cell[r['kernel']]=r['us']
        pattern = r'(?:^|\.)layers\.(?:[0-9]|[1-4][0-9]|5[0-5])\.mlp\.'+suffix+r'$'
        previous=0
        for m, cell in sorted(by_m.items()):
            if baseline not in cell:
                continue
            winner=min(cell, key=cell.get)
            gain=1-cell[winner]/cell[baseline]
            print(f'{shape} M={m}: {winner}, {gain:.1%} vs {baseline}', file=sys.stderr)
            if gain <= threshold or winner==baseline:
                continue
            if m-1>previous:
                rules.append(dict(layer=pattern, m_max=m-1, kernel=baseline))
            rules.append(dict(layer=pattern, m_max=m, kernel=winner))
            previous=m
        if previous:
            rules.append(dict(layer=pattern, m_max=None, kernel=baseline))
    return {'rules': rules}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('census'); args=p.parse_args()
    with open(args.census) as f: data=json.load(f)
    print(json.dumps(make_table(data), indent=2))


if __name__=='__main__': main()
