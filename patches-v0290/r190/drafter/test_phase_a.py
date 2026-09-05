#!/usr/bin/env python3
"""Usage: python -m unittest discover -s deliver -p test_phase_a.py (no ML dependencies)."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
from build_calib_corpus import build, sources, inline_sources, DryTokenizer
from measure_acceptance import parse_metrics, summarize, COUNTERS, GAUGES, self_test
from recalibrate_drafter import parameter_ledger, weight_headers, REFERENCE


class OfflineTests(unittest.TestCase):
    def test_metrics(self):
        self_test()
        with self.assertRaises(ValueError):
            parse_metrics('', 'm')
        lines = '\n'.join(f'{k}{{model_name="m"}} 0' for k in COUNTERS + GAUGES)
        parsed = parse_metrics(lines, 'm')
        with self.assertRaises(ValueError):
            summarize(parsed, parsed)
        with self.assertRaises(ValueError):
            parse_metrics(lines + '\n' + lines, 'm')
        with self.assertRaises(ValueError):
            parse_metrics(lines, 'other')

    def test_corpus_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = inline_sources()
            (root / 'example.py').write_text(samples[0]['content'])
            (root / 'example.md').write_text(samples[1]['content'])
            (root / 'calls.jsonl').write_text(json.dumps(samples[2]) + '\n')
            (root / 'package.json').write_text('{"name":"ignored"}')
            records = sources([tmp])
            a = build(records, DryTokenizer(), 3, 1, 1024, 28)
            b = build(records, DryTokenizer(), 3, 1, 1024, 28)
            self.assertEqual(a, b)
            self.assertEqual({r['kind'] for r in a}, {'code', 'tool', 'prose'})
            tool = next(r for r in a if r['kind'] == 'tool')
            self.assertEqual(tool['tools'], samples[2]['tools'])
            self.assertEqual(tool['messages'], samples[2]['messages'])
            with self.assertRaises(ValueError):
                build(records, DryTokenizer(), 3, 10000, 11000, 28)

    def test_metadata(self):
        ledger = parameter_ledger(json.loads(REFERENCE.read_text()))
        self.assertEqual(ledger['full_parameters'], 4467201280)
        self.assertEqual(ledger['features_bytes_per_token'], 51200)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.safetensors'
            header = json.dumps({'fc.weight': {'dtype': 'BF16', 'shape': [2, 3], 'data_offsets': [0, 12]}}).encode()
            path.write_bytes(struct.pack('<Q', len(header)) + header + bytes(12))
            self.assertEqual(weight_headers(tmp)['parameters'], 6)
            path.write_bytes(b'bad')
            with self.assertRaises(ValueError):
                weight_headers(tmp)


if __name__ == '__main__':
    unittest.main()
