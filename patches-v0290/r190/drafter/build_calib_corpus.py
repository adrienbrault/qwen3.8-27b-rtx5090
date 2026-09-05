#!/usr/bin/env python3
"""Usage: python build_calib_corpus.py DIR... --tokenizer /target --out corpus.jsonl [--url http://localhost:8029]."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import urllib.request

KINDS = ('code', 'tool', 'prose')
CODE = {'.py', '.rs', '.go', '.ts', '.tsx', '.js', '.c', '.h', '.cpp', '.sh', '.diff', '.patch'}


def request_json(url, body, api_key=None):
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = 'Bearer ' + api_key
    req = urllib.request.Request(url, json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=3600) as response:
        return json.load(response)


class DryTokenizer:
    """Test double only; deliberately NOT a Qwen chat template or tokenizer."""
    def apply_chat_template(self, messages, tools=None, **kwargs):
        return json.dumps({'messages': messages, 'tools': tools}, ensure_ascii=False)

    def encode(self, text, **kwargs):
        return text.split()


def render(tokenizer, messages, tools=None):
    kw = dict(tokenize=False, add_generation_prompt=False)
    if tools:
        kw['tools'] = tools
    text = tokenizer.apply_chat_template(messages, **kw)
    return text, len(tokenizer.encode(text, add_special_tokens=False))


def sources(directories):
    records = []
    seen = set()
    for directory in directories:
        root = Path(directory)
        if not root.is_dir():
            raise ValueError(f'Not a directory: {root}')
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.is_symlink() or any(p.startswith('.') for p in path.relative_to(root).parts):
                continue
            if path.suffix not in CODE | {'.txt', '.md', '.json', '.jsonl'} or path.stat().st_size > 8_000_000:
                continue
            try:
                content = path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                continue
            if path.suffix in {'.json', '.jsonl'}:
                values = [json.loads(x) for x in content.splitlines() if x.strip()] if path.suffix == '.jsonl' else [json.loads(content)]
                for value in values:
                    if not isinstance(value, dict) or not value.get('messages') or not value.get('tools'):
                        continue
                    if not any(m.get('role') == 'tool' for m in value['messages']):
                        raise ValueError(f'{path}: tool transcript needs actual tool results')
                    records.append(dict(kind='tool', messages=value['messages'], tools=value['tools'], source=str(path)))
            else:
                digest = hashlib.sha256(content.encode()).hexdigest()
                if digest in seen or not content.strip():
                    continue
                seen.add(digest)
                kind = 'code' if path.suffix in CODE else 'prose'
                records.append(dict(kind=kind, content=content, source=str(path)))
    return records


def inline_sources():
    tools = [{'type': 'function', 'function': {'name': 'read_file', 'description': 'Read a file',
              'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}}]
    return [dict(kind='code', content='def add(a, b):\n    return a + b\n' * 50, source='inline.py'),
            dict(kind='prose', content='Explain how rivers shape a landscape. ' * 50, source='inline.txt'),
            dict(kind='tool', source='inline.jsonl', tools=tools, messages=[
                {'role': 'user', 'content': 'Inspect math.py and explain its contents.'},
                {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call_1', 'type': 'function',
                 'function': {'name': 'read_file', 'arguments': '{"path":"math.py"}'}}]},
                {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'def add(a,b): return a+b'},
                {'role': 'assistant', 'content': 'It adds two values.'}])]


def build(records, tokenizer, count, minimum, maximum, seed, endpoint=None, model=None, api_key=None):
    rng = random.Random(seed)
    pools = {k: [r for r in records if r['kind'] == k] for k in KINDS}
    if any(not p for p in pools.values()):
        raise ValueError('Need local code, prose, and JSON/JSONL tool transcripts with tools schemas')
    seen = set()
    rows = []
    for index in range(count):
        kind = KINDS[index % 3]
        for attempt in range(200):
            source = rng.choice(pools[kind])
            if kind == 'tool':
                messages = list(source['messages'])
            else:
                # Token bounds apply to the final template. Slice source text, never tool JSON.
                content = source['content']
                start = rng.randrange(max(1, len(content) // 2))
                content = content[start:]
                instruction = 'Review this code or diff and explain it:\n' if kind == 'code' else 'Discuss this passage:\n'
                lo, hi = 0, len(content)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    _, n = render(tokenizer, [{'role': 'user', 'content': instruction + content[:mid]}])
                    if n <= maximum - (640 if endpoint else 0):
                        lo = mid
                    else:
                        hi = mid - 1
                messages = [{'role': 'user', 'content': instruction + content[:lo]}]
            if endpoint:
                # Reserve generation space; skip too-long prompts rather than truncate messages.
                _, prompt_n = render(tokenizer, messages, source.get('tools'))
                reserve = maximum - prompt_n - 64
                if reserve < 64:
                    continue
                body = dict(model=model, messages=messages, max_tokens=min(512, reserve), temperature=0.6)
                if source.get('tools'):
                    body.update(tools=source['tools'], tool_choice='none')
                reply = request_json(endpoint.rstrip('/').removesuffix('/v1') + '/v1/chat/completions', body, api_key)
                messages = messages + [reply['choices'][0]['message']]
            text, n = render(tokenizer, messages, source.get('tools'))
            digest = hashlib.sha256(text.encode()).hexdigest()
            if minimum <= n <= maximum and digest not in seen:
                seen.add(digest)
                rows.append(dict(id=digest, kind=kind, source=source['source'], messages=messages,
                                 tools=source.get('tools'), text=text, num_tokens=n))
                break
        else:
            raise ValueError(f'Cannot produce unique {kind} sample within {minimum}..{maximum} tokens; add suitable sources')
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directories', nargs='*')
    p.add_argument('--tokenizer')
    p.add_argument('--out', required=True)
    p.add_argument('--count', type=int, default=1024)
    p.add_argument('--min-tokens', type=int, default=1024)
    p.add_argument('--max-tokens', type=int, default=4096)
    p.add_argument('--seed', type=int, default=28)
    p.add_argument('--url')
    p.add_argument('--model', default='qwen3.8-27b')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.count < 3 or not 0 < a.min_tokens <= a.max_tokens:
        p.error('count >= 3 and 0 < min-tokens <= max-tokens required')
    if a.dry_run:
        rows = build(inline_sources(), DryTokenizer(), 3, 1, 1024, a.seed)
    else:
        if not a.tokenizer or not a.directories:
            p.error('--tokenizer (local directory) and source directories required')
        from transformers import AutoTokenizer
        import os
        tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True, trust_remote_code=False)
        rows = build(sources(a.directories), tokenizer, a.count, a.min_tokens, a.max_tokens,
                     a.seed, a.url, a.model, os.environ.get('OPENAI_API_KEY'))
    with Path(a.out).open('x') as f:
        for row in rows:
            row['dry_run'] = a.dry_run
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'R190-28 CORPUS rows={len(rows)} dry_run={a.dry_run}')


if __name__ == '__main__':
    main()
