#!/usr/bin/env python3
"""R156 fidelity corpus builder — fixed, deterministic, domain-mixed.

Every ladder arm must score the SAME text, so this emits a corpus once and records a
manifest (sha256 + per-domain counts) that pins it. Rebuilding with the same args
reproduces it byte-for-byte.

Doc length is capped so that arm A (bf16) can score every doc: bf16 at TP2 leaves only
~2.6-3.7 GB/card after 55.6 GB of weights, so max-len is ~8K. Docs are capped below
that; positions within a doc give the depth axis (compounding shows as flip rate
rising with position).

Domains (each optional — missing sources are skipped and noted in the manifest):
  code       .py sources (deterministic file list, sorted)
  tech-prose local technical markdown (results/notes shipped alongside)
  prose      wikitext-103 (HF datasets, pinned split)
  reasoning  gsm8k main train (question + reference solution)

Usage:
  build_fidelity_corpus.py --out corpus.jsonl --manifest corpus-manifest.json \
      [--target-tokens 800000] [--max-doc-chars 24000] \
      [--code-dir /usr/lib/python3/dist-packages] [--tech-dir /srv/qwen5090/r156-text]
"""
import argparse, glob, hashlib, json, os, sys

CHARS_PER_TOK = 3.5  # rough; actual token counts come back from the scorer


def chunk(text, max_chars):
    """Split on paragraph boundaries into <=max_chars pieces (never mid-paragraph)."""
    paras = [p for p in text.split("\n\n") if p.strip()]
    out, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 > max_chars:
            if cur.strip():
                out.append(cur.strip())
            cur = p if len(p) <= max_chars else p[:max_chars]
        else:
            cur += ("\n\n" if cur else "") + p
    if cur.strip():
        out.append(cur.strip())
    return out


def from_files(paths, max_chars, min_chars=800):
    docs = []
    for p in sorted(paths):
        try:
            with open(p, encoding="utf-8", errors="ignore") as f:
                t = f.read()
        except OSError:
            continue
        if len(t) < min_chars:
            continue
        docs.extend(chunk(t, max_chars))
    return docs


def from_hf(repo, parquet_path, field, max_chars, limit_chars):
    """Pinned HF slice read straight from parquet.

    Deliberately avoids datasets.load_dataset: on this box (datasets 5.0.1 +
    huggingface_hub 1.28) it fails with "Invalid HF URI ... .huggingface.yaml".
    hf_hub_download + pyarrow is both more robust and more precisely pinned.
    """
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        return [], f"missing dep: {e}"
    try:
        p = hf_hub_download(repo_id=repo, filename=parquet_path, repo_type="dataset")
        rows = pq.read_table(p).to_pylist()
    except Exception as e:  # noqa: BLE001 - offline/gated/missing all mean "skip"
        return [], f"load failed: {str(e)[:120]}"
    buf, docs, total = "", [], 0
    for row in rows:
        t = row.get(field) if isinstance(field, str) else " ".join(
            str(row.get(k, "")) for k in field
        )
        if not t or not t.strip():
            continue
        buf += t.strip() + "\n\n"
        if len(buf) >= max_chars:
            docs.append(buf[:max_chars].strip())
            total += max_chars
            buf = ""
            if total >= limit_chars:
                break
    if buf.strip() and total < limit_chars:
        docs.append(buf.strip())
    return docs, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--target-tokens", type=int, default=800000)
    ap.add_argument("--max-doc-chars", type=int, default=24000, help="~7K tokens, fits bf16 8K max-len")
    ap.add_argument("--code-dir", default="/usr/lib/python3/dist-packages")
    ap.add_argument("--tech-dir", default="/srv/qwen5090/r156-text")
    a = ap.parse_args()

    target_chars = int(a.target_tokens * CHARS_PER_TOK)
    per_domain = target_chars // 4
    domains, notes = {}, {}

    code_files = sorted(glob.glob(os.path.join(a.code_dir, "**", "*.py"), recursive=True))[:1500]
    d = from_files(code_files, a.max_doc_chars)
    domains["code"] = d[: max(1, per_domain // a.max_doc_chars)] if d else []
    if not d:
        notes["code"] = f"no .py under {a.code_dir}"

    tech_files = sorted(glob.glob(os.path.join(a.tech_dir, "**", "*.md"), recursive=True))
    d = from_files(tech_files, a.max_doc_chars)
    domains["tech-prose"] = d[: max(1, per_domain // a.max_doc_chars)] if d else []
    if not d:
        notes["tech-prose"] = f"no .md under {a.tech_dir}"

    d, err = from_hf("Salesforce/wikitext",
                     "wikitext-103-raw-v1/test-00000-of-00001.parquet",
                     "text", a.max_doc_chars, per_domain)
    domains["prose"] = d
    if err:
        notes["prose"] = err

    d, err = from_hf("openai/gsm8k",
                     "main/train-00000-of-00001.parquet",
                     ["question", "answer"], a.max_doc_chars, per_domain)
    domains["reasoning"] = d
    if err:
        notes["reasoning"] = err

    # deterministic interleave so every domain is represented at every depth of the run
    ordered, i = [], 0
    while any(i < len(v) for v in domains.values()):
        for name in sorted(domains):
            if i < len(domains[name]):
                ordered.append((name, domains[name][i]))
        i += 1

    n_chars = 0
    with open(a.out, "w") as f:
        for idx, (dom, text) in enumerate(ordered):
            f.write(json.dumps({"id": idx, "domain": dom, "text": text}) + "\n")
            n_chars += len(text)

    h = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    manifest = {
        "corpus": os.path.abspath(a.out),
        "sha256": h,
        "docs": len(ordered),
        "chars": n_chars,
        "est_tokens": int(n_chars / CHARS_PER_TOK),
        "max_doc_chars": a.max_doc_chars,
        "domain_counts": {k: len(v) for k, v in domains.items()},
        "skipped": notes,
        "note": "Every ladder arm MUST score this exact file; the comparator verifies "
                "per-position token alignment and refuses mismatched arms.",
    }
    with open(a.manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    if manifest["est_tokens"] < a.target_tokens * 0.5:
        print(f"\nWARNING: only ~{manifest['est_tokens']:,} est tokens "
              f"(target {a.target_tokens:,}) — certain-bucket CIs will be wide. "
              f"Add sources or raise --target-tokens.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
