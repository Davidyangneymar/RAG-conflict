#!/usr/bin/env python
"""
Build BM25 index from FEVER-style Wikipedia JSONL corpus.
Place this script in P4/scripts/.
Input: FEVER Dataset/wiki_pages_matched_sample.jsonl (relative to project root)
Output: P4/data_backup/bm25_corpus.json (used by P4 local BM25 retriever)
"""

import json
import re
from pathlib import Path

def tokenize(text: str):
    """Simple tokenizer: lowercase, keep only a-z0-9."""
    return re.findall(r'\b[a-z0-9]+\b', text.lower())

def get_project_root() -> Path:
    """Find project root by locating P3/ directory upwards from this script."""
    current = Path(__file__).resolve().parent
    for parent in current.parents:
        if (parent / "P3").exists():
            return parent
    raise RuntimeError("Cannot locate project root (no P3/ directory found)")

def main():
    project_root = get_project_root()
    wiki_path = project_root / "FEVER Dataset" / "wiki_pages_matched_sample.jsonl"
    output_dir = project_root / "P4" / "data_backup"

    if not wiki_path.exists():
        print(f"Wikipedia file not found: {wiki_path}")
        print("Please ensure the file exists under 'FEVER Dataset/wiki_pages_matched_sample.jsonl'")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_ids = []
    tokenized_corpus = []
    print(f"Loading documents from {wiki_path}...")
    with open(wiki_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # Optional: limit for testing, e.g. if i >= 10000: break
            data = json.loads(line)
            doc_id = data.get("id") or data.get("doc_id")
            text = data.get("text") or data.get("full_text")
            if not doc_id or not text:
                continue
            chunk_ids.append(doc_id)
            tokenized_corpus.append(tokenize(text))
            if (i + 1) % 100000 == 0:
                print(f"Processed {i+1} documents...")

    print(f"Loaded {len(chunk_ids)} documents.")
    index_data = {
        "chunk_ids": chunk_ids,
        "tokenized_corpus": tokenized_corpus
    }
    out_file = output_dir / "bm25_corpus.json"
    with open(out_file, 'w') as f:
        json.dump(index_data, f)
    print(f"BM25 index written to {out_file}")

if __name__ == "__main__":
    main()