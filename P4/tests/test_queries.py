#!/usr/bin/env python
"""
Read test queries from 'queries.txt' (one per line) in the same directory,
send requests to P4 /api/v1/query, and write results to 'queries_output.jsonl'.

Usage: python test_queries.py [--input INPUT] [--output OUTPUT] [--url URL] [--top_k K]
"""

import sys
import json
import requests
from pathlib import Path

def main():
    # Default files relative to script location
    script_dir = Path(__file__).parent
    input_file = script_dir / "queries.txt"
    output_file = script_dir / "queries_output.jsonl"
    url = "http://localhost:8000/api/v1/query"
    top_k = 3

    # Simple argument override (optional)
    args = sys.argv[1:]
    if "--input" in args:
        idx = args.index("--input") + 1
        if idx < len(args):
            input_file = Path(args[idx])
    if "--output" in args:
        idx = args.index("--output") + 1
        if idx < len(args):
            output_file = Path(args[idx])
    if "--url" in args:
        idx = args.index("--url") + 1
        if idx < len(args):
            url = args[idx]
    if "--top_k" in args:
        idx = args.index("--top_k") + 1
        if idx < len(args):
            top_k = int(args[idx])

    if not input_file.exists():
        print(f"Input file not found: {input_file}", file=sys.stderr)
        print("Please create test_queries.txt with one query per line.")
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(queries)} queries from {input_file}")
    print(f"Sending to {url} with top_k={top_k}")

    results = []
    for idx, q in enumerate(queries, 1):
        payload = {"text": q, "top_k": top_k}
        source = None
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            source = data.get("retrieval_source")
            results.append({
                "index": idx,
                "query": q,
                "answer": data.get("answer"),
                "conflict_type": data.get("conflict_type"),
                "resolution_policy": data.get("resolution_policy"),
                "evidence": data.get("evidence", []),
                "abstained": data.get("abstained", False),
                "confidence": data.get("confidence"),
                "retrieval_source": source
            })
            print(f"{idx}. {q[:50]}... -> {data.get('answer', '')[:50]}")
        except Exception as e:
            print(f"{idx}. ERROR: {e}", file=sys.stderr)
            results.append({"index": idx, "query": q, "error": str(e)})

    if source is not None:
        output_file = script_dir / f"queries_output_{source}.jsonl"
    print(f"Writing results to {output_file}")

    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Done. Results saved to {output_file}")

if __name__ == "__main__":
    main()