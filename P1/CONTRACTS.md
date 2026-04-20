# P1 对外接口契约

本文件包含 P1 与下游模块的全部数据交换契约。

## 目录

- [P1 → P2 输出契约](#p1--p2-输出契约)
- [P3 → P1 检索输入契约](#p3--p1-检索输入契约)
- [P1 → P5 Benchmark 导出契约](#p1--p5-benchmark-导出契约)

---

## P1 → P2 输出契约

Last updated: 2026-04-16

This file freezes what `P2` should consume from `P1` after claim extraction,
blocking, and NLI.

### Entry Point

Use:

- `src/p1/handoff.py`
- `pipeline_output_to_p2_payload()`

This converts `PipelineOutput` into a stable JSON-friendly payload.

### Top-Level Shape

```json
{
  "sample_id": "string",
  "claims": [],
  "candidate_pairs": [],
  "nli_results": []
}
```

### `claims`

Each claim contains:

```json
{
  "claim_id": "string",
  "text": "raw extracted claim text",
  "source_doc_id": "string",
  "source_chunk_id": "string",
  "entities": ["..."],
  "subject": "optional",
  "relation": "optional",
  "object": "optional",
  "qualifier": "optional",
  "time": "optional",
  "polarity": "positive|negative|uncertain",
  "certainty": 0.0,
  "metadata": {},
  "source_metadata": {
    "role": "query|retrieved_evidence|headline|body|..."
  }
}
```

`P2` should treat `claim_id` as the stable key.

### `candidate_pairs`

Each candidate pair contains:

```json
{
  "claim_a_id": "string",
  "claim_b_id": "string",
  "entity_overlap": ["..."],
  "lexical_similarity": 0.0,
  "embedding_similarity": 0.0
}
```

These are the claim pairs that passed blocking and were forwarded to NLI.

### `nli_results`

Each NLI result contains:

```json
{
  "claim_a_id": "string",
  "claim_b_id": "string",
  "label": "entailment|contradiction|neutral",
  "entailment_score": 0.0,
  "contradiction_score": 0.0,
  "neutral_score": 0.0,
  "is_bidirectional": false,
  "metadata": {}
}
```

`P2` should join `nli_results` back to `claims` using `claim_a_id` and
`claim_b_id`, not by list order.

### Practical Reading Rule For P2

If `P2` only needs the minimum stable subset, read:

- from `claims`:
  - `claim_id`
  - `text`
  - `subject`
  - `relation`
  - `object`
  - `qualifier`
  - `time`
  - `source_metadata.role`
- from `nli_results`:
  - `claim_a_id`
  - `claim_b_id`
  - `label`

The other fields are support signals, not mandatory dependencies.

---

## P3 → P1 检索输入契约

Last updated: 2026-04-16

This file freezes the retrieval-shaped input contract that `P1` can consume for
Phase `3.5 End-to-end with P3 retrieval output`.

### Required Top-Level Shape

`P1` accepts either:

1. A single JSON object
2. A JSON array of objects
3. A JSON object with a `records` array

Each object must follow this shape:

```json
{
  "sample_id": "string",
  "query": "claim text or user query",
  "label": "optional gold label",
  "metadata": {
    "dataset": "optional",
    "split": "optional",
    "speaker": "optional",
    "claim_date": "optional"
  },
  "retrieved_chunks": [
    {
      "chunk_id": "string",
      "text": "retrieved evidence text",
      "rank": 1,
      "retrieval_score": 1.0,
      "source_url": "optional source url",
      "source_medium": "optional medium",
      "metadata": {
        "title": "optional",
        "source_doc_id": "optional",
        "question": "optional provenance field"
      }
    }
  ]
}
```

### What P1 Does With It

- `query` becomes one `ChunkInput` with `metadata.role = "query"`
- every item in `retrieved_chunks` becomes one `ChunkInput` with
  `metadata.role = "retrieved_evidence"`
- the current pipeline then runs:
  1. claim extraction
  2. blocking
  3. pairwise NLI

This is implemented in:

- `src/p1/data/retrieval.py`
- `src/p1/pipeline.py`
- `scripts/eval_p3_retrieval_hook.py`

### Current Simulation Source

Until the real `P3` module emits this JSON directly, `P1` simulates retrieval
input from AVeriTeC:

- `query` = `record["claim"]`
- `retrieved_chunks[].text` = `questions[].answers[].answer`
- `retrieved_chunks[].metadata.question` = source question text

This is implemented by `averitec_record_to_retrieval_input()` in
`src/p1/data/averitec.py`.

### Current Limits

- This contract is now stable enough for team handoff.
- The current validation uses AVeriTeC answer snippets, not the full external
  knowledge store.
- When the real `P3` retrieval output is available, `P1` should consume it
  without schema changes if it matches the shape above.

---

## P1 → P5 Benchmark 导出契约

Last updated: 2026-04-16

This file freezes the benchmark-facing export shape that `P5` can score or
compare without depending on `P1` internals.

### Export Entry Point

Use:

- `scripts/export_p5_benchmark.py`

Supported datasets:

- `averitec`
- `fnc1`

The script can print a preview summary and optionally write JSONL rows.

### One JSONL Row Per Record

Each exported row has this shape:

```json
{
  "sample_id": "string",
  "dataset": "averitec|fnc1",
  "split": "optional",
  "gold_label": "entailment|contradiction|neutral|Supported|Refuted|...",
  "query": "query or claim text",
  "retrieved_chunk_count": 0,
  "claim_count": 0,
  "candidate_pair_count": 0,
  "cross_source_pair_count": 0,
  "predicted_label": "entailment|contradiction|neutral|null",
  "best_entailment_score": 0.0,
  "best_contradiction_score": 0.0,
  "best_neutral_score": 0.0,
  "best_entailment_pair": {
    "claim_a_id": "string",
    "claim_b_id": "string"
  },
  "best_contradiction_pair": {
    "claim_a_id": "string",
    "claim_b_id": "string"
  },
  "best_neutral_pair": {
    "claim_a_id": "string",
    "claim_b_id": "string"
  },
  "claims": [],
  "cross_source_nli_results": []
}
```

### Practical Scoring Rule

If `P5` only wants the minimum stable benchmark view, it can read:

- `sample_id`
- `gold_label`
- `predicted_label`
- `best_entailment_score`
- `best_contradiction_score`
- `best_neutral_score`
- `cross_source_pair_count`

### Notes

- For `fnc1`, the export uses the current P1 evidence unit setting via
  `sample_to_retrieval_input()` and defaults to `top2_span`.
- For `averitec`, the export uses `claim` as query and
  `questions[].answers[].answer` as retrieved evidence chunks.
- `predicted_label` is the strongest cross-source NLI label currently produced
  by P1 for that record.
