# P3 -> P1 Batch Integration Validation

## Scope

This run validates the existing P3 retrieval handoff against the local P1 pipeline for a batch FEVER-style workflow.

## Environment / Setup

- P3 workspace: `path/to/project_root/P3`
- Existing lightweight virtualenv reused: `.venv`
- Existing local P1 codebase found at: `path/to/local/P1`
- For safe local integration, a workspace-local copy was created at `artifacts/p1_local`
- One minimal compatibility fix was applied in the local P1 copy only:
  - guard against `None.split()` inside `src/p1/claim_extraction.py`
  - this was required to let the structured extractor finish on retrieval-shaped input

## Exact Commands Run

```bash
mkdir -p artifacts/p1_eval data/processed
```

```bash
.venv/bin/python scripts/export_p1_handoff.py \
  --claims-path "FEVER Dataset/shared_task_dev_sample.jsonl" \
  --top-k 5 \
  --mode hybrid \
  --split dev \
  --limit 30 \
  --config-path config/retrieval.yaml \
  --output-path data/processed/p3_to_p1_batch.json
```

```bash
cp -R path/to/local/P1 artifacts/p1_local
```

```bash
.venv/bin/python artifacts/p1_local/scripts/eval_p3_retrieval_hook.py \
  --input data/processed/p3_to_p1_batch.json \
  --input-kind retrieval_json \
  --limit 30 \
  --extractor-kind structured \
  --entity-backend auto \
  --preview 10 \
  > artifacts/p1_eval_hook.log 2>&1
```

```bash
.venv/bin/python artifacts/p1_local/scripts/export_p5_benchmark.py \
  --dataset retrieval_json \
  --input data/processed/p3_to_p1_batch.json \
  --limit 30 \
  --output artifacts/p5_from_p3.jsonl \
  --preview 5 \
  > artifacts/p5_export.log 2>&1
```

```bash
.venv/bin/python scripts/analyze_p1_fragmentation.py \
  --input artifacts/p5_from_p3.jsonl \
  --output artifacts/p1_eval/fragmentation_analysis.json \
  --max-examples 10
```

## Batch Integration Status

- Batch handoff export succeeded
- P3 payload saved to `data/processed/p3_to_p1_batch.json`
- Payload shape validated as `{ "records": [...] }`
- Record count: `30`
- P1 hook evaluation succeeded on all `30` records
- P1 benchmark export succeeded and produced `artifacts/p5_from_p3.jsonl`

## Outcome A — Cross-Source Pair Coverage

- Records with cross-source candidate pairs: `26`
- Total valid records: `30`
- Cross-source pair coverage: `0.8667`

Interpretation:
- The batch handoff is structurally working
- Most records produce at least one query-vs-evidence candidate pair inside P1

## Outcome B — Decisive NLI Coverage

- Records with decisive cross-source NLI: `1`
- Total valid records: `30`
- Decisive cross-source NLI coverage: `0.0333`

Interpretation:
- P1 is receiving retrieval input correctly
- But the retrieved chunks are rarely turning into decisive entailment / contradiction outcomes
- Most cross-source NLI labels in this run are `neutral`

## Outcome C — Claim Fragmentation Findings

Source:
- `artifacts/p5_from_p3.jsonl`
- `artifacts/p1_eval/fragmentation_analysis.json`

Metrics:
- Claims evaluated: `395`
- Average claim token length: `19.719`
- Claims shorter than 6 tokens: `20` (`0.0506`)
- Obviously broken claims: `15` (`0.0380`)
- Entity-only-like claims: `6` (`0.0152`)
- Title-only-like claims: `20` (`0.0506`)
- Records containing broken claims: `12 / 30` (`0.4000`)

Concrete bad examples:

1. `137334:Soul_Food_-LRB-film-RRB-::chunk-0:sent2`
   `Williams , Vivica A.`
   reasons: `broken_fragment`, `entity_only_like`, `title_only_like`

2. `111897:Food_Network::chunk-0:sent2`
   `-LRB- which owns the remaining 30 % -RRB- .`
   reasons: `broken_fragment`

3. `181634:Mogadishu::chunk-3:sent0`
   `, Design -LRB- TEDx -RRB- conference .`
   reasons: `broken_fragment`, `title_only_like`

4. `204361:query:sent0`
   `The Cretaceous ended.`
   reasons: `broken_fragment`, `entity_only_like`, `title_only_like`

5. `204361:Cretaceous::chunk-1:sent0`
   `plants , appeared .`
   reasons: `broken_fragment`, `entity_only_like`, `title_only_like`

6. `89891:Drake_Bell::chunk-1:sent0`
   `Spider-Man on Disney XD .`
   reasons: `title_only_like`

7. `54168:Mogadishu::chunk-3:sent0`
   `, Design -LRB- TEDx -RRB- conference .`
   reasons: `broken_fragment`, `title_only_like`

8. `192714:Food_Network::chunk-0:sent2`
   `-LRB- which owns the remaining 30 % -RRB- .`
   reasons: `broken_fragment`

Interpretation:
- The main failure mode is not handoff schema mismatch
- The main failure mode is fragmentation/noise inside retrieved chunks, which produces poor structured claims in P1
- This aligns with the very low decisive NLI coverage

## Primary Next Action

`2) tune chunking granularity`

Why this is the single best next step:
- Cross-source pair coverage is already fairly high (`0.8667`), so the handoff and basic retrieval connectivity are not the bottleneck
- Source diversification is not the first-order blocker in this batch
- The clearest bottleneck is chunk quality for P1 claim extraction:
  - broken fragments
  - title/entity-only snippets
  - low decisive NLI despite many candidate pairs

Recommended P3 focus:
- make chunks more sentence-aware
- reduce fragment carry-over at chunk boundaries
- bias retrieval output toward compact evidence-bearing spans rather than long noisy windows
