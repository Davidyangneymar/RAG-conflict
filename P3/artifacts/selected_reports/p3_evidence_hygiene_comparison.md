# P3 Evidence Hygiene Comparison

## Scope
- Baseline: chunking v2 only.
- Iteration: chunking v2 + lightweight evidence-hygiene reranking in the P3 export path.
- Constraint: no handoff schema changes, no P1/P2 core logic changes, no new retrieval or NLI models.
- Current workspace baseline remains chunking v2 only. The evidence-hygiene code is kept available but is not left enabled by default.

## Exact Commands Run
Note: the hygiene experiment was run with `enable_evidence_hygiene: true` temporarily toggled in `config/retrieval.yaml`. The workspace default was restored to `false` after validation.

```bash
.venv/bin/python scripts/export_p1_handoff.py \
  --claims-path "FEVER Dataset/shared_task_dev_sample.jsonl" \
  --top-k 5 \
  --mode hybrid \
  --split dev \
  --limit 30 \
  --config-path config/retrieval.yaml \
  --output-path data/processed/p3_to_p1_batch_hygiene.json

.venv/bin/python artifacts/p1_local/scripts/eval_p3_retrieval_hook.py \
  --input data/processed/p3_to_p1_batch_hygiene.json \
  --input-kind retrieval_json \
  --limit 30 \
  --extractor-kind structured \
  --entity-backend auto \
  --preview 10 \
  > artifacts/p1_eval_hook_hygiene.log 2>&1

.venv/bin/python artifacts/p1_local/scripts/export_p5_benchmark.py \
  --dataset retrieval_json \
  --input data/processed/p3_to_p1_batch_hygiene.json \
  --limit 30 \
  --output artifacts/p5_from_p3_hygiene.jsonl \
  --preview 5 \
  > artifacts/p5_export_hygiene.log 2>&1

.venv/bin/python scripts/analyze_p1_fragmentation.py \
  --input artifacts/p5_from_p3_hygiene.jsonl \
  --output artifacts/p1_eval/fragmentation_analysis_hygiene.json \
  --max-examples 10

.venv/bin/python scripts/analyze_p1_query_diagnostics.py \
  --benchmark-input artifacts/p5_from_p3_hygiene.jsonl \
  --handoff-input data/processed/p3_to_p1_batch_hygiene.json \
  --output artifacts/p1_eval/query_claim_diagnostics_hygiene.json
```

## Integration Status
- Batch export succeeded.
- P1 eval hook succeeded.
- P5 benchmark export succeeded.
- Fragmentation analysis succeeded.
- Query-side diagnostics succeeded.

## Headline Result
The hygiene pass made selected evidence somewhat cleaner, but it did not improve decisive NLI coverage and slightly regressed broken-claim indicators. It should stay as experimental code, not the default retrieval-export path.

## Metric Comparison

### Cross-Source Pair Coverage
| Metric | Chunking v2 Baseline | Evidence Hygiene Iteration | Delta |
| --- | ---: | ---: | ---: |
| Records with cross-source pairs | 24 / 30 | 24 / 30 | 0 |
| Coverage ratio | 0.8000 | 0.8000 | 0 |
| Avg cross-source pairs per record | 1.5667 | 1.5667 | 0 |

Assessment:
- Coverage did not materially collapse.
- Hygiene reranking was at least safe on this metric.

### Decisive Cross-Source NLI Coverage
| Metric | Chunking v2 Baseline | Evidence Hygiene Iteration | Delta |
| --- | ---: | ---: | ---: |
| Records with decisive cross-source NLI | 1 / 30 | 1 / 30 | 0 |
| Coverage ratio | 0.0333 | 0.0333 | 0 |

Assessment:
- No improvement.
- The current evidence-hygiene pass did not unlock more decisive NLI outcomes.

### Broken Claim Indicators
| Metric | Chunking v2 Baseline | Evidence Hygiene Iteration | Delta |
| --- | ---: | ---: | ---: |
| Claims evaluated | 426 | 418 | -8 |
| Avg claim token length | 22.3568 | 22.0000 | -0.3568 |
| Short-claim ratio (`<6` tokens) | 0.0235 | 0.0287 | +0.0052 |
| Broken-claim ratio | 0.0117 | 0.0144 | +0.0027 |
| Entity-only-like ratio | 0.0070 | 0.0096 | +0.0026 |
| Title-only-like ratio | 0.0188 | 0.0239 | +0.0051 |
| Records with broken claims | 5 / 30 | 5 / 30 | 0 |

Assessment:
- Broken-claim indicators did not improve further.
- The hygiene pass slightly worsened fragment-like claim behavior, even though it reduced some noisy chunk forms upstream.

### Evidence Noise Indicators
Measured by re-scoring the final exported chunks with the same hygiene heuristic.

| Metric | Chunking v2 Baseline | Evidence Hygiene Iteration | Delta |
| --- | ---: | ---: | ---: |
| Avg chunk hygiene penalty | 0.1192 | 0.0958 | -0.0234 |
| Flagged chunk ratio | 0.6000 | 0.4833 | -0.1167 |
| Parenthetical-heavy selected chunks | 35 | 28 | -7 |

Assessment:
- Selected evidence got cleaner on average.
- The reduction was mostly in parenthetical-heavy chunks.
- That cleanliness gain did not translate into better downstream NLI decisions.

## Concrete Examples

### Evidence Noise Reduced
1. `sample_id=111897`
   Baseline second chunk: `Food_Network::chunk-0`, heavily parenthetical and ownership-clause heavy.
   Hygiene iteration second chunk: `Telemundo::chunk-1`, a cleaner within-doc continuation sentence.
   Effect: lower syntax noise, but no NLI gain.

2. `sample_id=166846`
   Baseline second chunk: `Drake_Bell::chunk-0`, biography lead with parenthetical intro.
   Hygiene iteration second chunk: `Drake_Bell::chunk-2`, later release-history chunk.
   Effect: less parenthetical lead noise, but it introduced another short title-like fragment (`Ready Steady Go !`) downstream.

3. Batch-level effect:
   Parenthetical-heavy selected chunks dropped from `35` to `28`.
   Effect: hygiene is doing what it was designed to do at the text-shape level.

### Still Failing or Regressing
4. `sample_id=129441`
   Baseline second chunk: `Saxony::chunk-0`, on-topic but parenthetical-heavy.
   Hygiene iteration second chunk: `Soul_Food_(film)::chunk-1`, cleaner syntax but semantically off-topic.
   Effect: evidence got cleaner but relevance got worse.

5. `sample_id=18708`
   Baseline second chunk: `Charles_Manson::chunk-0` plus `Saxony::chunk-1`.
   Hygiene iteration replaced the second chunk with `Magic_Johnson::chunk-2`.
   Effect: cleaner sentence form, weaker topical fit.

6. `sample_id=64721`
   Baseline: both chunks stayed on Aristotle.
   Hygiene iteration swapped one Aristotle chunk for `Charles_Manson::chunk-2`.
   Effect: syntax hygiene overrode topical usefulness.

7. `sample_id=166846`
   New bad examples appeared:
   - `under indie label Surfdog Records .`
   - `Ready Steady Go !`
   Effect: less parenthetical noise did not mean less fragmentation after extraction.

8. `sample_id=142454`
   Query claim is still `Advertising is a personal message.`
   Evidence hygiene had zero penalty on the selected evidence, but the record still stayed neutral.
   Effect: this points away from evidence noise and toward short query semantics / downstream extraction.

## Query-Side Short Claim Diagnostics
- Records with short query claims: `7 / 30` (`0.2333`)
- Decisive short-query records: `0 / 7`
- Decisive non-short-query records: `1 / 23`
- Short query + poor evidence records: `4 / 7`
- Dominant recurring short-query patterns:
  - `very_short_query` (`7`)
  - `copular_fact` (`4`)
  - `location_or_time_phrase` (`3`)

Interpretation:
- Short query claims are consistently bad for decisive NLI in this sample.
- Even when evidence hygiene is acceptable, short query claims such as `Saxony is in Ireland.` or `Advertising is a personal message.` still fail to produce decisive outcomes.
- Evidence noise is real, but it is no longer the clearest dominant blocker.

## Bottleneck Diagnosis
Most likely dominant bottleneck now: `query-side short claims`.

Why:
- Evidence hygiene improved the exported chunk set on its own metric.
- Cross-source pair coverage held steady.
- Decisive NLI still did not move.
- All 7 short-query records remained non-decisive, including cases with low hygiene penalties.

Secondary bottleneck:
- residual evidence noise, especially parenthetical wiki syntax and list/appositive sentences.
- However, this now looks secondary rather than primary.

## Recommendation
- Keep chunking v2 as the baseline.
- Keep the evidence-hygiene code available behind config, but do not enable it by default.
- The next focused P3 investigation should target query-side short-claim diagnostics and claimability, not another chunking redesign.
