# P3 Chunking V2 Comparison

## Scope
- Goal: test whether a sentence-aware chunking pass inside P3 reduces downstream claim fragmentation in the existing `P3 -> P1 -> P2` workflow.
- Constraint: no handoff schema changes, no P1/P2 redesign, no new retrieval or NLI algorithms.
- Comparison basis: same 30 FEVER-dev records, same P3 export path, same local P1 hook and P5 benchmark scripts.

## Exact Commands Run
```bash
python3 -m pytest tests/test_splitter.py -q

.venv/bin/python scripts/ingest_corpus.py \
  --input-path "FEVER Dataset/wiki_pages_matched_sample.jsonl" \
  --loader fever_wiki \
  --dataset fever_wiki_sample \
  --config-path config/retrieval.yaml

.venv/bin/python scripts/export_p1_handoff.py \
  --claims-path "FEVER Dataset/shared_task_dev_sample.jsonl" \
  --top-k 5 \
  --mode hybrid \
  --split dev \
  --limit 30 \
  --config-path config/retrieval.yaml \
  --output-path data/processed/p3_to_p1_batch_v2.json

.venv/bin/python artifacts/p1_local/scripts/eval_p3_retrieval_hook.py \
  --input "data/processed/p3_to_p1_batch_v2.json" \
  --input-kind retrieval_json \
  --limit 30 \
  --extractor-kind structured \
  --entity-backend auto \
  --preview 10 \
  > artifacts/p1_eval_hook_v2.log 2>&1

.venv/bin/python artifacts/p1_local/scripts/export_p5_benchmark.py \
  --dataset retrieval_json \
  --input "data/processed/p3_to_p1_batch_v2.json" \
  --limit 30 \
  --output "artifacts/p5_from_p3_v2.jsonl" \
  --preview 5 \
  > artifacts/p5_export_v2.log 2>&1

.venv/bin/python scripts/analyze_p1_fragmentation.py \
  --input artifacts/p5_from_p3_v2.jsonl \
  --output artifacts/p1_eval/fragmentation_analysis_v2.json \
  --max-examples 10
```

## Integration Status
- Batch export succeeded.
- P1 eval hook succeeded.
- P5 benchmark export succeeded.
- Fragmentation analysis succeeded.
- Regression harness remained apples-to-apples with the previous 30-record run.

## Metric Comparison

### Cross-Source Pair Coverage
| Metric | Baseline | Chunking v2 | Delta |
| --- | ---: | ---: | ---: |
| Records with cross-source pairs | 26 / 30 | 24 / 30 | -2 |
| Coverage ratio | 0.8667 | 0.8000 | -0.0667 |
| Avg cross-source pairs per record | 1.7333 | 1.5667 | -0.1666 |

Assessment:
- Coverage did not collapse.
- It stayed on the practical floor set for this pass (`>= 0.80`), but there is a modest drop.

### Decisive Cross-Source NLI Coverage
| Metric | Baseline | Chunking v2 | Delta |
| --- | ---: | ---: | ---: |
| Records with decisive cross-source NLI | 1 / 30 | 1 / 30 | 0 |
| Coverage ratio | 0.0333 | 0.0333 | 0 |

Assessment:
- Chunking v2 did not improve decisive NLI coverage.
- The chunking hypothesis improved chunk cleanliness, but it was not enough on its own to unlock more decisive NLI outcomes in this 30-record batch.

### Fragmentation Indicators
| Metric | Baseline | Chunking v2 | Delta |
| --- | ---: | ---: | ---: |
| Claims evaluated | 395 | 426 | +31 |
| Avg claim token length | 19.7190 | 22.3568 | +2.6378 |
| Short-claim ratio (`<6` tokens) | 0.0506 | 0.0235 | -0.0271 |
| Broken-claim ratio | 0.0380 | 0.0117 | -0.0263 |
| Entity-only-like ratio | 0.0152 | 0.0070 | -0.0082 |
| Title-only-like ratio | 0.0506 | 0.0188 | -0.0318 |
| Records with broken claims | 12 / 30 | 5 / 30 | -7 |
| Records with broken-claim ratio | 0.4000 | 0.1667 | -0.2333 |

Assessment:
- Fragmentation improved materially.
- Claims got longer on average and visibly less broken.
- The strongest gain is the drop in records with broken claims: `12 -> 5`.

## Concrete Examples

### Clear Improvements
1. `sample_id=89891`
   Old: `Drake_Bell::chunk-1` started with `Spider-Man on Disney XD .`
   New: the same retrieval slot now begins with `Bell was the voice of Peter Parker / Spider-Man...`
   Why it matters: the chunk is now sentence-complete instead of title-tail only.

2. `sample_id=181634`
   Old: `Mogadishu::chunk-3` started with `, Design -LRB- TEDx -RRB- conference .`
   New: retrieved Mogadishu chunks begin with coherent lead sentences such as `Mogadishu ... is the capital...` and `Mogadishu Stadium was constructed...`
   Why it matters: the old orphan fragment disappeared from the top retrieved set.

3. `sample_id=204443`
   Old: one retrieved chunk started with the same `Spider-Man on Disney XD .` orphan fragment.
   New: the second retrieved chunk is now `Following that band's breakup in October 2000...`
   Why it matters: the retrieval output no longer opens on a sentence tail.

4. `sample_id=142454`
   Old: `Advertising::chunk-1` started with `presentation of the message in a medium...`
   New: `Advertising::chunk-1` starts with `Advertising is communicated through various mass media...`
   Why it matters: the chunk now preserves its governing clause and is more claim-complete.

5. `sample_id=137334`
   Old: `Soul_Food_(film)::chunk-1` started mid-thought with `more positive image of African-Americans...`
   New: it starts with `Tillman based the family in the film on his own...`
   Why it matters: the abbreviation-offset fix removed a mid-sentence truncation bug and restored local context.

### Still Failing or Partially Failing
6. `sample_id=137334`
   `Soul_Food_(film)::chunk-0` still yields cast-list fragments such as `Williams , Vivica A.` and `Hall , Gina Ravera and Brandon Hammond .`
   Why it still fails: the source sentence is a long enumeration, so P1 still extracts name-only claims from a valid chunk.

7. `sample_id=111897` and `sample_id=192714`
   `Food_Network::chunk-0` still yields `-LRB- which owns the remaining 30 % -RRB- .`
   Why it still fails: parenthetical ownership clauses remain extractor-hostile even when the chunk itself is coherent.

8. `sample_id=52175`
   `Magic_Johnson::chunk-0` still yields `Earvin `` Magic '' Johnson Jr.`
   Why it still fails: the lead sentence opens with a person-name apposition that P1 interprets as an entity-only claim.

9. `sample_id=204361`
   Query-side claim `The Cretaceous ended.` still appears as broken/entity-like.
   Why it still fails: this comes from the query itself, so chunking cannot solve it.

10. `sample_id=64721` and `sample_id=104386`
    Query-side short claims such as `Aristotle spent time in Athens.` and `Tenacious D started in 1997.` remain title-like under the current heuristic.
    Why it still fails: these are downstream extraction heuristics, not retrieval chunk boundary failures.

## Interpretation
- The sentence-aware v2 chunker successfully reduced fragmentation and removed several obvious sentence-tail retrieval artifacts.
- The decisive NLI metric did not move, which suggests fragmentation was only one part of the bottleneck.
- The remaining failures skew toward:
  - enumerative source sentences
  - parenthetical wiki syntax
  - query-side short claims that chunking cannot affect

## Final Recommendation
Keep chunking v2 as the new P3 baseline.

Reason:
- It materially improves chunk cleanliness and fragmentation metrics.
- Cross-source coverage stays acceptable at `0.8000`.
- The decisive NLI metric did not improve, but v2 removes enough obvious chunk-boundary damage that reverting would throw away a real quality gain.

Follow-up implication:
- The next bottleneck is no longer raw chunk boundary fragmentation alone.
- Any further improvement should be treated as a separate pass, not a reason to revert this chunking change.
