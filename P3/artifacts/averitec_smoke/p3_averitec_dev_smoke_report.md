# P3 AVeriTeC Dev Smoke Report

## 1. Scope

本轮使用的数据文件是 `data/raw/averitec/fever7/dev.json`。

这次没有使用完整的 AVeriTeC evidence collection，也没有下载或处理约 116GB 的 knowledge store。本轮目标只是验证 P3 对 AVeriTeC dev claim / label / metadata / QA-answer 格式的最小适配，以及 P3 -> P1 `retrieval_json` handoff 是否能跑通。

FEVER 小样本仍然是当前 P3 regression baseline。AVeriTeC dev smoke 不替代 FEVER baseline，也不用于宣称真实 web evidence retrieval 效果。

## 2. dev.json 字段结构摘要

`dev.json` 当前是一个包含 500 条 record 的 JSON list。前 1-3 条 record 显示字段结构稳定，主要字段包括：

- `claim`
- `required_reannotation`
- `label`
- `justification`
- `claim_date`
- `speaker`
- `original_claim_url`
- `fact_checking_article`
- `reporting_source`
- `location_ISO_code`
- `claim_types`
- `fact_checking_strategies`
- `questions`
- `cached_original_claim_url`

`questions` 是 list，每个问题包含：

- `question`
- `answers`

`answers` 是 list，每个 answer 通常包含：

- `answer`
- `answer_type`
- `source_url`
- `source_medium`
- `cached_source_url`

标签分布：

- `Refuted`: 305
- `Supported`: 122
- `Conflicting Evidence/Cherrypicking`: 38
- `Not Enough Evidence`: 35

## 3. Evidence-like Text 判断

AVeriTeC dev 文件中包含 QA answers 和 justification，可以构造小型 smoke corpus。

本地检查结果：

- dev records: 500
- evidence-like non-empty answers: 1360
- answers with source URL or cached source URL: 1360
- explicit no-answer entries: 39

因此，本轮可以做 dev-only smoke retrieval，但它仍然不是完整 web evidence retrieval。因为这些 answer 已经是 AVeriTeC 标注/整理后的 QA evidence-like 文本，不等价于从完整 web knowledge store 中检索原始网页证据。

## 4. 生成的 Small Subset

生成命令：

```bash
python scripts/prepare_averitec_smoke.py \
  --input-path data/raw/averitec/fever7/dev.json \
  --output-dir data/processed/averitec \
  --claim-limit 50 \
  --max-documents 100
```

生成产物：

- `data/processed/averitec/averitec_dev_smoke_corpus.jsonl`
- `data/processed/averitec/averitec_dev_smoke_claims.jsonl`
- `data/processed/averitec/averitec_dev_smoke_summary.json`

本轮 smoke slice：

- claims: 50
- smoke corpus documents: 100

## 5. P3 Ingestion / Retrieval Smoke

为避免覆盖 FEVER baseline，本轮使用独立配置：

- `config/retrieval_averitec_smoke.yaml`
- Qdrant collection: `conflict_aware_rag_averitec_smoke`
- Qdrant path: `data/processed/averitec/qdrant`
- chunk store: `data/processed/averitec/chunks.jsonl`
- BM25 store: `data/processed/averitec/bm25_corpus.json`

Ingestion 命令：

```bash
python scripts/ingest_corpus.py \
  --input-path data/processed/averitec/averitec_dev_smoke_corpus.jsonl \
  --loader generic \
  --dataset averitec_dev_smoke \
  --config-path config/retrieval_averitec_smoke.yaml
```

结果：

- smoke documents ingested: 100
- chunks created: 101
- vectors indexed: 101

## 6. P3 -> P1 Handoff Smoke

导出命令：

```bash
python scripts/export_p1_handoff.py \
  --claims-path data/processed/averitec/averitec_dev_smoke_claims.jsonl \
  --claims-loader averitec_dev_smoke \
  --top-k 5 \
  --mode hybrid \
  --split averitec_dev_smoke \
  --limit 30 \
  --config-path config/retrieval_averitec_smoke.yaml \
  --output-path artifacts/averitec_smoke/p3_to_p1_averitec_dev_smoke.json
```

结果：

- 成功导出 P3 -> P1 `retrieval_json`
- exported records: 30
- record shape: `sample_id`, `query`, `label`, `metadata`, `retrieved_chunks`
- metadata 中保留 `dataset=averitec_dev_smoke`、`split=averitec_dev_smoke`、`claim_date`

## 7. P1 Hook Smoke

本地存在 P1 副本时，运行：

```bash
PYTHONPATH=artifacts/p1_local/src python artifacts/p1_local/scripts/eval_p3_retrieval_hook.py \
  --input artifacts/averitec_smoke/p3_to_p1_averitec_dev_smoke.json \
  --input-kind retrieval_json \
  --limit 30 \
  --extractor-kind structured \
  --entity-backend auto \
  --preview 5 \
  > artifacts/averitec_smoke/p1_hook_averitec_dev_smoke.log 2>&1
```

结果：

- evaluated records: 30
- records with retrieved chunks: 30
- cross-source pair coverage: 1.0
- decisive cross-source NLI coverage: 1.0

注意：这里的 P1 hook 指标只说明 handoff 与 P1 结构化流程能跑通，不应与 FEVER regression harness 指标直接比较。AVeriTeC smoke corpus 使用的是 dev QA answers / justifications，而不是完整 evidence collection 中的原始网页检索结果。

## 8. 当前结论

AVeriTeC dev 文件可以提供足够的 evidence-like text，用于 P3 dev-only smoke corpus、metadata preservation、chunking v2、hybrid retrieval、P3 -> P1 handoff 兼容性验证。

本轮已经成功完成：

- AVeriTeC dev 格式检查
- AVeriTeC QA-answer / justification smoke corpus 构造
- 独立 AVeriTeC smoke index ingestion
- P3 -> P1 `retrieval_json` 导出
- P1 hook smoke

## 9. 后续真实 Retrieval 所需数据

如果后续要评估真实 AVeriTeC web evidence retrieval，需要额外下载并处理 `dev_knowledge_store.zip` 或更大的 evidence collection。当前 dev-only smoke 不能替代真实 evidence collection retrieval。
