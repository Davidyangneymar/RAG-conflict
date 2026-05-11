# P3 Final Summary

## 1. 模块定位

P3 是 Conflict-Aware RAG 课程项目中的 retrieval layer。它负责把 claim/query 检索成 citation-ready evidence，并以稳定的 `retrieval_json` handoff 给 P1/P2/P5 使用。

P3 不负责 P1 claim extraction / NLI 主逻辑，不负责 P2 conflict typing，也不负责最终 answer generation。

## 2. 已完成内容

当前 P3 已完成：

- ingestion pipeline
- `chunking v2`
- BM25 retrieval
- dense retrieval
- hybrid retrieval
- retrieve-then-rerank
- optional BGE reranker backend
- source diversification
- citation-ready evidence schema
- P3 -> P1 handoff adapter
- P3 -> P1 batch harness
- fragmentation analysis
- query-side short-claim diagnostics
- P4/local live retrieval reproducibility docs and readiness check
- AVeriTeC dev smoke adaptation

## 3. 当前 Baseline

当前默认 baseline：

- `chunking_version: "v2"`
- `sentence_aware_chunking: true`
- `enable_evidence_hygiene: false`
- FEVER sample 作为 regression baseline / pipeline sanity check

`evidence hygiene` 代码保留，但默认关闭，不作为当前 baseline。

可选 BGE reranker：

- 配置文件：`config/retrieval_bge_reranker.yaml`
- 可选依赖：`pip install -e ".[bge]"`
- 模型：`BAAI/bge-reranker-v2-m3`
- 作用位置：BM25 + dense/hybrid candidate retrieval 之后，P3 -> P1 handoff 导出之前
- 保留 metadata：`pre_rerank_rank`、`rerank_score`、`reranker_backend`、`reranker_model`

BGE reranker 不改变默认 baseline，不替换 dense retrieval，不构建 Qdrant index，也不改变 `retrieval_json` handoff schema。如果 BGE inference 失败，P3 会在允许 fallback 时回退到 heuristic reranking，并在 metadata 中记录 fallback reason。

## 4. 数据集口径

### FEVER Sample

FEVER sample 用于 regression baseline 和 pipeline sanity check。它主要验证：

- corpus ingestion 是否可跑
- chunking / BM25 / dense / hybrid retrieval 是否可跑
- P3 -> P1 handoff schema 是否稳定
- P1 hook 是否可以消费 P3 output

### AVeriTeC Dev Smoke

AVeriTeC dev smoke 用于验证主数据集格式、metadata preservation 和 handoff compatibility。

本轮 smoke 使用 `data/raw/averitec/fever7/dev.json` 中的 QA answers / justifications 构造 small smoke corpus。它不是完整 AVeriTeC web evidence retrieval benchmark。

### Full AVeriTeC Evidence Collection

当前阶段没有使用 full AVeriTeC evidence collection，也没有下载或处理约 116GB 的 knowledge store。

如需真实 AVeriTeC retrieval，需要额外准备 `dev_knowledge_store.zip` 或更大的 evidence collection，并重新做 ingestion 规模控制。

## 5. 当前结论

- `chunking v2` 已缓解明显的 chunk boundary damage。
- `evidence hygiene` 能降低部分 noisy evidence，但没有显著提升 decisive NLI，因此不提升为 baseline。
- 当前 residual bottleneck 更接近 query-side short claims / P1 query claim transformation。
- P3 当前建议冻结 retrieval baseline，避免继续在 retrieval 侧做低收益调参。
- AVeriTeC dev smoke 已证明 P3 可以适配主数据集风格的 claim / label / metadata / handoff。
- BGE reranker 已通过 P3 -> P1 hook 和 P5 export smoke 验证，新增 rerank metadata 不破坏下游 schema。

## 6. 对下游的交付物

P3 对下游正式交付：

- `retrieval_json`
- citation-ready evidence chunks
- selected reports
- local retrieval reproducibility docs
- readiness check script

核心对接文件：

- `scripts/export_p1_handoff.py`
- `src/services/handoff_adapter.py`
- `config/retrieval_bge_reranker.yaml`
- `data/examples/p3_to_p1_batch_query30.json`
- `P3_LOCAL_RETRIEVAL_SETUP.md`
- `artifacts/selected_reports/`
- `artifacts/averitec_smoke/p3_averitec_dev_smoke_report.md`

## 7. 当前限制

- FEVER sample 不是最终主实验。
- AVeriTeC dev smoke 不是完整 web evidence retrieval benchmark。
- 当前 decisive NLI 低的问题主要转向 P1/query-side claim transformation。
- P3 不负责 final conflict typing / answer policy / answer generation。
- GitHub 不应包含 raw dataset、full evidence collection、Qdrant index、`.venv` 或大 log。

## 8. 后续扩展建议

如果课程后续继续扩展 P3，可以考虑：

- 小规模接入 AVeriTeC dev knowledge store
- 扩大 corpus coverage
- 针对真实 web evidence 做 source normalization
- 继续维护 P3 -> P1/P4/P5 contract tests

当前阶段不建议处理 116GB full evidence collection，也不建议继续大幅改 retrieval baseline。
