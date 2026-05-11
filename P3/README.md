# Conflict-Aware RAG：P3 Retrieval Layer

P3 是 Conflict-Aware RAG 课程项目中的 **检索层（retrieval layer）**。本模块不负责最终回答生成，也不负责 P1/P2/P5 的任务边界；它的核心职责是把输入 claim/query 转换成 citation-ready evidence，并以稳定的 `retrieval_json` 格式交给下游模块。

当前主线：`chunking v2` 是默认 baseline；`evidence hygiene` 代码保留但默认关闭；P3 -> P1 batch integration 已打通。

当前还提供 reranker ablation 配置：`config/retrieval_no_rerank.yaml` 直接导出 hybrid retrieval 顺序，`config/retrieval_bge_reranker.yaml` 在 hybrid 候选后应用 BGE reranker。两者都不会替换默认 `config/retrieval.yaml`，也不会改变 P3 -> P1 handoff schema。

## 1. 模块定位

P3 当前负责：

- 文档 ingestion 与索引构建
- sentence-aware `chunking v2`
- BM25 / dense / hybrid retrieval
- retrieve-then-rerank
- optional BGE reranking
- citation-ready evidence 输出
- `P3 -> P1` handoff 导出
- 固定 batch harness 回归验证
- fragmentation / query short-claim diagnostics
- 本地 retrieval 复现检查


## 2. Current Status

- FEVER sample 继续作为 regression baseline / pipeline sanity check。
- AVeriTeC dev smoke 用于主数据集格式、metadata preservation、handoff compatibility check。
- Full AVeriTeC evidence collection 当前阶段未使用，也没有加入仓库。
- 当前 P3 不代表 full AVeriTeC web evidence retrieval benchmark。
- AVeriTeC dev smoke 的 evidence-like text 来自 dev data 中的 QA answers / justifications，不应和 FEVER regression 指标直接比较。
- 当前 residual bottleneck 更接近 query-side short claims / P1 query claim transformation，而不是继续做 P3 retrieval tuning。
- BGE reranker 已作为可选 rerank backend 接入；no-rerank ablation config 已提供；默认 baseline 仍保持不变。

## 3. Dataset Status

| 数据口径 | 当前用途 | 是否作为 baseline | 说明 |
| --- | --- | --- | --- |
| FEVER sample | regression baseline / pipeline sanity check | 是 | 用于验证 ingestion、retrieval、P3 -> P1 handoff 是否稳定 |
| AVeriTeC dev smoke | format / metadata / handoff compatibility check | 否 | 使用 dev JSON 中 QA answers / justifications 构造 small smoke corpus |
| Full AVeriTeC evidence collection | 未使用 | 否 | 体积过大，当前阶段不下载、不处理、不提交 |

如果后续要做真实 AVeriTeC web evidence retrieval，需要额外准备 `dev_knowledge_store.zip` 或更大的 evidence collection，并重新设计数据 ingestion 规模控制。本仓库当前只提供 smoke adaptation。

## 4. 仓库结构

```text
P3/
├── README.md
├── P3_LOCAL_RETRIEVAL_SETUP.md
├── P3_FINAL_SUMMARY.md
├── pyproject.toml
├── config/
│   ├── retrieval.yaml
│   ├── retrieval_averitec_smoke.yaml
│   ├── retrieval_no_rerank.yaml
│   └── retrieval_bge_reranker.yaml
├── src/
│   ├── ingestion/
│   ├── retrieval/
│   ├── schemas/
│   ├── services/
│   ├── app/
│   └── utils/
├── scripts/
├── tests/
├── data/
│   └── examples/
└── artifacts/
    ├── selected_reports/
    └── averitec_smoke/
```

- `src/`：核心 Python 模块，包括 ingestion、retrieval、schemas、services 和 FastAPI 入口。
- `scripts/`：命令行入口，包括 ingest、retrieval、handoff export、diagnostics、readiness check。
- `tests/`：轻量回归测试和 contract smoke tests。
- `config/`：FEVER baseline 配置与 AVeriTeC smoke 独立配置。
- `data/examples/`：小型 handoff JSON 示例，可用于 P1/P2 离线联调。
- `artifacts/selected_reports/`：精选 markdown/json 报告，用于组内交接和课程汇报。
- `artifacts/averitec_smoke/`：AVeriTeC dev smoke markdown 报告；大 JSON/log 不应提交。

默认不要把 raw dataset、完整 processed 数据、Qdrant index、`.venv`、log 或 cache 上传 GitHub。

## 5. 环境安装

建议使用 Python `3.11+`。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

如果需要运行 BGE reranker，请额外安装可选依赖：

```bash
pip install -e ".[bge]"
```

BGE 模型首次运行会下载 `BAAI/bge-reranker-v2-m3`，本地环境需要能访问 Hugging Face 或已提前缓存模型。

如果只做 P1/P2 离线联调，可以直接使用 `data/examples/` 中的 handoff JSON，不一定需要本地 Qdrant。

如果要做 live retrieval，需要先准备 corpus 并运行 ingestion。详见 [P3_LOCAL_RETRIEVAL_SETUP.md](P3_LOCAL_RETRIEVAL_SETUP.md)。

## 6. 快速开始

### 6.1 FEVER baseline ingestion

需要手动放置 FEVER sample corpus，例如：

```text
FEVER Dataset/wiki_pages_matched_sample.jsonl
```

运行：

```bash
python scripts/ingest_corpus.py \
  --input-path "FEVER Dataset/wiki_pages_matched_sample.jsonl" \
  --loader fever_wiki \
  --dataset fever_wiki_sample \
  --config-path config/retrieval.yaml
```

检查本地 retrieval 是否 ready：

```bash
python scripts/check_retrieval_ready.py \
  --config-path config/retrieval.yaml \
  --corpus-path "FEVER Dataset/wiki_pages_matched_sample.jsonl"
```

### 6.2 导出 P3 -> P1 handoff

```bash
python scripts/export_p1_handoff.py \
  --claims-path "FEVER Dataset/shared_task_dev_sample.jsonl" \
  --top-k 5 \
  --mode hybrid \
  --split dev \
  --limit 30 \
  --config-path config/retrieval.yaml \
  --output-path data/processed/p3_to_p1_batch_query30.json
```

### 6.3 Reranker 开关速查

P3 的 reranker 主要通过 config 控制。后续测试时不要改代码，直接切换 `--config-path`。

| 测试目标 | 配置文件 | 行为 | 依赖 |
| --- | --- | --- | --- |
| 当前轻量 baseline | `config/retrieval.yaml` | hybrid retrieval 后使用 heuristic rerank | `pip install -e .[dev]` |
| 不使用 rerank | `config/retrieval_no_rerank.yaml` | 直接导出 hybrid retrieval 顺序 | `pip install -e .[dev]` |
| 使用 BGE rerank | `config/retrieval_bge_reranker.yaml` | hybrid candidates -> BGE rerank -> export | `pip install -e ".[bge]"` |

单次 CLI 调试也可以用 `--no-rerank` 临时跳过 rerank：

```bash
python scripts/run_retrieval.py \
  --config-path config/retrieval_bge_reranker.yaml \
  --query "Does coffee cause cancer?" \
  --top-k 5 \
  --mode hybrid \
  --no-rerank
```

注意：正式做 no-rerank vs BGE ablation 时，推荐使用 `config/retrieval_no_rerank.yaml`，因为它会在 metadata 中明确记录 `reranker_backend: "none"`，更容易对比和复现。

### 6.4 可选 BGE reranker

默认 `config/retrieval.yaml` 不启用 BGE。若要验证 BGE reranker，请使用独立配置：

```bash
python scripts/run_retrieval.py \
  --config-path config/retrieval_bge_reranker.yaml \
  --query "Does coffee cause cancer?" \
  --top-k 5 \
  --mode hybrid
```

导出 BGE reranked handoff：

```bash
python scripts/export_p1_handoff.py \
  --config-path config/retrieval_bge_reranker.yaml \
  --claims-path "FEVER Dataset/shared_task_dev_sample.jsonl" \
  --top-k 5 \
  --mode hybrid \
  --split dev \
  --limit 30 \
  --output-path artifacts/p1_integration_test/p3_to_p1_bge_query30.json
```

After hybrid candidate retrieval, P3 optionally applies `BAAI/bge-reranker-v2-m3` as a post-retrieval reranker. BM25 and dense retrieval first collect candidate evidence chunks, while the BGE reranker takes each query-passage pair as input and outputs a relevance score. The candidates are then reordered by `rerank_score` before being exported as `retrieval_json`.

Reranking metadata is preserved for debugging and traceability:

- `pre_rerank_rank`
- `rerank_score`
- `reranker_backend`
- `reranker_model`

If BGE loading or inference fails and `allow_model_fallback` is enabled, P3 falls back to heuristic reranking and records the fallback reason in metadata. BGE does not train a model, does not replace dense retrieval, and does not build the Qdrant index.

### 6.5 Reranker Ablation: no-rerank vs BGE rerank

We provide two P3 configurations for reranker ablation. The no-rerank baseline directly exports the hybrid retrieval order, while the BGE reranker version applies `BAAI/bge-reranker-v2-m3` to rescore query-passage pairs before exporting `retrieval_json`. Both variants keep the downstream handoff schema unchanged, so P1 and P5 can compare reranking effects without interface changes.

No-rerank run:

```bash
python scripts/run_retrieval.py \
  --config-path config/retrieval_no_rerank.yaml \
  --query "Does coffee cause cancer?" \
  --top-k 5 \
  --mode hybrid
```

BGE-rerank run:

```bash
python scripts/run_retrieval.py \
  --config-path config/retrieval_bge_reranker.yaml \
  --query "Does coffee cause cancer?" \
  --top-k 5 \
  --mode hybrid
```

The no-rerank config uses `reranker_backend: "none"` and preserves `score_hybrid` as the handoff score. The BGE config writes `score_rerank` plus metadata such as `pre_rerank_rank`, `rerank_score`, `reranker_backend`, and `reranker_model`.

Do not interpret this ablation as final benchmark improvement unless the full evaluation is rerun.

### 6.6 AVeriTeC dev smoke

AVeriTeC raw dev 文件不应提交到 GitHub。若本地已有：

```text
data/raw/averitec/fever7/dev.json
```

可运行：

```bash
python scripts/prepare_averitec_smoke.py \
  --input-path data/raw/averitec/fever7/dev.json \
  --output-dir data/processed/averitec \
  --claim-limit 50 \
  --max-documents 100
```

再用独立 smoke config ingestion：

```bash
python scripts/ingest_corpus.py \
  --input-path data/processed/averitec/averitec_dev_smoke_corpus.jsonl \
  --loader generic \
  --dataset averitec_dev_smoke \
  --config-path config/retrieval_averitec_smoke.yaml
```

## 7. Integration Notes

### P3 -> P1

P3 对外正式输出是 `retrieval_json`，不是 Qdrant 本体。

handoff schema 固定为：

- `sample_id`
- `query`
- `label`
- `metadata`
- `retrieved_chunks`

`retrieved_chunks[]` 应包含：

- `chunk_id`
- `text`
- `rank`
- `retrieval_score`
- `source_url`
- `source_medium`
- `metadata`

P1 可以直接使用 P3 example handoff JSON 做离线联调，例如：

- `data/examples/p3_to_p1_batch_query30.json`
- `data/examples/p3_to_p1_single.json`

如果本地生成了 AVeriTeC smoke handoff，也可以使用：

- `artifacts/averitec_smoke/p3_to_p1_averitec_dev_smoke.json`

但该 JSON 属于本地生成产物，默认不提交 GitHub。

### P3 -> P4

如果 P4 只是做离线 pipeline 联调，可以直接读取 P3 handoff JSON，不需要 Qdrant。

如果 P4 要做 live backend demo，则必须本地跑 P3 retrieval，至少需要：

- corpus
- ingestion
- Qdrant collection
- chunks / BM25 corpus
- `config/retrieval.yaml` 或 `config/retrieval_averitec_smoke.yaml`

注意：

- Qdrant 服务或本地目录存在，不等于 collection 里已经有数据。
- 必须先运行 ingestion，生成 chunks、BM25 corpus 和 Qdrant points。
- 如果修改 embedding backend / model / dim，必须重新 ingest，并建议使用新的 collection name。
- 不要混用旧 collection 和新 embedding 配置。

P4 live backend 最小接入流程：

1. 在 P3 目录安装依赖。

```bash
pip install -e .[dev]
```

如果 P4 要测试 BGE rerank：

```bash
pip install -e ".[bge]"
```

2. 放置 corpus，并先建库。

```bash
python scripts/ingest_corpus.py \
  --input-path "FEVER Dataset/wiki_pages_matched_sample.jsonl" \
  --loader fever_wiki \
  --dataset fever_wiki_sample \
  --config-path config/retrieval.yaml
```

3. 启动前检查本地 retrieval 是否 ready。

```bash
python scripts/check_retrieval_ready.py \
  --config-path config/retrieval.yaml \
  --corpus-path "FEVER Dataset/wiki_pages_matched_sample.jsonl" \
  --sample-query "Fox 2000 Pictures released the film Soul Food."
```

4. 选择 P4 要调用的 config，并启动 FastAPI。

默认轻量 baseline：

```bash
RETRIEVAL_CONFIG_PATH=config/retrieval.yaml \
uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

no-rerank ablation：

```bash
RETRIEVAL_CONFIG_PATH=config/retrieval_no_rerank.yaml \
uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

BGE rerank：

```bash
RETRIEVAL_CONFIG_PATH=config/retrieval_bge_reranker.yaml \
uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

5. P4 调用接口。

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

检索请求：

```bash
curl -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Fox 2000 Pictures released the film Soul Food.",
    "top_k": 5,
    "mode": "hybrid",
    "use_rerank": true,
    "use_diversify": true
  }'
```

返回值是 `RetrievalResponse`，P4 主要读取：

- `results[].chunk_id`
- `results[].doc_id`
- `results[].text`
- `results[].rank`
- `results[].score_hybrid`
- `results[].score_rerank`
- `results[].source_url`
- `results[].source_name`
- `results[].metadata`

如果 P4 只是想临时关闭 rerank，也可以在请求里设：

```json
{
  "use_rerank": false
}
```

但严格 ablation 建议仍使用 `RETRIEVAL_CONFIG_PATH=config/retrieval_no_rerank.yaml` 启动服务。

### P5

P5 可以基于 P3/P1/P2 的导出结果做评测。P3 可提供：

- `retrieval_json`
- smoke report
- query diagnosis report
- selected summary JSON

不建议 P5 直接依赖 P3 本地 Qdrant 状态，因为 Qdrant 是本地运行产物，不适合作为跨模块 contract。

## 8. 关键脚本

- `scripts/ingest_corpus.py`：构建 chunk、BM25 corpus 和 Qdrant index。
- `scripts/run_retrieval.py`：单条 query 检索调试。
- `scripts/export_p1_handoff.py`：导出 P3 -> P1 `retrieval_json`。
- `config/retrieval_no_rerank.yaml`：reranker ablation 的 no-rerank 配置，直接保留 hybrid retrieval 顺序。
- `config/retrieval_bge_reranker.yaml`：可选 BGE reranker 配置，不改变默认 baseline。
- `scripts/check_retrieval_ready.py`：检查本地 corpus / artifacts / Qdrant collection 是否 ready。
- `scripts/prepare_averitec_smoke.py`：从 AVeriTeC dev JSON 构造 small smoke corpus/claims。
- `scripts/analyze_p1_fragmentation.py`：分析 P1 输出中的 broken / short claims。
- `scripts/analyze_p1_query_diagnostics.py`：诊断 query-side short-claim 和 evidence quality。
- `scripts/build_query_short_claim_report.py`：生成 query taxonomy / failure attribution 报告。

## 9. 当前实验结论

- `chunking v2` 相比旧版缓解了 chunk boundary damage。
- `evidence hygiene` 降低了一部分 evidence noise，但未显著提升 decisive NLI，因此默认关闭。
- query-side family failure 已高于 retrieval-side family。
- 当前 P3 baseline 建议冻结，后续主要问题更适合交给 P1 / query-understanding 侧继续处理。
- AVeriTeC dev smoke 证明 P3 能适配主数据集风格的 claim / label / metadata / handoff，但不能代表完整 AVeriTeC web evidence retrieval benchmark。

## 10. 关键产物

推荐优先阅读：

- `P3_FINAL_SUMMARY.md`
- `P3_LOCAL_RETRIEVAL_SETUP.md`
- `artifacts/selected_reports/p3_chunking_v2_comparison.md`
- `artifacts/selected_reports/p3_evidence_hygiene_comparison.md`
- `artifacts/selected_reports/p3_query_short_claim_report.md`
- `artifacts/averitec_smoke/p3_averitec_dev_smoke_report.md`

## 11. 测试

常用轻量测试：

```bash
pytest tests/test_splitter.py -q
pytest tests/test_handoff_adapter.py -q
pytest tests/test_retrieval_pipeline.py -q
pytest tests/test_check_retrieval_ready.py -q
```

部分 end-to-end retrieval / export 命令依赖本地 corpus 和已构建 index。如果缺少数据，请先阅读 `P3_LOCAL_RETRIEVAL_SETUP.md`。

## 12. 已知限制

- 当前数据覆盖有限，FEVER sample 主要用于 regression，不是最终主实验。
- AVeriTeC dev smoke 不是完整 web evidence retrieval benchmark。
- 当前没有处理 full AVeriTeC evidence collection。
- P3 不负责 final conflict typing / answer policy / answer generation。
- decisive NLI 低的问题不应被包装成“全链路已解决”；当前判断更偏 P1/query-side claim transformation。

## 13. GitHub / 分享建议

建议提交：

- code、config、tests、docs
- 小型 example JSON
- 小型 markdown reports / summary JSON

不建议提交：

- raw datasets
- full wiki / AVeriTeC evidence collection
- `data/processed/` 大文件
- Qdrant index
- `.venv`
- logs / cache
- zip bundles

如果组员需要本地 live retrieval 复现，优先阅读 `P3_LOCAL_RETRIEVAL_SETUP.md`，或通过组内私发轻量复现包。GitHub 主仓库只保留可解释、可对接、可复现的最小材料。
