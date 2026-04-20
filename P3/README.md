# Conflict-Aware RAG：P3 检索层

## 1. 项目简介

P3 是 Conflict-Aware RAG 课程项目中的 **检索层（retrieval layer）**。  
本模块负责：

- 文档 ingestion 与索引构建
- chunking
- BM25 / dense / hybrid retrieval
- reranking
- citation-ready evidence 输出
- `P3 -> P1` handoff 导出
- 用固定 batch harness 做回归验证与诊断

当前版本的目标是：为 P1 提供更稳定、可引用、可回归的检索证据输入。

---

## 2. 当前状态

- 当前默认 baseline：`chunking v2`
- `evidence hygiene` 代码保留，但默认关闭，不作为当前 baseline
- 已经打通 `P3 -> P1` batch integration
- 当前固定 30 条回归 harness 上：
  - cross-source pair coverage = `0.8000`
  - decisive cross-source NLI coverage = `0.0333`
- 当前主瓶颈更接近 **query-side short claims / P1 query claim transformation**，而不是继续做纯 retrieval tuning

---

## 3. 仓库结构

```text
P3_share/
├── README.md
├── P3_share_manifest.md
├── pyproject.toml
├── config/
├── src/
├── scripts/
├── tests/
├── data/
│   └── examples/
└── artifacts/
    └── selected_reports/
```

- `src/`：P3 核心代码，包含 ingestion、retrieval、schemas、services、FastAPI 入口等
- `scripts/`：命令行入口，覆盖 ingest、handoff export、fragmentation 分析、query 诊断
- `tests/`：轻量回归测试
- `config/`：当前 baseline 配置
- `data/examples/`：精简保留的 handoff JSON 样例
- `artifacts/selected_reports/`：关键对比报告和结构化 summary，用于组内交接和汇报

---

## 4. 环境安装

建议环境：

- Python `3.11+`

创建虚拟环境并安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

运行前需要准备：

- `config/retrieval.yaml`
- 本地 FEVER / matched wiki 数据
- 本地 P1 仓库路径（如需跑联调）

说明：

- 本公开版不包含原始大数据集、完整 wiki 语料或本地 P1 副本
- `data/examples/` 仅用于展示 handoff 结构和结果样例

---

## 5. 快速开始

### 5.1 构建索引

```bash
.venv/bin/python scripts/ingest_corpus.py \
  --input-path "path/to/wiki_pages_matched_sample.jsonl" \
  --loader fever_wiki \
  --dataset fever_wiki_sample \
  --config-path config/retrieval.yaml
```

### 5.2 导出 `P3 -> P1` handoff

```bash
.venv/bin/python scripts/export_p1_handoff.py \
  --claims-path "path/to/shared_task_dev_sample.jsonl" \
  --top-k 5 \
  --mode hybrid \
  --split dev \
  --limit 30 \
  --config-path config/retrieval.yaml \
  --output-path data/processed/p3_to_p1_batch_query30.json
```

### 5.3 运行 P1 hook / benchmark

本分享包不包含 P1 仓库；如需联调，请将下列命令中的 `path/to/local/P1` 替换为你本地的 P1 路径。

```bash
.venv/bin/python path/to/local/P1/scripts/eval_p3_retrieval_hook.py \
  --input "data/processed/p3_to_p1_batch_query30.json" \
  --input-kind retrieval_json \
  --limit 30 \
  --extractor-kind structured \
  --entity-backend auto
```

```bash
.venv/bin/python path/to/local/P1/scripts/export_p5_benchmark.py \
  --dataset retrieval_json \
  --input "data/processed/p3_to_p1_batch_query30.json" \
  --limit 30 \
  --output "artifacts/p5_from_p3_query30.jsonl"
```

### 5.4 运行诊断脚本

```bash
.venv/bin/python scripts/analyze_p1_fragmentation.py \
  --input artifacts/p5_from_p3_query30.jsonl \
  --output artifacts/p1_eval/fragmentation_analysis_query30.json
```

```bash
.venv/bin/python scripts/build_query_short_claim_report.py \
  --regression-benchmark artifacts/p5_from_p3_query30.jsonl \
  --regression-handoff data/processed/p3_to_p1_batch_query30.json \
  --diagnostic-benchmark artifacts/p5_from_p3_diag150.jsonl \
  --diagnostic-handoff data/processed/p3_to_p1_batch_diag150.json \
  --taxonomy-output artifacts/p1_eval/query_taxonomy_summary.json \
  --failure-output artifacts/p1_eval/query_failure_attribution.json \
  --report-output artifacts/p3_query_short_claim_report.md \
  --config-path config/retrieval.yaml
```

---

## 6. 关键脚本

- `scripts/ingest_corpus.py`：构建 chunk、BM25 语料和 Qdrant 本地索引
- `scripts/export_p1_handoff.py`：导出稳定的 `P3 -> P1` handoff JSON
- `scripts/run_retrieval.py`：单条 query 检索调试
- `scripts/eval_retrieval.py`：轻量 retrieval 评估
- `scripts/analyze_p1_fragmentation.py`：分析 P1 输出中的 broken / short claims
- `scripts/analyze_p1_query_diagnostics.py`：快速查看 short query claim 与 evidence 关系
- `scripts/build_query_short_claim_report.py`：生成 query taxonomy、failure attribution 和 markdown 总结报告

---

## 7. 当前实验结论

- `chunking v2` 相比旧版缓解了 chunk boundary damage，已经冻结为当前 baseline
- `evidence hygiene` 降低了一部分 evidence noise，但没有显著提升 decisive NLI，因此默认关闭
- 在当前固定 30 条回归 harness 上，query-side family failure 已高于 retrieval-side family
- 因此，P3 当前更合理的策略是冻结 baseline，并把后续主问题交给 **P1 / query-understanding** 侧继续处理

建议优先阅读：

- `artifacts/selected_reports/p3_chunking_v2_comparison.md`
- `artifacts/selected_reports/p3_evidence_hygiene_comparison.md`
- `artifacts/selected_reports/p3_query_short_claim_report.md`

---

## 8. 对接说明

### 给 P1 / 组内同学

- P3 当前已经稳定导出 `sample_id / query / metadata / retrieved_chunks` 结构
- handoff 适配逻辑在 `src/services/handoff_adapter.py`
- `data/examples/` 中保留了单条与 batch handoff 样例，适合快速看结构
- 当前最值得继续跟进的不是 retrieval 算法扩展，而是 **query-side short claims / query claim transformation**

### 给需要复现实验的同学

- 本分享包只保留了最小代码、配置、测试和关键报告
- 如果要重跑完整流程，需要自行准备：
  - FEVER 样例数据
  - matched wiki 样例
  - 本地 P1 仓库

---

## 9. 测试

运行全部测试：

```bash
.venv/bin/python -m pytest tests -q
```

常用测试：

```bash
.venv/bin/python -m pytest tests/test_splitter.py -q
.venv/bin/python -m pytest tests/test_retrieval_pipeline.py -q
.venv/bin/python -m pytest tests/test_handoff_adapter.py -q
.venv/bin/python -m pytest tests/test_evidence_hygiene.py -q
.venv/bin/python -m pytest tests/test_query_short_claim_report.py -q
```

---

## 10. 已知限制

- 当前公开版不包含原始大数据集、完整 wiki 语料和本地索引
- 30-record harness 主要用于回归，不等于大规模 benchmark
- P3 只负责 retrieval layer，不负责最终 conflict typing / answer policy / answer generation
- 当前 decisive NLI 仍然较低，不能夸大为“全链路已解决”
- 一些选入的报告是历史实验产物，命令与路径说明已经做了公开版清理，但并不意味着分享包内包含了所有原始依赖
