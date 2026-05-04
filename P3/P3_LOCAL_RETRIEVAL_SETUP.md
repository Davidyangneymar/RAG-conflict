# P3 本地检索复现说明

## 1. 这份文档解决什么问题

这份文档用于帮助组员在自己的机器上重建 P3 retrieval 环境，避免出现“代码和 config 都有，但检索结果为空”的情况。

需要特别注意：

- Qdrant 能初始化或目录存在，不等于 collection 里已经有语料。
- 只有 `config/retrieval.yaml` 不能直接检索。
- 必须先准备 corpus，再运行 ingestion，生成 chunks、BM25 corpus 和 Qdrant collection。
- 如果本地没有跑过 ingestion，retrieval pipeline 可能能启动，但检索会为空或报 collection/artifact 缺失。

当前 P3 baseline 不变：

- `chunking v2` 是默认 baseline。
- `evidence hygiene` 代码保留，但默认关闭。
- 本文档只说明本地复现，不改变检索逻辑。

## 2. 必需文件

本地复现至少需要：

- P3 代码目录
- `config/retrieval.yaml`
- `FEVER Dataset/wiki_pages_matched_sample.jsonl`
- Python 3.11+ 环境
- 项目依赖

建议安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## 3. 推荐目录结构

建议把文件放成下面的结构：

```text
P3/
├── config/
│   └── retrieval.yaml
├── FEVER Dataset/
│   └── wiki_pages_matched_sample.jsonl
├── scripts/
│   ├── ingest_corpus.py
│   ├── check_retrieval_ready.py
│   └── run_retrieval.py
├── src/
├── data/
│   └── processed/
└── README.md
```

`data/processed/` 可以一开始不存在，运行 ingestion 后会自动生成需要的中间产物。

## 4. 从 0 建库步骤

在 P3 项目根目录运行：

```bash
python scripts/ingest_corpus.py \
  --input-path "FEVER Dataset/wiki_pages_matched_sample.jsonl" \
  --loader fever_wiki \
  --dataset fever_wiki_sample \
  --config-path config/retrieval.yaml
```

成功后通常会生成：

- `data/processed/chunks.jsonl`
- `data/processed/bm25_corpus.json`
- `data/processed/qdrant/`

这些文件和目录是 retrieval 能返回结果的关键。

## 5. 建库后如何检查

建议每次换机器、换 config、换语料后都先运行 readiness 检查：

```bash
python scripts/check_retrieval_ready.py \
  --config-path config/retrieval.yaml \
  --corpus-path "FEVER Dataset/wiki_pages_matched_sample.jsonl"
```

如果想同时尝试一条检索 query：

```bash
python scripts/check_retrieval_ready.py \
  --config-path config/retrieval.yaml \
  --corpus-path "FEVER Dataset/wiki_pages_matched_sample.jsonl" \
  --sample-query "Fox 2000 Pictures released the film Soul Food."
```

检查重点：

- config 文件是否存在
- corpus 文件是否存在且非空
- `chunks.jsonl` 是否生成
- `bm25_corpus.json` 是否生成
- Qdrant path 是否存在
- Qdrant collection 是否存在
- collection point 数量是否大于 0
- collection vector size 是否和当前 embedding 配置兼容
- sample query 是否能返回非空 evidence

如果最终状态是 `READY`，说明本地检索环境基本可用。

## 6. Embedding 配置注意事项

`config/retrieval.yaml` 中的 embedding 配置必须和已建索引匹配。

当前轻量 baseline 使用：

```yaml
embedding_backend: "hash"
fallback_embedding_dim: 256
qdrant_collection_name: "conflict_aware_rag_dev"
```

如果你修改了以下任意配置：

- `embedding_backend`
- `embedding_model_name`
- `fallback_embedding_dim`
- 显式 embedding dimension
- `qdrant_collection_name`

就应该重新运行 ingestion 并重建 collection。

特别注意：

- 不要把旧 collection 和新 embedding dimension 混用。
- 如果从 `hash/256` 改成 `sentence_transformers/384`，必须重建索引。
- 建议改 embedding 设置时同时换一个新的 `qdrant_collection_name`。
- 临时本地跑通可以改自己的 local config，但不要把它当成共享 baseline。

## 7. 常见问题

### Qdrant 开着但检索为空

Qdrant 进程或本地目录存在，只代表存储服务可用，不代表 collection 里有 points。

处理方式：

```bash
python scripts/check_retrieval_ready.py \
  --config-path config/retrieval.yaml \
  --corpus-path "FEVER Dataset/wiki_pages_matched_sample.jsonl"
```

如果 point count 是 0，请重新跑 ingestion。

### Collection 为空

通常原因：

- ingestion 没有成功完成
- corpus 是空文件
- loader 选错
- collection 被删除或覆盖

处理方式：确认 corpus 非空后重新运行 `scripts/ingest_corpus.py`。

### `qdrant_path` 指错

如果 config 中的 `qdrant_path` 指向了另一个目录，P3 会去那个目录找 collection。

处理方式：

- 检查 `config/retrieval.yaml`
- 确认 `data/processed/qdrant/` 是否存在
- 不确定时重新跑 ingestion

### Collection name 不一致

如果 ingestion 用的是一个 collection name，retrieval 用的是另一个 collection name，就会找不到数据。

处理方式：

- 检查 `qdrant_collection_name`
- 如果改过名字，重新 ingestion

### 只有 config 没有 corpus

config 只告诉系统去哪里读写索引，不包含语料本身。

处理方式：先准备 `FEVER Dataset/wiki_pages_matched_sample.jsonl`，再跑 ingestion。

### 改了 embedding dim 但没有重建索引

这是最容易造成检索异常的情况之一。

处理方式：

- 换新的 collection name
- 重新跑 ingestion
- 不要混用旧 Qdrant collection

## 8. 最小 Smoke 流程

### Step 1：放好 corpus

确认文件存在：

```text
FEVER Dataset/wiki_pages_matched_sample.jsonl
```

### Step 2：跑 ingestion

```bash
python scripts/ingest_corpus.py \
  --input-path "FEVER Dataset/wiki_pages_matched_sample.jsonl" \
  --loader fever_wiki \
  --dataset fever_wiki_sample \
  --config-path config/retrieval.yaml
```

### Step 3：跑 readiness check

```bash
python scripts/check_retrieval_ready.py \
  --config-path config/retrieval.yaml \
  --corpus-path "FEVER Dataset/wiki_pages_matched_sample.jsonl" \
  --sample-query "Fox 2000 Pictures released the film Soul Food."
```

### Step 4：跑一个 retrieval 命令

```bash
python scripts/run_retrieval.py \
  --query "Fox 2000 Pictures released the film Soul Food." \
  --top-k 5 \
  --mode hybrid \
  --config-path config/retrieval.yaml
```

### Step 5：可选导出 P3 -> P1 handoff

如果本地也有 `FEVER Dataset/shared_task_dev_sample.jsonl`，可以运行：

```bash
python scripts/export_p1_handoff.py \
  --claims-path "FEVER Dataset/shared_task_dev_sample.jsonl" \
  --top-k 5 \
  --mode hybrid \
  --split dev \
  --limit 5 \
  --config-path config/retrieval.yaml \
  --output-path data/processed/p3_to_p1_smoke.json
```

如果没有 claims 文件，可以跳过这一步。本次最小复现重点是 corpus ingestion 和 retrieval readiness。

## 9. 联调替代方案

如果只是继续 P1/P2 联调，不一定要每个人都本地跑 Qdrant。

可以直接使用已经导出的 P3 handoff/example JSON，例如：

```text
data/examples/p3_to_p1_batch_query30.json
```

这种方式适合：

- P1/P2 接口联调
- 查看 handoff schema
- 快速验证 P3 输出是否能被下游消费

但如果你要验证 P3 retrieval 本身，就仍然需要准备 corpus 并跑 ingestion。
