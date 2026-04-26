# P5: 数据与评测工作区

本目录用于完成 P5 任务，且不修改 P1/P2/P3 源码。

## 已实现能力

- 统一 benchmark 数据格式（FaithEval / AmbigDocs / CONFLICTS / P1 导出）
- 生成小样本人工标注模板
- 四组基线统一评测：`vanilla RAG` / `+reranker` / `+NLI` / `full system`
- 指标输出：`contradiction-F1`、`stance-macro-F1`、`abstention rate`、`missing prediction rate`、`alignment rate`
- 提供兼容 P1/P2/P3 的环境构建脚本（Python 3.12）

## 目录

- `src/p5/`: 评测核心逻辑
- `scripts/setup_env_p123.ps1`: 新建兼容环境
- `scripts/normalize_benchmarks.py`: 统一 benchmark 格式
- `scripts/make_annotation_template.py`: 小样本标注模板
- `scripts/generate_preds_nli.py`: 生成 +NLI baseline 预测
- `scripts/evaluate_baselines.py`: 四基线评测
- `scripts/run_p1_export_for_p5.py`: 调用 P1 导出契约数据

## 1) 新建兼容环境（P1/P2/P3）

在 `P5/` 下执行：

```powershell
./scripts/setup_env_p123.ps1
```

说明：
- 环境位于 `P5/.venv_p123`
- 脚本默认安装 P2 依赖、P1 editable、P3 editable、P5 editable
- 若只想快速建空环境可用：

```powershell
./scripts/setup_env_p123.ps1 -Lightweight
```

## 2) 统一 benchmark 数据

```powershell
.\.venv_p123\Scripts\python.exe scripts/normalize_benchmarks.py --source faitheval --input data/raw/faith_eval.jsonl --output data/processed/faith_eval.normalized.jsonl
```

`--source` 支持：`faitheval` / `ambigdocs` / `conflicts` / `p1_benchmark`

默认要求输入包含可监督标签（如 `gold_label` / `label` / `stance` / `verdict`）。
若仅做预处理或人工标注模板生成，可显式开启：

```powershell
.\.venv_p123\Scripts\python.exe scripts/normalize_benchmarks.py --source faitheval --input data/raw/faith_eval.jsonl --output data/processed/faith_eval.normalized.jsonl --allow-unlabeled
```

## 3) 生成小样本标注模板

```powershell
.\.venv_p123\Scripts\python.exe scripts/make_annotation_template.py --input data/processed/faith_eval.normalized.jsonl --output data/annotation/small_sample_annotations.csv --sample-size 60
```

## 4) 生成 +NLI baseline 预测

+NLI baseline 使用 P1 的 `HeuristicNLIModel` 对每条 claim 进行启发式 NLI 推理（仅依赖词汇和否定信号，不需要检索结果）。

生成预测文件：

```powershell
.\.venv_p123\Scripts\python.exe scripts/generate_preds_nli.py \
  --input data/processed/fever_dev.normalized.jsonl \
  --output outputs/preds_nli.jsonl
```

说明：
- 策略：对每条 claim，构造一对 (原始claim, 去否定词后的claim)，用 P1 NLI 模型判断
- 标签映射：`entailment → SUPPORTS`, `contradiction → REFUTES`, `neutral → NOT ENOUGH INFO`
- 预期：无检索的启发式基线，仅能捕获显式否定/反驳词汇的样本
- 输出：`sample_id` + `predicted_label` JSONL

## 5) 四组基线评测

输入 JSONL 至少包含：`sample_id`, `predicted_label`

```powershell
.\.venv_p123\Scripts\python.exe scripts/evaluate_baselines.py \
  --gold data/processed/fever_dev.normalized.jsonl \
  --vanilla outputs/preds_vanilla.jsonl \
  --reranker outputs/preds_reranker.jsonl \
  --nli outputs/preds_nli.jsonl \
  --full outputs/preds_full.jsonl \
  --min-align-rate 0.8 \
  --output outputs/baseline_eval.json
```

说明：
- 每个 baseline 都会先做 `sample_id` 对齐率检查；若低于 `--min-align-rate` 则直接失败。
- `abstention_rate` 只统计“已对齐样本中的主动弃答”。
- 缺失预测会单独记为 `missing_prediction_rate`。

会同时生成：
- `outputs/baseline_eval.json`
- `outputs/baseline_eval.md`
- `outputs/baseline_eval.unmatched.json`（每个 baseline 未对齐的 sample_id 列表）

## 6) 演示数据快速验证

```powershell
.\.venv_p123\Scripts\python.exe scripts/evaluate_baselines.py \
  --gold data/processed/demo_gold.normalized.jsonl \
  --vanilla outputs/demo_preds_vanilla.jsonl \
  --reranker outputs/demo_preds_reranker.jsonl \
  --nli outputs/demo_preds_nli.jsonl \
  --full outputs/demo_preds_full.jsonl \
  --output outputs/demo_eval.json
```

## 7) 调用 P1 导出契约（不改 P1 代码）

```powershell
.\.venv_p123\Scripts\python.exe scripts/run_p1_export_for_p5.py \
  --python ./.venv_p123/Scripts/python.exe \
  --dataset retrieval_json \
  --input ../P3/data/processed/p3_to_p1_batch_query30.json \
  --output outputs/p1_benchmark_from_p3.jsonl \
  --limit 30
```

## 路径约束

- 本目录所有脚本默认使用相对路径。
- 若传入自定义路径，也建议保持为相对路径以便跨机器复现。

推荐两种写法（都可复现）：

- 在工作区根目录 `NLP/` 执行：

```powershell
./P5/.venv_p123/Scripts/python.exe P5/scripts/normalize_benchmarks.py --source faitheval --input data/raw/faith_eval.jsonl --output P5/data/processed/faith_eval.normalized.jsonl
```

- 在 `P5/` 目录执行：

```powershell
./.venv_p123/Scripts/python.exe scripts/normalize_benchmarks.py --source faitheval --input ../data/raw/faith_eval.jsonl --output data/processed/faith_eval.normalized.jsonl
```

## 已知问题

- 在部分 Windows 机器上，`torch` 可能出现 `WinError 1114`（DLL 初始化失败）。
- `scripts/setup_env_p123.ps1` 已内置自动回退：当默认安装后导入失败，会自动切到可工作的 CPU 组合：
  `torch==2.3.1`, `torchvision==0.18.1`, `torchaudio==2.3.1`。
- 若自动回退后仍失败，再检查系统运行时（VC++ Redistributable x64）与显卡/驱动环境。
