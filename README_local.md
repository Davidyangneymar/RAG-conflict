# Conflict-Aware RAG Project (P2-Centric)

本仓库是小组项目中的 P2 子系统实现，目标是把 P1 输出的 claim/pair/NLI
转换成可被下游策略与生成模块直接消费的结构化冲突信号。

核心数据流：

```text
P1 JSON
  -> src.p2.p1_adapter.parse_p1_payload
  -> src.p2.stance.PairStanceRunner
  -> src.p2.conflict_typing.type_sample
  -> ConflictTypedOutput(JSON)
```

输出是 pair 级别的：

- stance 结果
- conflict_type
- resolution_policy
- rationale(可解释证据)

## 1. 仓库结构整理

### 核心代码

- [src/p2](src/p2): P2 主包
- [src/p2/p1_adapter.py](src/p2/p1_adapter.py): P1 输入解析与 schema 校验
- [src/p2/stance](src/p2/stance): 立场推理、role 路由、NLI 融合
- [src/p2/conflict_typing](src/p2/conflict_typing): 冲突类型规则路由
- [src/p2/pipeline.py](src/p2/pipeline.py): 对外 pipeline 入口

### 运行与测试脚本

- [scripts/run_all_tests.sh](scripts/run_all_tests.sh): 一键验证
- [scripts/run_p2_pipeline.py](scripts/run_p2_pipeline.py): stance-only
- [scripts/run_averitec_pipeline.py](scripts/run_averitec_pipeline.py): 全链路
- [scripts/inspect_p2_on_averitec.py](scripts/inspect_p2_on_averitec.py): 批量诊断
- [scripts/analyze_p2_pairs_csv.py](scripts/analyze_p2_pairs_csv.py): 对 pairs.csv 做统计汇总
- [scripts/profile_averitec_sources.py](scripts/profile_averitec_sources.py): 提取 AVeriTeC 来源分布
- [scripts/smoke_test_p2_fusion.py](scripts/smoke_test_p2_fusion.py): 融合单测
- [scripts/smoke_test_conflict_typing.py](scripts/smoke_test_conflict_typing.py): 冲突规则单测
- [scripts/test_contract.py](scripts/test_contract.py): 输入契约测试

### 模型与实验

- [fnc1_bert_stance_module.py](fnc1_bert_stance_module.py): 冻结推理模块
- [outputs/fnc1_bert_upgrade_full](outputs/fnc1_bert_upgrade_full): 运行时模型产物
- [fnc1_bert_upgrade_experiment.py](fnc1_bert_upgrade_experiment.py): 训练脚本
- [fnc1_baseline.py](fnc1_baseline.py): baseline
- [fnc1_error_analysis_experiments.py](fnc1_error_analysis_experiments.py): 误差分析

### 文档

- [docs/CONTRACTS.md](docs/CONTRACTS.md): P1->P2 与 P2->下游契约
- [P2_DELIVERABLE.md](P2_DELIVERABLE.md): 交付清单
- [BERT_STANCE_MODULE_USAGE.md](BERT_STANCE_MODULE_USAGE.md): BERT 模块说明

## 2. 环境与安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

运行时如需完整链路，请确保目录存在：

- [outputs/fnc1_bert_upgrade_full](outputs/fnc1_bert_upgrade_full)

## 3. 快速开始

### 一键验证

```bash
bash scripts/run_all_tests.sh
```

当前验证内容：

1. P1->P2 契约测试
2. stance+NLI 融合 smoke tests
3. conflict typing smoke tests
4. AVeriTeC 全链路小样本运行

### 运行全链路示例

```bash
python scripts/run_averitec_pipeline.py scripts/sample_averitec_records.json
```

输出默认写入：

- [outputs/averitec_demo/p1_payload.json](outputs/averitec_demo/p1_payload.json)
- [outputs/averitec_demo/p2_final.json](outputs/averitec_demo/p2_final.json)

### 运行诊断报告

```bash
python scripts/inspect_p2_on_averitec.py \
  scripts/sample_averitec_records.json \
  --out_dir outputs/p2_inspection_sample_after
```

输出：

- [outputs/p2_inspection_sample_after/report.txt](outputs/p2_inspection_sample_after/report.txt)
- [outputs/p2_inspection_sample_after/pairs.csv](outputs/p2_inspection_sample_after/pairs.csv)

## 4. 本次优化内容(基于报告)

根据你给的诊断报告，问题集中在冲突召回不足和规则触发失衡。本次优化主要落在
[src/p2/conflict_typing/typer.py](src/p2/conflict_typing/typer.py) 与
[scripts/inspect_p2_on_averitec.py](scripts/inspect_p2_on_averitec.py)。

已完成优化：

1. 增加 agreement/neutral 的“隐藏冲突回收”路径
2. 增加 subject 缺失时的人名同名实体消歧回退
3. 增加 source-quality gap 的保守 override 路由（优先落到 ambiguity）
4. 增强 source_medium 归一化、reporting_source 推断和 archived URL 域名解析
5. 把关键阈值集中成常量，便于后续 ablation
6. 扩展 smoke tests 到 22 条断言，覆盖新增行为
7. 更新 inspection 规则归因，新增 R0d_hidden_conflict_override / R0e_source_gap_override
8. AVeriTeC adapter 增补 query 侧来源元数据（reporting_source/source_medium/source_url）

## 5. 优化结果

### 100 样本对比（与你之前报告同口径）

- before: [outputs/p2_inspection/report.txt](outputs/p2_inspection/report.txt)
- final: [outputs/p2_inspection_final_100_v3/report.txt](outputs/p2_inspection_final_100_v3/report.txt)

关键变化：

1. none 从 60.5% 降到 46.9%
2. misinformation 从 0 提升到 6.2%
3. hard_contradiction + ambiguity 合计从 3.1% 提升到 7.0%
4. suspicious samples 从 33 降到 21

### 全量 dev 对比（500 samples）

- previous full baseline: [outputs/p2_inspection_after_opt_full/report.txt](outputs/p2_inspection_after_opt_full/report.txt)
- final full: [outputs/p2_inspection_final_full_v3/report.txt](outputs/p2_inspection_final_full_v3/report.txt)

关键变化：

1. none 从 49.0% 降到 44.7%
2. misinformation 从 0 提升到 6.0%
3. R5_misinfo 从 0 次触发提升到 33 次
4. suspicious samples 从 97 降到 91

### 4 样本 smoke 对比

- before: [outputs/p2_inspection_sample_before/report.txt](outputs/p2_inspection_sample_before/report.txt)
- after: [outputs/p2_inspection_sample_after/report.txt](outputs/p2_inspection_sample_after/report.txt)

关键变化：

1. none 从 87.5% 降到 75.0%
2. hard_contradiction 从 0 提升到 12.5%
3. suspicious samples 从 1 降到 0
4. 新规则 R0d 成功触发(可解释回收路径生效)

## 6. 对外 API

```python
from src.p2 import (
    load_p1_payload,
    parse_p1_payload,
    run_p2_pipeline_from_path,
    run_full_p2_pipeline_from_path,
    StancedPair,
    TypedPair,
    ConflictTypedOutput,
)
```

生产环境建议直接使用：

- run_full_p2_pipeline_from_path(...)

契约以 [docs/CONTRACTS.md](docs/CONTRACTS.md) 为准。

## 7. 注意事项

1. 当前 conflict typing 是规则系统，强调可解释性与可调试性
2. AVeriTeC adapter 属于 dev-only，P1 正式联调后可替换
3. 模型训练脚本与运行脚本解耦，线上推理只依赖冻结模型目录

## 8. 建议的下一步

1. 对完整 AVeriTeC dev 集合重跑 inspection，观察 R3/R5 的真实触发率
2. 在不改变输出契约的前提下，把规则阈值外置到配置文件
3. 为 R5(misinformation)增加 source_url/domain 级信号
