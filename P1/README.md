# P1：NLI 矛盾检测 + Claim 抽取

Conflict-Aware RAG 课程项目的 P1 模块。负责把上游 chunk 转成结构化 claim、做候选对 blocking、跑 NLI 判定 entailment / contradiction / neutral，输出给 P2 / P5。

## 模块文档

- `CONTRACTS.md` — P1 与 P2 / P3 / P5 的对外接口契约（这是同组同学需要看的唯一接口文档）
- `DEMO_RUN_SUMMARY.md` — 6 个可演示脚本的跑测输出汇总

## 目录结构

```
src/p1/
├── schemas.py             # Claim / ClaimPair / NLIPairResult 数据契约
├── claim_extraction.py    # 三层抽取器：sentence / structured / structured_llm
├── blocking.py            # 实体→词汇→嵌入级联 blocking
├── evidence_selection.py  # CCES (Claim-Conditioned Evidence Selector)
├── nli.py                 # 启发式 NLI + HF 交叉编码器接口
├── llm_nli.py             # LLM-backed NLI 接口
├── nli_ensemble.py        # 多 backend 加权融合 + 温度校准 + 阈值
├── pipeline.py            # chunk → claim → blocking → NLI 编排
├── handoff.py             # P2 payload 转换
├── benchmark.py           # P5 benchmark 导出
├── stats.py               # paired bootstrap 显著性
└── data/
    ├── fnc1.py            # FNC-1 adapter
    ├── averitec.py        # AVeriTeC adapter
    └── retrieval.py       # P3 retrieval-shaped 输入读取

scripts/                   # 数据预处理 / 评估 / 导出 / demo 脚本
tests/                     # pytest 用例
```

## 环境

```bash
python -m venv .venv312
source .venv312/bin/activate
pip install -e .
```

可选依赖：
- HuggingFace 交叉编码器（推荐本地放在 `manual_models/DeBERTa-v3-base-mnli-fever-anli/`）
- spaCy 英文模型（`python -m spacy download en_core_web_sm`）
- 火山引擎 Ark API：`export P1_LLM_API_KEY=... P1_LLM_MODEL=...`

## 数据准备

> **注意**：本提交不含数据集和模型权重。FNC-1 / AVeriTeC 数据集请按 `scripts/prepare_fnc1.py` 注释里的链接自行下载并放到 `data/` 下；NLI 模型 `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` 第一次跑会自动从 HuggingFace 下载到本地缓存。

FNC-1：
```bash
python scripts/prepare_fnc1.py \
  --bodies data/fnc1/train_bodies.csv \
  --stances data/fnc1/train_stances.csv \
  --output data/processed/fnc1_train.jsonl
```

AVeriTeC：直接用官方 dev split，路径 `data/averitec/dev/dev.json`。

## 快速跑通

最小端到端 demo：
```bash
python scripts/run_p1_demo.py
```

CCES 真实样本对比 demo：
```bash
python scripts/demo_cces_vs_baseline.py
```

FNC-1 离线评估：
```bash
python scripts/eval_fnc1_nli.py --limit 360
```

P2 payload 预览（同学对接用）：
```bash
python scripts/export_p2_contract_preview.py --limit 2
```

P5 benchmark JSONL 导出（支持 fnc1 / averitec / P3 retrieval_json 三种输入）：
```bash
# FNC-1
python scripts/export_p5_benchmark.py \
  --dataset fnc1 \
  --input data/processed/fnc1_train.jsonl \
  --output data/processed/p5_benchmark_fnc1.jsonl

# P3 retrieval JSON（单条 dict 或数组都支持）
python scripts/export_p5_benchmark.py \
  --dataset retrieval_json \
  --input path/to/p3_to_p1.json \
  --output outputs/p5_from_p3.jsonl
```

更多脚本输出示例见 `DEMO_RUN_SUMMARY.md`。

## 测试

```bash
pytest tests/
```

## 给同组同学的对接说明

- **P2 接 P1**：调用 `pipeline_output_to_p2_payload()`，按 `CONTRACTS.md` "P1→P2" 节消费
- **P3 喂 P1**：按 `CONTRACTS.md` "P3→P1" 节构造 retrieval-shaped JSON，调 `pipeline.run_retrieval_input()`
- **P5 评 P1**：跑 `scripts/export_p5_benchmark.py` 产出 JSONL，按 "P1→P5" 节消费

> `metadata` 字段是开放字典，请按"读已知 key"原则消费，不要断言只有契约里列出的 key。
