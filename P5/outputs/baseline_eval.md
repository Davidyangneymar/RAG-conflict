# P5 Baseline Evaluation — FEVER Dev Set

## 指标说明
| 指标 | 含义 |
|---|---|
| contradiction_f1 | 矛盾类（REFUTES）的 F1 |
| stance_macro_f1 | 三分类宏平均 F1（SUPPORTS/REFUTES/NOT ENOUGH INFO）|
| missing_prediction_rate | 缺失预测比例 |
| abstention_rate | 主动弃答比例（已对齐样本中）|
| alignment_rate | sample_id 对齐率 |
| support | gold 样本总数 |

## 结果汇总

| baseline | contradiction_f1 | stance_macro_f1 | missing_prediction_rate | abstention_rate | alignment_rate | support |
|---|---:|---:|---:|---:|---:|---:|
| vanilla | 0.0000 | 0.1667 | 0.0000 | 0.0000 | 1.0000 | 19998 |
| reranker | 0.1926 | 0.3324 | 0.0000 | 0.0000 | 1.0000 | 19998 |
| nli | 0.1734 | 0.2279 | 0.0000 | 0.0000 | 1.0000 | 19998 |
| full | 0.1734 | 0.2279 | 0.0000 | 0.0000 | 1.0000 | 19998 |

## 各基线说明

| baseline | 策略 |
|---|---|
| vanilla | 常数预测：全部预测 NOT ENOUGH INFO（无任何推理）|
| reranker | 关键词重叠启发式：含否定词 → REFUTES；含正向词 → SUPPORTS；否则 → NOT ENOUGH INFO |
| nli | P1 HeuristicNLIModel：基于词汇否定信号的双向 NLI 推理（无外部模型）|
| full | 与 nli 相同（无真实检索时，完整系统降级为 NLI-only）|

## 分析

- **vanilla**：stance_macro_f1=0.1667，contradiction_f1=0（常数基线，仅预测 NEI，矛盾检测完全失效）
- **reranker**：stance_macro_f1=0.3324，contradiction_f1=0.1926
  - 关键词启发式优于常数基线，但因大量 FEVER claim 包含正向词（is/was/are），SUPPORTS 严重虚报（precision 低）
- **nli**：stance_macro_f1=0.2279，contradiction_f1=0.1734
  - P1 HeuristicNLI 仅捕获显式否定词，REFUTES 召回低；NEI 类比例(33.3%)高时 macro_f1 受抑制
- **full**：与 nli 相同（无真实检索输入时，全系统等价于 NLI-only）

### 主要发现
1. 无真实检索结果时，所有基线均受限于 claim-only 信息，macro_f1 在 0.17–0.33 区间。
2. contradiction_f1（矛盾检测核心指标）最高仅 0.19（reranker），说明关键词策略对显式否定有一定捕获能力。
3. P1 HeuristicNLI 在 macro_f1 上低于关键词启发式，原因在于 FEVER claim 的否定词策略过于保守。
4. 引入真实检索（P3 BM25 + reranker）后，nli/full 基线预计有显著提升空间。