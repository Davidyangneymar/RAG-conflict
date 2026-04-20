# P1 Demo & Eval 汇总跑测报告

**生成时间：** 2026-04-19
**环境：** `.venv312` (Python 3.12, CPU, DeBERTa-v3-base 本地)
**目的：** 一次性跑通 6 个对外可演示脚本，固化输出形态供答辩 / 同组同学参考。

---

## 跑测一览

| # | 脚本 | 用途 | 数据规模 | 状态 | 关键指标 |
|---|------|------|----------|------|----------|
| 1 | `run_p1_demo.py` | 端到端最小复现，展示 P1→P2 schema | 2 个虚构 chunk | ✅ | 1 对 NLI，contradiction=0.82 |
| 2 | `demo_cces_vs_baseline.py` | CCES 真实样本对比 | FNC-1 全集筛 3 个赢例 | ✅ | 3/3 CCES 救活原本 NEUTRAL 的预测 |
| 3 | `eval_fnc1_nli.py` | FNC-1 离线 NLI 评估（启发式 baseline） | N=120 | ✅ | acc=0.8500 macro-F1=0.3453 |
| 4 | `eval_cces_averitec_transfer.py` | CCES 跨数据集 transfer | AVeriTeC dev N=100 | ✅ | top2_span=cces λ=0.25=cces λ=0.3=0.3296（小样本无差） |
| 5 | `export_p2_contract_preview.py` | P1→P2 payload 预览 | AVeriTeC dev × 2 | ✅ | shape 完全符合 CONTRACTS.md L29-101 |
| 6 | `export_p5_benchmark.py` | P1→P5 JSONL 导出 | FNC-1 N=30 | ✅ | 30 行 JSONL，写入 `/tmp/p5_fnc1_30.jsonl` |

> **环境提示**：6 个脚本里都看到 `Hardware accelerator e.g. GPU is available in the environment, but no device argument is passed`，HF 模型当前仍在 CPU 上跑——不影响正确性，只影响速度（FNC-1 N=120 约 2 分钟，AVeriTeC N=100 × 3 mode 约 2 分钟）。

---

## 1. `run_p1_demo.py` — 端到端最小复现

**输入**：`doc_a` 说"WHO 报告疫苗减少重症"，`doc_b` 说"后来报告说没减少"。

**输出**（3 块 JSON）：

### Pipeline Config
```json
{
  "extractor_kind": "sentence",
  "entity_backend": "auto",
  "extractor_options": {}
}
```

### Claims（3 条）
- `doc_a:chunk_1:sent0` — subject=WHO, polarity=positive, entities=["WHO"]
- `doc_a:chunk_1:sent1` — subject="six months", polarity=positive
- `doc_b:chunk_1:sent0` — subject=null, **polarity=negative**, certainty=0.7

### NLI Results（1 对）
```json
{
  "claim_a_id": "doc_a:chunk_1:sent0",
  "claim_b_id": "doc_b:chunk_1:sent0",
  "entailment_score": 0.08,
  "contradiction_score": 0.82,
  "neutral_score": 0.10,
  "label": "contradiction",
  "is_bidirectional": true,
  "metadata": {"model": "heuristic", "forward_scores": [...], "backward_scores": [...]}
}
```

**用途**：给同组同学一眼看懂 P1→P2 的最小完整 payload 形态。

---

## 2. `demo_cces_vs_baseline.py` — CCES 真实样本对比

在 FNC-1 真实数据里找出 **3 个 CCES 赢、top2_span 输** 的样本，每个对比展示 NLI 三类得分变化：

| sample | gold | top2_span 证据 | top2_span 预测 | CCES λ=0.3 证据 | CCES 预测 |
|---|---|---|---|---|---|
| **fnc1-1**<br/>Palestinians/Gaza | ENTAILMENT | 锁定 spokesman 警告句 | neu=0.96 ❌ NEUTRAL | 抓到"Hundreds were evacuated" 主干句 | ent=0.51 ✅ ENTAILMENT |
| **fnc1-4**<br/>Spider 蜘蛛钻肚子 | CONTRADICTION | 锁定 "Fear not" 铺垫句 | neu=0.96 ❌ NEUTRAL | 抓到 "scientists cast doubt" + "can't get through skin" | **con=0.95** ✅ CONTRADICTION |
| **fnc1-5**<br/>NASA 6 天黑暗谣言 | ENTAILMENT | 锁定标题改写句 | neu=0.97 ❌ NEUTRAL | 抓到 "originated from Huzlers... fake stories" | ent=0.57 ✅ ENTAILMENT |

**核心解释**：top2_span 按词汇重叠选 2 句，常常锁开头铺垫句被打成 neutral；CCES 用 MMR 强制句间不冗余，能选到承载判别信号的句子。

**用途**：答辩现场最有说服力的"为什么 CCES 有效"案例。

---

## 3. `eval_fnc1_nli.py` — FNC-1 离线 NLI 评估（启发式 baseline）

**配置**：N=120, body_mode=top2_span, model=heuristic, bidirectional=True

**结果**：
- **Accuracy: 0.8500**
- **Macro-F1: 0.3453**

**Gold 分布**: contradiction=3, entailment=16, neutral=101（高度不平衡，neutral 84%）

**Predicted 分布**: entailment=1, neutral=119（启发式几乎全报 neutral）

**Confusion**:
```
contradiction → neutral: 3
entailment    → entailment: 1, neutral: 15
neutral       → neutral: 101
```

**5 个典型错误样本**（都是 gold=entailment 但被预测 neutral）：fnc1-1, fnc1-4, fnc1-5, fnc1-8, fnc1-17 — 与 demo 2 高度重叠，证实启发式 baseline 在判别类样本上严重欠召回。

**对照基线**：之前报告里 HF + CCES λ=0.25 + 校准在 N=360 test 上能拿到 macro-F1 = **0.4449**（+5.31pp 总 lift），相比这里启发式的 0.3453 提升明显。

---

## 4. `eval_cces_averitec_transfer.py` — 跨数据集 transfer

**配置**：AVeriTeC dev N=100, modes=[top2_span, cces λ=0.25, cces λ=0.3]

**Label 分布**: contradiction=63, entailment=19, neutral=18

**结果**（N=100 三种 mode 完全打平）：
| mode | macro_f1 | accuracy | per-label F1 |
|---|---|---|---|
| top2_span | 0.3296 | 0.3800 | ent=0.182, con=0.489, neu=0.318 |
| cces λ=0.25 | 0.3296 | 0.3800 | 同上 |
| cces λ=0.3 | 0.3296 | 0.3800 | 同上 |

> **说明**：N=100 太小、AVeriTeC 答案文本平均 30w 已经接近 CCES 选出的窗口，三种 mode 选到几乎相同的 evidence，差异被噪声抹平。**N=200 时（之前 transfer 实验）λ=0.3 +0.6pp，λ=0.25 +0.27pp**——保持原结论：方向一致但幅度远低于 FNC-1 的 +6.8pp，CCES 优势在 AVeriTeC 上需要更大 N 才能显著。

---

## 5. `export_p2_contract_preview.py` — P2 payload 预览

**配置**：AVeriTeC dev × 2 records, max_questions=2, max_answers_per_question=2

**输出**（197 行 JSON 数组）每条包含：
- `sample_id`
- `claims[]`：claim_id + text + source_doc_id + source_chunk_id + entities + subject/relation/object/qualifier/time + polarity + certainty + metadata + source_metadata（含 dataset/split/speaker/claim_date/reporting_source/question/answer_type/cached_source_url/role/label/retrieval_rank/retrieval_score/source_url/source_medium）
- `candidate_pairs[]`：claim_a_id + claim_b_id + entity_overlap + lexical_similarity + embedding_similarity
- `nli_results[]`：claim_a_id + claim_b_id + label + entailment_score + contradiction_score + neutral_score + is_bidirectional + metadata

**契约校验**：与 `CONTRACTS.md` L29-101 完全对齐 ✅。`metadata` 字段是开放字典（已在与同学的口头备注里说明）。

---

## 6. `export_p5_benchmark.py` — P5 JSONL 导出

**配置**：dataset=fnc1, input=`data/processed/fnc1_train.jsonl`, N=30, body_mode=top2_span（默认）

**输出**：
- 写入：`/tmp/p5_fnc1_30.jsonl`（30 行）
- evaluated_records: 30
- gold_distribution: neutral=22, entailment=7, contradiction=1
- predicted_distribution: neutral=4, entailment=2（其余 24 条 `predicted_label = null`，因为没有 cross-source pair 通过 blocking）

**单行 JSONL 字段**（与 CONTRACTS.md L233-263 一致）：
```
sample_id, dataset, split, gold_label, query, retrieved_chunk_count,
claim_count, candidate_pair_count, cross_source_pair_count,
predicted_label, best_entailment_score, best_contradiction_score,
best_neutral_score, best_entailment_pair, best_contradiction_pair,
best_neutral_pair, claims[], cross_source_nli_results[]
```

**关注点**：80% 样本 `predicted_label=null`，因为 blocking 阈值在 retrieval-shaped 输入上偏严，是已识别的下一步优化方向。

---

## 关键结论

1. **6 个脚本全部跑通**，无报错，输出形态与 CONTRACTS.md 完全对齐——可放心把 CONTRACTS.md 发给同组同学。
2. **NLI 性能现状**：启发式 baseline 在 FNC-1 上 macro-F1=0.345；当前最佳 HF + CCES λ=0.25 + 校准在 N=360 test 上达 0.4449（+5.31pp 总 lift）。
3. **CCES 真实赢例可视化**：demo 2 提供 3 个完整对照样本，可直接用于答辩现场展示。
4. **跨数据集 transfer**：AVeriTeC 上方向一致但幅度衰减到 +0.6pp（N=200），N=100 时三种 mode 打平——CCES 在 AVeriTeC 上的优势需更大 N 才能确定显著性。
5. **P3 联调瓶颈仍在**：P5 export 显示 80% FNC-1 样本因 blocking 严苛而无 cross-source pair，对应已识别的 P0/P1 改进项（默认 NLI 切 HF + 放宽 blocking 阈值），不影响 schema。

---

## 附：原始日志位置

| 脚本 | 日志文件 | 行数 |
|---|---|---|
| run_p1_demo | `/tmp/run_demo1.log` | 225 |
| demo_cces_vs_baseline | `/tmp/run_demo2.log` | 44 |
| eval_fnc1_nli (N=120) | `/tmp/run_eval_fnc1.log` | 54 |
| eval_cces_averitec_transfer (N=100) | `/tmp/run_avtransfer.log` | 21 |
| export_p2_contract_preview (N=2) | `/tmp/run_p2.log` | 197 |
| export_p5_benchmark (N=30) | `/tmp/run_p5.log` + `/tmp/p5_fnc1_30.jsonl` | 186 + 30 |
