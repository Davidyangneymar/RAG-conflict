# P2 对外接口契约

本文件 freeze P2 (stance analysis + conflict typing) 与上下游模块的数据
交换契约,补全队友 P1 维护的 `CONTRACTS.md` 中缺失的 **P2 → conflict
strategy / generation** 那一节。

## 目录

- [P1 → P2 输入契约](#p1--p2-输入契约) (消费端视角,字段取自 P1 契约)
- [P2 → 下游输出契约](#p2--下游输出契约)
- [Dev-only: AVeriTeC → P1-shape 适配](#dev-only-averitec--p1-shape-适配)

---

## P1 → P2 输入契约

Last verified: 2026-04-18

P2 消费 P1 的 JSON(顶层可以是单个 sample dict 或 list),入口是
`src.p2.load_p1_payload(path)` 或 `src.p2.parse_p1_payload(data)`。
所有 join 都按 id (`claim_id`, `claim_a_id`, `claim_b_id`),不依赖列表
顺序。

必读字段(来自 P1 `CONTRACTS.md`):

- `sample_id`
- `claims[].claim_id, text, subject?, relation?, object?, qualifier?, time?, source_metadata.role?`
- `candidate_pairs[].claim_a_id, claim_b_id`
- `nli_results[].claim_a_id, claim_b_id, label`  (label ∈ {entailment, contradiction, neutral})

缺失 optional 字段容错;缺失 required 字段抛 `P1SchemaError` 并带定位路径
(如 `payload[0].claims[2]: missing required field 'text'`)。

---

## P2 → 下游输出契约

Last updated: 2026-04-18

### Entry Point

- `src.p2.run_full_p2_pipeline_from_path(p1_payload_path, ...)`
- `src.p2.run_full_p2_pipeline_from_records(records, ...)`

返回类型 `ConflictTypedOutput`,调用 `.to_dict()` 即得 JSON-safe dict。

### Top-Level Shape

```json
{
  "samples": [ <TypedSample>, ... ]
}
```

### `TypedSample`

```json
{
  "sample_id": "string",
  "pair_results": [ <TypedPair>, ... ],
  "type_counts": { "<conflict_type>": <int>, ... },
  "gold_verdict": "Supported | Refuted | Not Enough Evidence | Conflicting Evidence/Cherrypicking | null"
}
```

`gold_verdict` 仅在从 benchmark 数据(AVeriTeC 等)入口跑时填充,其他
情况为 `null`。下游不能假设它总是存在。

### `TypedPair`

```json
{
  "stance": <StancedPair>,
  "conflict_type": "none | hard_contradiction | temporal_conflict | opinion_conflict | ambiguity | misinformation | noise",
  "typing_confidence": 0.0,
  "resolution_policy": "pass_through | prefer_latest | show_all_sides | disambiguate_first | down_weight_low_quality | abstain | skip",
  "rationale": ["..."]
}
```

`rationale` 是人类可读的诊断,做展示/调试用;回答策略路由只应 switch
`conflict_type` 或 `resolution_policy`,**不要**解析 `rationale` 字符串。

### `StancedPair` (nested inside TypedPair.stance)

```json
{
  "claim_a_id": "string",
  "claim_b_id": "string",
  "nli_label": "entailment | contradiction | neutral | null",
  "stance_label": "support | oppose | neutral | filtered | null",
  "stance_decision_score": 0.0,
  "stance_direction": "a_as_claim | b_as_claim | bidirectional",
  "is_filtered": false,
  "agreement_signal": "agreement | conflict | neutral | unrelated | inconclusive",
  "fusion_confidence": 0.0,
  "notes": ["..."]
}
```

### Conflict Type → Resolution Policy 默认路由

| conflict_type | resolution_policy | 语义 |
|---|---|---|
| `none` | `pass_through` | 正常回答 |
| `temporal_conflict` | `prefer_latest` | 优先较新证据 |
| `opinion_conflict` | `show_all_sides` | 并列展示双方立场 |
| `ambiguity` | `disambiguate_first` | 先消歧或多答案 |
| `misinformation` | `down_weight_low_quality` | 降权可疑来源 |
| `hard_contradiction` | `abstain` | 拒答 / 低置信度提示 |
| `noise` | `skip` | 跳过该 pair |

### 最小可用子集

下游如果只 care 路由决策,读这四个字段就够:

- `sample_id`
- `pair_results[i].stance.claim_a_id / claim_b_id`  (拿回原 claim)
- `pair_results[i].conflict_type`
- `pair_results[i].resolution_policy`

### Stability Notes

- `conflict_type` / `resolution_policy` 词表是 frozen 的;任何扩展都要
  加新值,不能重命名旧值(下游可能 switch)。
- `stance_label="filtered"` 严格等价于 `agreement_signal="unrelated"`
  和 `conflict_type="noise"`,三者同时成立。
- `fusion_confidence` 和 `typing_confidence` 都是 0..1,但语义不同:
  前者衡量 stance 和 NLI 是否互相印证,后者衡量 conflict type 判断
  的可信度。不要互相比较。

---

## Dev-only: AVeriTeC → P1-shape 适配

Last updated: 2026-04-18

`src.p2.datasets.averitec` 提供开发期适配,用于在 P1 未完成前
验证 P2 端到端闭环。**不是正式契约的一部分**,P1 就绪后应移除调用方。

映射规则:

| AVeriTeC 字段 | P1-shape 字段 |
|---|---|
| `claim` | `claims[0].text`, `claim_id="Q"`, `source_metadata.role="query"` |
| `claim_date` | `claims[0].time`, `source_metadata.claim_date` |
| `speaker` | `claims[0].source_metadata.speaker` |
| `questions[].answers[].answer` | 每个生成一条 claim, `role="retrieved_evidence"` |
| `questions[].answers[].source_url` | `claims[i].source_metadata.source_url` |
| `questions[].answers[].source_medium` | `claims[i].source_metadata.source_medium` |
| `questions[].question` | `claims[i].source_metadata.question` |
| `label` | benchmark passthrough, 写入 `TypedSample.gold_verdict` |

注意:

- `nli_results` 一律留空(`[]`)。这很重要:我们**不**用 AVeriTeC 的
  verdict 标签伪造 NLI 标签。P2 融合层对 `nli_label=None` 已经有明确
  fallback 路径(stance-only)。
- `candidate_pairs` 是 `Q × {E0, E1, ...}` 的 cross-product — 查询
  对每条 evidence。这是一条简化的 blocking 规则;P1 正式的 blocking
  用实体重叠 / 时间窗口 / embedding 相似度,会更稀疏。
