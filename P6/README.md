# P6：Controlled Generation Strategy Layer

P6 是独立策略层，专门消费 P2 `ConflictTypedOutput`，不重复冲突分类。

## 目录

```
P6/
├── README.md
├── pyproject.toml
├── docs/
│   └── TECHNICAL_DESIGN.md
└── src/p6/
    ├── __init__.py
    ├── contracts.py
    ├── planner.py
    └── extensions.py
```

## 功能

- 构建 `AnswerContext`：query / 证据簇 / 冲突摘要 / 引用来源 / claim trace
- 按 `resolution_policy` 选择 Prompt 策略
- 两阶段生成提示（分析草案 -> 结构化最终答案）
- 拒答门控（hard contradiction 与低置信条件触发）
- 预留 P5 和其他模块的标准化交换接口

## 与 P2 的兼容

- P2 仍对外导出原有 `build_answer_plan_for_sample` 等 API
- 现通过兼容层转发到 `P6/src/p6` 实现
- 现有数据流与调用方式保持不变

## 快速使用

```python
from p6 import build_answer_plan_for_sample

plan = build_answer_plan_for_sample(typed_sample, input_record)
print(plan.prompt_bundle.strategy_name)
```

## 与 P5/未来模块的扩展通道

- `to_exchange_payload(plan)`：标准化跨模块交互数据
- `build_p5_feedback_payload(plan)`：生成可直接聚合的 P5 反馈指标载荷
- `SimpleP5FeedbackHook`：默认 P5 反馈回调实现
- `JsonlDownstreamExporter`：通用 JSONL 下游发布实现
- `P5FeedbackHook` / `DownstreamExporter`：可替换协议接口
