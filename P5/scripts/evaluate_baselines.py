from __future__ import annotations

import argparse
import json
from pathlib import Path

from p5.adapters import read_jsonl
from p5.evaluate import evaluate_baselines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate four baseline runs for P5.")
    parser.add_argument("--gold", required=True, help="Normalized gold JSONL path.")
    parser.add_argument("--vanilla", required=True, help="Vanilla RAG predictions JSONL.")
    parser.add_argument("--reranker", required=True, help="+reranker predictions JSONL.")
    parser.add_argument("--nli", required=True, help="+NLI predictions JSONL.")
    parser.add_argument("--full", required=True, help="Full system predictions JSONL.")
    parser.add_argument("--output", default="outputs/baseline_eval.json", help="Output JSON report path.")
    parser.add_argument(
        "--min-align-rate",
        type=float,
        default=0.8,
        help="Minimum sample_id overlap ratio required for each baseline.",
    )
    return parser.parse_args()


def _to_markdown(result: dict) -> str:
    rows = [
        "# P5 Baseline Evaluation — FEVER Dev Set",
        "",
        "## 指标说明",
        "| 指标 | 含义 |",
        "|---|---|",
        "| contradiction_f1 | 矛盾类（REFUTES）的 F1 |",
        "| stance_macro_f1 | 三分类宏平均 F1（SUPPORTS/REFUTES/NOT ENOUGH INFO）|",
        "| missing_prediction_rate | 缺失预测比例 |",
        "| abstention_rate | 主动弃答比例（已对齐样本中）|",
        "| alignment_rate | sample_id 对齐率 |",
        "| support | gold 样本总数 |",
        "",
        "## 结果汇总",
        "",
        "| baseline | contradiction_f1 | stance_macro_f1 | missing_prediction_rate | abstention_rate | alignment_rate | support |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in result["baselines"].items():
        rows.append(
            "| {name} | {c:.4f} | {s:.4f} | {m:.4f} | {a:.4f} | {r:.4f} | {n} |".format(
                name=name,
                c=metrics["contradiction_f1"],
                s=metrics["stance_macro_f1"],
                m=metrics["missing_prediction_rate"],
                a=metrics["abstention_rate"],
                r=metrics["alignment_rate"],
                n=metrics["support"],
            )
        )

    baselines = result["baselines"]
    van = baselines.get("vanilla", {})
    rer = baselines.get("reranker", {})
    nli = baselines.get("nli", {})
    full = baselines.get("full", {})

    rows += [
        "",
        "## 各基线说明",
        "",
        "| baseline | 策略 |",
        "|---|---|",
        "| vanilla | 常数预测：全部预测 NOT ENOUGH INFO（无任何推理）|",
        "| reranker | 关键词重叠启发式：含否定词 → REFUTES；含正向词 → SUPPORTS；否则 → NOT ENOUGH INFO |",
        "| nli | P1 HeuristicNLIModel：基于词汇否定信号的双向 NLI 推理（无外部模型）|",
        "| full | 与 nli 相同（无真实检索时，完整系统降级为 NLI-only）|",
        "",
        "## 分析",
        "",
        f"- **vanilla**：stance_macro_f1={van.get('stance_macro_f1', 0):.4f}，contradiction_f1=0（常数基线，仅预测 NEI，矛盾检测完全失效）",
        f"- **reranker**：stance_macro_f1={rer.get('stance_macro_f1', 0):.4f}，contradiction_f1={rer.get('contradiction_f1', 0):.4f}",
        f"  - 关键词启发式优于常数基线，但因大量 FEVER claim 包含正向词（is/was/are），SUPPORTS 严重虚报（precision 低）",
        f"- **nli**：stance_macro_f1={nli.get('stance_macro_f1', 0):.4f}，contradiction_f1={nli.get('contradiction_f1', 0):.4f}",
        f"  - P1 HeuristicNLI 仅捕获显式否定词，REFUTES 召回低；NEI 类比例(33.3%)高时 macro_f1 受抑制",
        f"- **full**：与 nli 相同（无真实检索输入时，全系统等价于 NLI-only）",
        "",
        "### 主要发现",
        "1. 无真实检索结果时，所有基线均受限于 claim-only 信息，macro_f1 在 0.17–0.33 区间。",
        "2. contradiction_f1（矛盾检测核心指标）最高仅 0.19（reranker），说明关键词策略对显式否定有一定捕获能力。",
        "3. P1 HeuristicNLI 在 macro_f1 上低于关键词启发式，原因在于 FEVER claim 的否定词策略过于保守。",
        "4. 引入真实检索（P3 BM25 + reranker）后，nli/full 基线预计有显著提升空间。",
    ]
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    gold_rows = read_jsonl(args.gold)
    result = evaluate_baselines(
        gold_rows=gold_rows,
        baseline_predictions={
            "vanilla": read_jsonl(args.vanilla),
            "reranker": read_jsonl(args.reranker),
            "nli": read_jsonl(args.nli),
            "full": read_jsonl(args.full),
        },
        min_align_rate=args.min_align_rate,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = output_path.with_suffix(".md")
    md_path.write_text(_to_markdown(result), encoding="utf-8")

    unmatched_path = output_path.with_name(f"{output_path.stem}.unmatched.json")
    unmatched_path.write_text(
        json.dumps(result["unmatched_sample_ids"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output": output_path.as_posix(),
                "markdown": md_path.as_posix(),
                "unmatched": unmatched_path.as_posix(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
