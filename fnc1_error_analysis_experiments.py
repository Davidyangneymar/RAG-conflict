from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


FNC1_LABEL_ORDER = ["agree", "disagree", "discuss", "unrelated"]

MODEL_CHOICES = ["logreg", "linear_svm"]
TEXT_MODE_CHOICES = [
    "headline_body",
    "headline_body_first3sent",
    "headline_body_first512tok",
]

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "FNC-1 error analysis + improvement experiments using TF-IDF with "
            "Logistic Regression and Linear SVM."
        )
    )
    parser.add_argument("--train_csv", type=str, default="train_processed.csv")
    parser.add_argument("--val_csv", type=str, default="val_processed.csv")
    parser.add_argument(
        "--pred_csv",
        type=str,
        default="outputs/fnc1_baseline/val_predictions.csv",
        help="Existing validation predictions for error analysis.",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/fnc1_error_analysis")
    parser.add_argument("--models", type=str, default=",".join(MODEL_CHOICES))
    parser.add_argument("--text_modes", type=str, default=",".join(TEXT_MODE_CHOICES))
    parser.add_argument("--text_separator", type=str, default=" [SEP] ")
    parser.add_argument("--max_features", type=int, default=120000)
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--ngram_min", type=int, default=1)
    parser.add_argument("--ngram_max", type=int, default=2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--body_preview_chars", type=int, default=400)
    parser.add_argument(
        "--skip_existing_analysis",
        action="store_true",
        help="Skip analyzing existing prediction file.",
    )
    return parser.parse_args()


def validate_dataset(df: pd.DataFrame, name: str) -> pd.DataFrame:
    required_cols = {"headline", "body", "label"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"{name} missing required columns: {sorted(missing_cols)}")

    out = df.copy()
    out["headline"] = out["headline"].fillna("").astype(str)
    out["body"] = out["body"].fillna("").astype(str)
    out["label"] = out["label"].fillna("").astype(str).str.strip().str.lower()
    out = out[out["label"].isin(FNC1_LABEL_ORDER)].reset_index(drop=True)
    return out


def pick_column(df: pd.DataFrame, candidates: list[str], kind: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Cannot find {kind} column in prediction file. Candidates: {candidates}")


def validate_prediction_df(df: pd.DataFrame) -> pd.DataFrame:
    true_col = pick_column(df, ["label", "true_label", "gold_label", "y_true"], "true label")
    pred_col = pick_column(df, ["pred_label", "prediction", "pred", "y_pred"], "pred label")

    out = df.copy()
    out[true_col] = out[true_col].fillna("").astype(str).str.strip().str.lower()
    out[pred_col] = out[pred_col].fillna("").astype(str).str.strip().str.lower()

    out = out[out[true_col].isin(FNC1_LABEL_ORDER)].reset_index(drop=True)

    unknown_pred = sorted(set(out[pred_col]) - set(FNC1_LABEL_ORDER))
    if unknown_pred:
        raise ValueError(
            "Prediction file contains labels not in FNC1_LABEL_ORDER: "
            f"{unknown_pred}"
        )

    if true_col != "label":
        out = out.rename(columns={true_col: "label"})
    if pred_col != "pred_label":
        out = out.rename(columns={pred_col: "pred_label"})

    return out


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=FNC1_LABEL_ORDER, average="macro", zero_division=0)
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=FNC1_LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )

    per_class_rows = []
    summary = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
    }
    for label in FNC1_LABEL_ORDER:
        f1_value = float(report_dict.get(label, {}).get("f1-score", 0.0))
        summary[f"f1_{label}"] = f1_value
        per_class_rows.append(
            {
                "label": label,
                "precision": float(report_dict.get(label, {}).get("precision", 0.0)),
                "recall": float(report_dict.get(label, {}).get("recall", 0.0)),
                "f1": f1_value,
                "support": int(report_dict.get(label, {}).get("support", 0)),
            }
        )
    per_class_df = pd.DataFrame(per_class_rows)

    cm = confusion_matrix(y_true, y_pred, labels=FNC1_LABEL_ORDER)
    cm_df = pd.DataFrame(cm, index=FNC1_LABEL_ORDER, columns=FNC1_LABEL_ORDER)
    cm_df.index.name = "true_label"
    cm_df.columns.name = "pred_label"

    return summary, per_class_df, cm_df


def build_misclassified_table(
    pred_df: pd.DataFrame,
    max_body_chars: int,
) -> pd.DataFrame:
    out = pred_df[pred_df["label"] != pred_df["pred_label"]].copy()
    out["error_type"] = out["label"] + " -> " + out["pred_label"]

    if "body" in out.columns:
        out["body_preview"] = out["body"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True)
        out["body_preview"] = out["body_preview"].str.slice(0, max_body_chars)
        out = out.drop(columns=["body"])

    preferred_cols = [
        "headline",
        "body_preview",
        "label",
        "pred_label",
        "error_type",
        "is_correct",
        "pred_confidence",
    ]
    existing_preferred = [c for c in preferred_cols if c in out.columns]
    tail_cols = [c for c in out.columns if c not in existing_preferred]
    return out[existing_preferred + tail_cols]


def first_n_sentences(text: str, n: int) -> str:
    normalized = " ".join(str(text).split())
    if not normalized:
        return ""

    parts = [part.strip() for part in SENT_SPLIT_RE.split(normalized) if part.strip()]
    if not parts:
        return normalized
    return " ".join(parts[:n])


def first_n_tokens(text: str, n: int) -> str:
    tokens = str(text).split()
    if not tokens:
        return ""
    return " ".join(tokens[:n])


def build_text_input(df: pd.DataFrame, text_mode: str, text_separator: str) -> pd.Series:
    if text_mode == "headline_body":
        body_part = df["body"]
    elif text_mode == "headline_body_first3sent":
        body_part = df["body"].map(lambda x: first_n_sentences(x, n=3))
    elif text_mode == "headline_body_first512tok":
        body_part = df["body"].map(lambda x: first_n_tokens(x, n=512))
    else:
        raise ValueError(f"Unsupported text_mode: {text_mode}")

    return "headline: " + df["headline"] + text_separator + "body: " + body_part


def build_pipeline(
    model_name: str,
    random_state: int,
    max_features: int,
    min_df: int,
    ngram_min: int,
    ngram_max: int,
) -> Pipeline:
    if model_name == "logreg":
        clf = LogisticRegression(
            max_iter=3000,
            random_state=random_state,
            solver="lbfgs",
            class_weight="balanced",
        )
    elif model_name == "linear_svm":
        clf = LinearSVC(
            random_state=random_state,
            class_weight="balanced",
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(ngram_min, ngram_max),
                    max_features=max_features,
                    min_df=min_df,
                    sublinear_tf=True,
                ),
            ),
            ("clf", clf),
        ]
    )


def analyze_existing_predictions(pred_csv: Path, output_dir: Path, max_body_chars: int) -> dict:
    pred_df = validate_prediction_df(pd.read_csv(pred_csv))
    summary, per_class_df, cm_df = compute_metrics(pred_df["label"], pred_df["pred_label"])

    pred_df = pred_df.copy()
    pred_df["is_correct"] = (pred_df["label"] == pred_df["pred_label"]).astype(int)
    mis_df = build_misclassified_table(pred_df, max_body_chars=max_body_chars)

    output_dir.mkdir(parents=True, exist_ok=True)
    cm_df.to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8")
    per_class_df.to_csv(output_dir / "per_class_metrics.csv", index=False)
    mis_df.to_csv(output_dir / "misclassified_samples.csv", index=False)
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return summary


def run_single_experiment(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str,
    text_mode: str,
    output_dir: Path,
    text_separator: str,
    random_state: int,
    max_features: int,
    min_df: int,
    ngram_min: int,
    ngram_max: int,
    max_body_chars: int,
) -> dict:
    x_train = build_text_input(train_df, text_mode=text_mode, text_separator=text_separator)
    x_val = build_text_input(val_df, text_mode=text_mode, text_separator=text_separator)

    y_train = train_df["label"]
    y_val = val_df["label"]

    model = build_pipeline(
        model_name=model_name,
        random_state=random_state,
        max_features=max_features,
        min_df=min_df,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_val)
    pred_df = val_df[["headline", "body", "label"]].copy()
    pred_df["pred_label"] = y_pred
    pred_df["is_correct"] = (pred_df["label"] == pred_df["pred_label"]).astype(int)

    summary, per_class_df, cm_df = compute_metrics(y_val, y_pred)
    mis_df = build_misclassified_table(pred_df, max_body_chars=max_body_chars)

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_dir / "val_predictions.csv", index=False)
    cm_df.to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8")
    per_class_df.to_csv(output_dir / "per_class_metrics.csv", index=False)
    mis_df.to_csv(output_dir / "misclassified_samples.csv", index=False)
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    result = {
        "model_name": model_name,
        "text_mode": text_mode,
        "accuracy": summary["accuracy"],
        "macro_f1": summary["macro_f1"],
    }
    for label in FNC1_LABEL_ORDER:
        result[f"f1_{label}"] = summary[f"f1_{label}"]
    return result


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = parse_csv_list(args.models)
    text_modes = parse_csv_list(args.text_modes)

    invalid_models = sorted(set(models) - set(MODEL_CHOICES))
    if invalid_models:
        raise ValueError(f"Unsupported models: {invalid_models}. Valid: {MODEL_CHOICES}")

    invalid_text_modes = sorted(set(text_modes) - set(TEXT_MODE_CHOICES))
    if invalid_text_modes:
        raise ValueError(
            f"Unsupported text_modes: {invalid_text_modes}. Valid: {TEXT_MODE_CHOICES}"
        )

    if not args.skip_existing_analysis:
        pred_path = Path(args.pred_csv)
        if pred_path.exists():
            analyze_existing_predictions(
                pred_csv=pred_path,
                output_dir=output_dir / "existing_predictions",
                max_body_chars=args.body_preview_chars,
            )
        else:
            print(f"[WARN] Existing prediction file not found, skipped: {pred_path}")

    train_path = Path(args.train_csv)
    val_path = Path(args.val_csv)
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation file not found: {val_path}")

    train_df = validate_dataset(pd.read_csv(train_path), "train_df")
    val_df = validate_dataset(pd.read_csv(val_path), "val_df")

    results: list[dict] = []
    experiments_root = output_dir / "experiments"
    for model_name in models:
        for text_mode in text_modes:
            exp_name = f"{model_name}__{text_mode}"
            print(f"[RUN] {exp_name}")
            result = run_single_experiment(
                train_df=train_df,
                val_df=val_df,
                model_name=model_name,
                text_mode=text_mode,
                output_dir=experiments_root / exp_name,
                text_separator=args.text_separator,
                random_state=args.random_state,
                max_features=args.max_features,
                min_df=args.min_df,
                ngram_min=args.ngram_min,
                ngram_max=args.ngram_max,
                max_body_chars=args.body_preview_chars,
            )
            result["experiment"] = exp_name
            results.append(result)

    summary_df = pd.DataFrame(results)
    summary_df = summary_df[
        [
            "experiment",
            "model_name",
            "text_mode",
            "accuracy",
            "macro_f1",
            "f1_agree",
            "f1_disagree",
            "f1_discuss",
            "f1_unrelated",
        ]
    ].sort_values(["macro_f1", "accuracy"], ascending=False)
    summary_df.to_csv(output_dir / "experiment_summary.csv", index=False)

    print("===== FNC-1 Error Analysis + Experiments Finished =====")
    print(f"saved to: {output_dir.resolve()}")
    print("\nTop experiments:")
    print(summary_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()