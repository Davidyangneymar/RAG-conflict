from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

FNC1_LABEL_ORDER = ["agree", "disagree", "discuss", "unrelated"]
THREE_WAY_LABEL_ORDER = ["support", "oppose", "neutral"]

TEXT_MODE_CHOICES = [
    "headline_body",
    "headline_body_first3sent",
    "headline_body_first512tok",
]

TRAINABLE_BACKEND_CHOICES = [
    "tfidf_linear_svm",
    "tfidf_logreg",
]

BACKEND_CHOICES = [
    "tfidf_linear_svm",
    "tfidf_logreg",
    "bert",
    "roberta",
]

SCHEME_A_NAME = "scheme_a_filter_unrelated"
SCHEME_B_NAME = "scheme_b_unrelated_to_neutral"

THREE_WAY_SCHEMES = {
    SCHEME_A_NAME: {
        "mapping": {
            "agree": "support",
            "disagree": "oppose",
            "discuss": "neutral",
            "unrelated": None,
        },
        "drop_unmapped": True,
    },
    SCHEME_B_NAME: {
        "mapping": {
            "agree": "support",
            "disagree": "oppose",
            "discuss": "neutral",
            "unrelated": "neutral",
        },
        "drop_unmapped": False,
    },
}

THREE_WAY_SCHEME_COLUMNS = {
    SCHEME_A_NAME: ("gold_label_3way_A", "pred_label_3way_A"),
    SCHEME_B_NAME: ("gold_label_3way_B", "pred_label_3way_B"),
}

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class StanceTrainingConfig:
    train_csv: str = "train_processed.csv"
    val_csv: str = "val_processed.csv"
    output_dir: str = "outputs/fnc1_baseline"
    backend_name: str = "tfidf_linear_svm"
    text_mode: str = "headline_body_first3sent"
    text_separator: str = " [SEP] "
    random_state: int = 42
    max_features: int = 120000
    min_df: int = 2
    ngram_min: int = 1
    ngram_max: int = 2


class BaseStanceBackend(ABC):
    @abstractmethod
    def fit(self, x_train: pd.Series, y_train: pd.Series) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: pd.Series) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def decision_score_for_predictions(self, x: pd.Series, y_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def save(self, model_path: Path) -> None:
        raise NotImplementedError


class SklearnTfidfBackend(BaseStanceBackend):
    def __init__(
        self,
        model_name: str,
        random_state: int,
        max_features: int,
        min_df: int,
        ngram_min: int,
        ngram_max: int,
        pipeline: Pipeline | None = None,
    ) -> None:
        self.model_name = model_name
        self.random_state = random_state
        self.max_features = max_features
        self.min_df = min_df
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.pipeline = pipeline if pipeline is not None else self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        if self.model_name == "tfidf_linear_svm":
            clf = LinearSVC(random_state=self.random_state)
        elif self.model_name == "tfidf_logreg":
            clf = LogisticRegression(
                max_iter=3000,
                random_state=self.random_state,
                solver="lbfgs",
            )
        else:
            raise ValueError(f"Unsupported sklearn backend: {self.model_name}")

        return Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode",
                        ngram_range=(self.ngram_min, self.ngram_max),
                        max_features=self.max_features,
                        min_df=self.min_df,
                        sublinear_tf=True,
                    ),
                ),
                ("clf", clf),
            ]
        )

    def fit(self, x_train: pd.Series, y_train: pd.Series) -> None:
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        self.pipeline.fit(x_train, y_train, clf__sample_weight=sample_weight)

    def predict(self, x: pd.Series) -> np.ndarray:
        return self.pipeline.predict(x)

    def decision_score_for_predictions(self, x: pd.Series, y_pred: np.ndarray) -> np.ndarray:
        clf = self.pipeline.named_steps["clf"]
        classes = list(clf.classes_)
        label_to_idx = {label: idx for idx, label in enumerate(classes)}

        if hasattr(self.pipeline, "decision_function"):
            decision = self.pipeline.decision_function(x)
            if np.ndim(decision) == 1:
                positive_label = classes[1]
                return np.asarray(
                    [float(score) if pred == positive_label else float(-score) for pred, score in zip(y_pred, decision)],
                    dtype=float,
                )
            return np.asarray(
                [float(decision[i, label_to_idx[pred]]) for i, pred in enumerate(y_pred)],
                dtype=float,
            )

        if hasattr(self.pipeline, "predict_proba"):
            prob = self.pipeline.predict_proba(x)
            return np.asarray(
                [float(prob[i, label_to_idx[pred]]) for i, pred in enumerate(y_pred)],
                dtype=float,
            )

        raise RuntimeError("Backend does not provide decision_function or predict_proba.")

    def save(self, model_path: Path) -> None:
        payload = {
            "backend_name": self.model_name,
            "random_state": self.random_state,
            "max_features": self.max_features,
            "min_df": self.min_df,
            "ngram_min": self.ngram_min,
            "ngram_max": self.ngram_max,
            "pipeline": self.pipeline,
        }
        joblib.dump(payload, model_path)

    @classmethod
    def load(cls, model_path: Path) -> "SklearnTfidfBackend":
        payload = joblib.load(model_path)
        return cls(
            model_name=payload["backend_name"],
            random_state=payload["random_state"],
            max_features=payload["max_features"],
            min_df=payload["min_df"],
            ngram_min=payload["ngram_min"],
            ngram_max=payload["ngram_max"],
            pipeline=payload["pipeline"],
        )


class TransformerBackendPlaceholder(BaseStanceBackend):
    """Reserved extension point for BERT/RoBERTa stance backends."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def fit(self, x_train: pd.Series, y_train: pd.Series) -> None:
        raise NotImplementedError(
            f"{self.model_name} backend is a placeholder. "
            "Replace this class with a Hugging Face Trainer/PyTorch implementation."
        )

    def predict(self, x: pd.Series) -> np.ndarray:
        raise NotImplementedError(
            f"{self.model_name} backend is a placeholder. "
            "Implement predict() when adding transformer inference."
        )

    def decision_score_for_predictions(self, x: pd.Series, y_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            f"{self.model_name} backend is a placeholder. "
            "Implement decision_score_for_predictions() for transformer logits/probabilities."
        )

    def save(self, model_path: Path) -> None:
        raise NotImplementedError(
            f"{self.model_name} backend is a placeholder and cannot be serialized yet."
        )


class StanceInferenceService:
    def __init__(self, backend: BaseStanceBackend, text_mode: str, text_separator: str) -> None:
        self.backend = backend
        self.text_mode = text_mode
        self.text_separator = text_separator

    def predict(self, claim: str, evidence_text: str) -> dict[str, float | str]:
        sample_df = pd.DataFrame({"headline": [claim], "body": [evidence_text]})
        model_input = build_model_input_text(sample_df, text_mode=self.text_mode, text_separator=self.text_separator)
        pred = self.backend.predict(model_input)
        decision_score = self.backend.decision_score_for_predictions(model_input, pred)
        return {
            "pred_label": str(pred[0]),
            "decision_score": float(decision_score[0]),
        }

    def predict_batch(self, headline: Iterable[str], evidence_text: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
        sample_df = pd.DataFrame(
            {
                "headline": pd.Series(list(headline)).fillna("").astype(str),
                "body": pd.Series(list(evidence_text)).fillna("").astype(str),
            }
        )
        model_input = build_model_input_text(sample_df, text_mode=self.text_mode, text_separator=self.text_separator)
        pred = self.backend.predict(model_input)
        decision_score = self.backend.decision_score_for_predictions(model_input, pred)
        return pred, decision_score


def validate_fnc1_dataframe(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    required_cols = {"headline", "body", "label"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"{df_name} is missing required columns: {sorted(missing_cols)}")

    out = df.copy()
    out["headline"] = out["headline"].fillna("").astype(str)
    out["body"] = out["body"].fillna("").astype(str)
    out["label"] = out["label"].fillna("").astype(str).str.strip().str.lower()
    out = out[out["label"].isin(FNC1_LABEL_ORDER)].reset_index(drop=True)
    return out


def load_datasets(train_csv: str, val_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = Path(train_csv)
    val_path = Path(val_csv)

    if not train_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation CSV not found: {val_path}")

    train_df = validate_fnc1_dataframe(pd.read_csv(train_path), "train_df")
    val_df = validate_fnc1_dataframe(pd.read_csv(val_path), "val_df")
    return train_df, val_df


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


def build_body_text(df: pd.DataFrame, text_mode: str) -> pd.Series:
    if text_mode == "headline_body":
        return df["body"]
    if text_mode == "headline_body_first3sent":
        return df["body"].map(lambda x: first_n_sentences(x, n=3))
    if text_mode == "headline_body_first512tok":
        return df["body"].map(lambda x: first_n_tokens(x, n=512))
    raise ValueError(f"Unsupported text_mode: {text_mode}")


def build_model_input_text(df: pd.DataFrame, text_mode: str, text_separator: str) -> pd.Series:
    body_text = build_body_text(df, text_mode=text_mode)
    return "headline: " + df["headline"] + text_separator + "body: " + body_text


def create_backend(config: StanceTrainingConfig) -> BaseStanceBackend:
    if config.backend_name in {"tfidf_linear_svm", "tfidf_logreg"}:
        return SklearnTfidfBackend(
            model_name=config.backend_name,
            random_state=config.random_state,
            max_features=config.max_features,
            min_df=config.min_df,
            ngram_min=config.ngram_min,
            ngram_max=config.ngram_max,
        )
    if config.backend_name in {"bert", "roberta"}:
        return TransformerBackendPlaceholder(model_name=config.backend_name)
    raise ValueError(f"Unsupported backend_name: {config.backend_name}")


def train_model(
    train_df: pd.DataFrame,
    config: StanceTrainingConfig,
    backend: BaseStanceBackend,
) -> None:
    x_train = build_model_input_text(
        train_df,
        text_mode=config.text_mode,
        text_separator=config.text_separator,
    )
    y_train = train_df["label"]
    backend.fit(x_train, y_train)


def predict_model(
    val_df: pd.DataFrame,
    config: StanceTrainingConfig,
    inference_service: StanceInferenceService,
) -> pd.DataFrame:
    y_pred, decision_score = inference_service.predict_batch(
        headline=val_df["headline"],
        evidence_text=val_df["body"],
    )

    sample_id = (
        val_df["sample_id"].copy()
        if "sample_id" in val_df.columns
        else pd.Series(np.arange(1, len(val_df) + 1), index=val_df.index)
    )

    pred_df = pd.DataFrame(
        {
            "sample_id": sample_id,
            "headline": val_df["headline"],
            "body_first3sent": val_df["body"].map(lambda x: first_n_sentences(x, n=3)),
            "gold_label_4way": val_df["label"],
            "pred_label_4way": y_pred,
            "decision_score": decision_score,
        }
    )
    return pred_df.reset_index(drop=True)


def append_three_way_mappings(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()

    mapping_a = THREE_WAY_SCHEMES[SCHEME_A_NAME]["mapping"]
    mapping_b = THREE_WAY_SCHEMES[SCHEME_B_NAME]["mapping"]

    out["gold_label_3way_A"] = out["gold_label_4way"].map(mapping_a)
    out["pred_label_3way_A"] = out["pred_label_4way"].map(mapping_a)
    out["gold_label_3way_B"] = out["gold_label_4way"].map(mapping_b)
    out["pred_label_3way_B"] = out["pred_label_4way"].map(mapping_b)

    return out


def prepare_three_way_eval_frame(
    pred_df: pd.DataFrame,
    scheme_name: str,
) -> tuple[pd.DataFrame, str, str, int]:
    if scheme_name not in THREE_WAY_SCHEMES:
        raise ValueError(
            f"Unsupported scheme_name: {scheme_name}. Valid: {sorted(THREE_WAY_SCHEMES.keys())}"
        )

    gold_col, pred_col = THREE_WAY_SCHEME_COLUMNS[scheme_name]
    scheme_cfg = THREE_WAY_SCHEMES[scheme_name]

    out = pred_df.copy()
    dropped_rows = 0
    if scheme_cfg["drop_unmapped"]:
        keep_mask = out[gold_col].notna() & out[pred_col].notna()
        dropped_rows = int((~keep_mask).sum())
        out = out[keep_mask].reset_index(drop=True)

    return out, gold_col, pred_col, dropped_rows


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    label_order: list[str],
) -> tuple[dict, pd.DataFrame, pd.DataFrame, str]:
    y_true = y_true.fillna("").astype(str)
    y_pred = y_pred.fillna("").astype(str)

    if len(y_true) == 0:
        summary = {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "label_order": label_order,
            "num_samples": 0,
        }
        per_class_df = pd.DataFrame(
            [
                {
                    "label": label,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "support": 0,
                }
                for label in label_order
            ]
        )
        cm_df = pd.DataFrame(
            np.zeros((len(label_order), len(label_order)), dtype=int),
            index=label_order,
            columns=label_order,
        )
        cm_df.index.name = "true_label"
        cm_df.columns.name = "pred_label"
        report_text = "No samples available after mapping/filtering."
        return summary, per_class_df, cm_df, report_text

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=label_order, average="macro", zero_division=0)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=label_order,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=label_order,
        digits=4,
        zero_division=0,
    )

    per_class_df = pd.DataFrame(
        [
            {
                "label": label,
                "precision": float(report_dict.get(label, {}).get("precision", 0.0)),
                "recall": float(report_dict.get(label, {}).get("recall", 0.0)),
                "f1": float(report_dict.get(label, {}).get("f1-score", 0.0)),
                "support": int(report_dict.get(label, {}).get("support", 0)),
            }
            for label in label_order
        ]
    )

    cm = confusion_matrix(y_true, y_pred, labels=label_order)
    cm_df = pd.DataFrame(cm, index=label_order, columns=label_order)
    cm_df.index.name = "true_label"
    cm_df.columns.name = "pred_label"

    summary = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "label_order": label_order,
        "num_samples": int(len(y_true)),
        "macro_avg": report_dict.get("macro avg", {}),
        "weighted_avg": report_dict.get("weighted avg", {}),
    }
    return summary, per_class_df, cm_df, report_text


def save_evaluation(
    output_dir: Path,
    summary: dict,
    per_class_df: pd.DataFrame,
    cm_df: pd.DataFrame,
    report_text: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    per_class_df.to_csv(output_dir / "per_class_metrics.csv", index=False)
    cm_df.to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8")
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")


def save_prediction_csv(pred_df: pd.DataFrame, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    required_cols = [
        "sample_id",
        "headline",
        "body_first3sent",
        "gold_label_4way",
        "pred_label_4way",
        "gold_label_3way_A",
        "pred_label_3way_A",
        "gold_label_3way_B",
        "pred_label_3way_B",
        "decision_score",
    ]
    pred_df[required_cols].to_csv(output_csv, index=False)


def save_three_way_prediction_csv(
    pred_df: pd.DataFrame,
    output_csv: Path,
    gold_col: str,
    pred_col: str,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out = pred_df[
        [
            "sample_id",
            "headline",
            "body_first3sent",
            "gold_label_4way",
            "pred_label_4way",
            gold_col,
            pred_col,
            "decision_score",
        ]
    ].rename(
        columns={
            gold_col: "gold_label_3way",
            pred_col: "pred_label_3way",
        }
    )
    out.to_csv(output_csv, index=False)


def run_training_pipeline(config: StanceTrainingConfig) -> dict:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_datasets(config.train_csv, config.val_csv)

    backend = create_backend(config)
    train_model(train_df=train_df, config=config, backend=backend)

    inference_service = StanceInferenceService(
        backend=backend,
        text_mode=config.text_mode,
        text_separator=config.text_separator,
    )

    val_pred_df = predict_model(
        val_df=val_df,
        config=config,
        inference_service=inference_service,
    )
    val_pred_df = append_three_way_mappings(val_pred_df)

    # 4-way evaluation (original FNC-1 setting)
    summary_4way, per_class_4way, cm_4way, report_4way = evaluate_predictions(
        y_true=val_pred_df["gold_label_4way"],
        y_pred=val_pred_df["pred_label_4way"],
        label_order=FNC1_LABEL_ORDER,
    )
    save_evaluation(
        output_dir=output_dir / "four_way",
        summary=summary_4way,
        per_class_df=per_class_4way,
        cm_df=cm_4way,
        report_text=report_4way,
    )

    # Keep compatibility with previous single CSV output.
    val_pred_df.rename(
        columns={
            "gold_label_4way": "label",
            "pred_label_4way": "pred_label",
        }
    ).to_csv(output_dir / "val_predictions.csv", index=False)

    # Save the consolidated prediction CSV with both 3-way mappings.
    save_prediction_csv(
        pred_df=val_pred_df,
        output_csv=output_dir / "val_predictions_with_3way.csv",
    )

    three_way_summary: dict[str, dict] = {}
    for scheme_name in THREE_WAY_SCHEMES:
        eval_df, gold_col, pred_col, dropped_rows = prepare_three_way_eval_frame(
            val_pred_df,
            scheme_name=scheme_name,
        )

        summary_3way, per_class_3way, cm_3way, report_3way = evaluate_predictions(
            y_true=eval_df[gold_col],
            y_pred=eval_df[pred_col],
            label_order=THREE_WAY_LABEL_ORDER,
        )
        summary_3way["scheme_name"] = scheme_name
        summary_3way["dropped_rows"] = dropped_rows
        summary_3way["gold_column"] = gold_col
        summary_3way["pred_column"] = pred_col

        scheme_dir = output_dir / "three_way" / scheme_name
        save_evaluation(
            output_dir=scheme_dir,
            summary=summary_3way,
            per_class_df=per_class_3way,
            cm_df=cm_3way,
            report_text=report_3way,
        )
        save_three_way_prediction_csv(
            pred_df=eval_df,
            output_csv=scheme_dir / "val_predictions_3way.csv",
            gold_col=gold_col,
            pred_col=pred_col,
        )

        three_way_summary[scheme_name] = summary_3way

    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    backend.save(model_dir / "model.joblib")

    (output_dir / "run_config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "four_way": summary_4way,
        "three_way": three_way_summary,
        "artifacts": {
            "output_dir": str(output_dir.resolve()),
            "model_path": str((model_dir / "model.joblib").resolve()),
            "prediction_csv": str((output_dir / "val_predictions_with_3way.csv").resolve()),
        },
    }


def build_inference_service_from_artifacts(output_dir: str | Path) -> StanceInferenceService:
    output_dir = Path(output_dir)
    config_path = output_dir / "run_config.json"
    model_path = output_dir / "model" / "model.joblib"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    config = StanceTrainingConfig(**json.loads(config_path.read_text(encoding="utf-8")))

    if config.backend_name in {"tfidf_linear_svm", "tfidf_logreg"}:
        backend = SklearnTfidfBackend.load(model_path)
    elif config.backend_name in {"bert", "roberta"}:
        raise NotImplementedError(
            "Transformer backend loading is reserved for future implementation."
        )
    else:
        raise ValueError(f"Unsupported backend_name in config: {config.backend_name}")

    return StanceInferenceService(
        backend=backend,
        text_mode=config.text_mode,
        text_separator=config.text_separator,
    )


_INFERENCE_SERVICE_CACHE: dict[str, StanceInferenceService] = {}


def predict_stance(
    claim: str,
    evidence_text: str,
    output_dir: str | Path = "outputs/fnc1_baseline",
) -> dict[str, float | str]:
    cache_key = str(Path(output_dir).resolve())
    if cache_key not in _INFERENCE_SERVICE_CACHE:
        _INFERENCE_SERVICE_CACHE[cache_key] = build_inference_service_from_artifacts(output_dir)
    return _INFERENCE_SERVICE_CACHE[cache_key].predict(claim=claim, evidence_text=evidence_text)
