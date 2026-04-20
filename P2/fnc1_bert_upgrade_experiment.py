from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fnc1_stance_module import (
    FNC1_LABEL_ORDER,
    SCHEME_A_NAME,
    THREE_WAY_LABEL_ORDER,
    THREE_WAY_SCHEMES,
    evaluate_predictions,
    first_n_sentences,
    save_evaluation,
)


@dataclass
class BertExperimentConfig:
    train_csv: str = "train_processed.csv"
    test_csv: str = "val_processed.csv"
    output_dir: str = "outputs/fnc1_bert_upgrade"
    svm_baseline_dir: str = "outputs/fnc1_baseline_frozen"
    model_name: str = "bert-base-uncased"
    text_mode: str = "headline_body_first3sent"
    validation_size: float = 0.1
    random_state: int = 42
    max_length: int = 256
    train_batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_train_epochs: int = 3
    max_train_samples: int = 0
    max_val_samples: int = 0
    max_test_samples: int = 0
    log_every_steps: int = 20
    disable_tqdm: bool = False


class FNC1SentencePairDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int,
        label_to_id: dict[str, int],
        with_labels: bool,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_id = label_to_id
        self.with_labels = with_labels

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        encoded = self.tokenizer(
            str(row["headline"]),
            str(row["body_first3sent"]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["sample_id"] = torch.tensor(int(row["sample_id"]), dtype=torch.long)

        if self.with_labels:
            item["labels"] = torch.tensor(self.label_to_id[str(row["label"])], dtype=torch.long)

        return item


class BertStanceInferenceService:
    def __init__(
        self,
        model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        max_length: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    @torch.no_grad()
    def predict(self, claim: str, evidence_text: str) -> dict[str, float | str]:
        body_first3sent = first_n_sentences(evidence_text, n=3)
        encoded = self.tokenizer(
            claim,
            body_first3sent,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        self.model.eval()
        logits = self.model(**encoded).logits.detach().cpu().numpy()
        probs = softmax_numpy(logits)
        pred_idx = int(np.argmax(probs[0]))

        pred_label = self.model.config.id2label.get(pred_idx, FNC1_LABEL_ORDER[pred_idx])
        decision_score = float(probs[0, pred_idx])
        return {
            "pred_label": pred_label,
            "decision_score": decision_score,
        }


def parse_args() -> BertExperimentConfig:
    parser = argparse.ArgumentParser(
        description=(
            "BERT upgrade experiment for FNC-1 stance classification using sentence-pair input "
            "(headline, body_first3sent), with strict comparison to frozen SVM baseline."
        )
    )
    parser.add_argument("--train_csv", type=str, default="train_processed.csv")
    parser.add_argument("--test_csv", type=str, default="val_processed.csv")
    parser.add_argument("--output_dir", type=str, default="outputs/fnc1_bert_upgrade")
    parser.add_argument("--svm_baseline_dir", type=str, default="outputs/fnc1_baseline_frozen")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument(
        "--text_mode",
        type=str,
        default="headline_body_first3sent",
        choices=["headline_body_first3sent"],
        help="Kept fixed to maintain strict comparability with current best SVM baseline.",
    )
    parser.add_argument("--validation_size", type=float, default=0.1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=0,
        help="Optional cap for smoke tests; 0 means full training set.",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=0,
        help="Optional cap for smoke tests; 0 means full validation split.",
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=0,
        help="Optional cap for smoke tests; 0 means full test split.",
    )
    parser.add_argument(
        "--log_every_steps",
        type=int,
        default=20,
        help="Print step-level training/eval logs every N steps (0 disables periodic prints).",
    )
    parser.add_argument(
        "--disable_tqdm",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return BertExperimentConfig(**vars(parser.parse_args()))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
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


def ensure_sample_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sample_id" not in out.columns:
        out["sample_id"] = np.arange(1, len(out) + 1)
    return out


def build_body_first3sent(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["body_first3sent"] = out["body"].map(lambda x: first_n_sentences(x, n=3))
    return out


def maybe_limit_samples(df: pd.DataFrame, max_samples: int, random_state: int) -> pd.DataFrame:
    if max_samples <= 0 or len(df) <= max_samples:
        return df.reset_index(drop=True)
    return df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)


def load_train_val_test(config: BertExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = Path(config.train_csv)
    test_path = Path(config.test_csv)

    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train_df = validate_dataframe(pd.read_csv(train_path), "train_df")
    test_df = validate_dataframe(pd.read_csv(test_path), "test_df")

    train_df = ensure_sample_id(train_df)
    test_df = ensure_sample_id(test_df)

    train_df = build_body_first3sent(train_df)
    test_df = build_body_first3sent(test_df)

    if not 0 < config.validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")

    train_split, val_split = train_test_split(
        train_df,
        test_size=config.validation_size,
        random_state=config.random_state,
        stratify=train_df["label"],
    )

    train_split = maybe_limit_samples(train_split, config.max_train_samples, config.random_state)
    val_split = maybe_limit_samples(val_split, config.max_val_samples, config.random_state)
    test_df = maybe_limit_samples(test_df, config.max_test_samples, config.random_state)

    return train_split.reset_index(drop=True), val_split.reset_index(drop=True), test_df.reset_index(drop=True)


def build_dataloader(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    max_length: int,
    label_to_id: dict[str, int],
    batch_size: int,
    shuffle: bool,
    with_labels: bool,
) -> DataLoader:
    dataset = FNC1SentencePairDataset(
        df=df,
        tokenizer=tokenizer,
        max_length=max_length,
        label_to_id=label_to_id,
        with_labels=with_labels,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def move_model_inputs_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    model_inputs: dict[str, torch.Tensor] = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
    }
    if "token_type_ids" in batch:
        model_inputs["token_type_ids"] = batch["token_type_ids"].to(device)
    if "labels" in batch:
        model_inputs["labels"] = batch["labels"].to(device)
    return model_inputs


def run_training_epoch(
    model: AutoModelForSequenceClassification,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    epoch_idx: int,
    num_epochs: int,
    log_every_steps: int,
    disable_tqdm: bool,
) -> float:
    model.train()
    total_loss = 0.0
    num_steps = len(dataloader)

    progress_bar = tqdm(
        enumerate(dataloader, start=1),
        total=num_steps,
        desc=f"train {epoch_idx}/{num_epochs}",
        unit="step",
        disable=disable_tqdm,
    )

    for step, batch in progress_bar:
        optimizer.zero_grad(set_to_none=True)
        model_inputs = move_model_inputs_to_device(batch, device)
        outputs = model(**model_inputs)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        step_loss = float(loss.detach().cpu().item())
        total_loss += step_loss

        avg_loss = total_loss / max(step, 1)
        lr = float(optimizer.param_groups[0]["lr"])
        progress_bar.set_postfix(loss=f"{step_loss:.4f}", avg_loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

        if log_every_steps > 0 and (step % log_every_steps == 0 or step == num_steps):
            print(
                f"[train][epoch {epoch_idx}/{num_epochs}] "
                f"step {step}/{num_steps} loss={step_loss:.4f} avg_loss={avg_loss:.4f} lr={lr:.2e}",
                flush=True,
            )

    if num_steps == 0:
        return 0.0
    return total_loss / num_steps


@torch.no_grad()
def predict_logits(
    model: AutoModelForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device,
    desc: str,
    disable_tqdm: bool,
    log_every_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_sample_ids: list[np.ndarray] = []
    num_steps = len(dataloader)

    progress_bar = tqdm(
        enumerate(dataloader, start=1),
        total=num_steps,
        desc=desc,
        unit="step",
        disable=disable_tqdm,
    )

    for step, batch in progress_bar:
        all_sample_ids.append(batch["sample_id"].detach().cpu().numpy())
        if "labels" in batch:
            all_labels.append(batch["labels"].detach().cpu().numpy())

        model_inputs = move_model_inputs_to_device(batch, device)
        if "labels" in model_inputs:
            model_inputs = {k: v for k, v in model_inputs.items() if k != "labels"}

        logits = model(**model_inputs).logits.detach().cpu().numpy()
        all_logits.append(logits)

        if log_every_steps > 0 and (step % log_every_steps == 0 or step == num_steps):
            print(f"[{desc}] step {step}/{num_steps}", flush=True)

    num_labels = len(FNC1_LABEL_ORDER)
    logits_array = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, num_labels))
    labels_array = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0,), dtype=int)
    sample_ids_array = np.concatenate(all_sample_ids, axis=0) if all_sample_ids else np.zeros((0,), dtype=int)

    return logits_array, labels_array, sample_ids_array


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    if len(logits) == 0:
        return np.zeros_like(logits)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def build_prediction_frame(
    source_df: pd.DataFrame,
    pred_label_ids: np.ndarray,
    decision_scores: np.ndarray,
) -> pd.DataFrame:
    id_to_label = {idx: label for idx, label in enumerate(FNC1_LABEL_ORDER)}
    pred_labels = [id_to_label[int(idx)] for idx in pred_label_ids]

    out = pd.DataFrame(
        {
            "sample_id": source_df["sample_id"].values,
            "headline": source_df["headline"].values,
            "body_first3sent": source_df["body_first3sent"].values,
            "gold_label_4way": source_df["label"].values,
            "pred_label_4way": pred_labels,
            "decision_score": decision_scores,
        }
    )

    mapping_a = THREE_WAY_SCHEMES[SCHEME_A_NAME]["mapping"]
    out["gold_label_3way_A"] = out["gold_label_4way"].map(mapping_a)
    out["pred_label_3way_A"] = out["pred_label_4way"].map(mapping_a)
    return out


def run_prediction(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    df: pd.DataFrame,
    config: BertExperimentConfig,
    device: torch.device,
    desc: str,
) -> pd.DataFrame:
    label_to_id = {label: idx for idx, label in enumerate(FNC1_LABEL_ORDER)}
    dataloader = build_dataloader(
        df=df,
        tokenizer=tokenizer,
        max_length=config.max_length,
        label_to_id=label_to_id,
        batch_size=config.eval_batch_size,
        shuffle=False,
        with_labels=True,
    )
    logits, _, _ = predict_logits(
        model=model,
        dataloader=dataloader,
        device=device,
        desc=desc,
        disable_tqdm=config.disable_tqdm,
        log_every_steps=config.log_every_steps,
    )

    probs = softmax_numpy(logits)
    pred_ids = np.argmax(probs, axis=1) if len(probs) > 0 else np.zeros((0,), dtype=int)
    decision_scores = (
        probs[np.arange(len(pred_ids)), pred_ids] if len(pred_ids) > 0 else np.zeros((0,), dtype=float)
    )

    return build_prediction_frame(
        source_df=df.reset_index(drop=True),
        pred_label_ids=pred_ids,
        decision_scores=decision_scores,
    )


def evaluate_and_save(
    pred_df: pd.DataFrame,
    split_name: str,
    output_dir: Path,
) -> tuple[dict, dict]:
    split_dir = output_dir / split_name

    summary_4way, per_class_4way, cm_4way, report_4way = evaluate_predictions(
        y_true=pred_df["gold_label_4way"],
        y_pred=pred_df["pred_label_4way"],
        label_order=FNC1_LABEL_ORDER,
    )
    save_evaluation(
        output_dir=split_dir / "four_way",
        summary=summary_4way,
        per_class_df=per_class_4way,
        cm_df=cm_4way,
        report_text=report_4way,
    )

    keep_mask = pred_df["gold_label_3way_A"].notna() & pred_df["pred_label_3way_A"].notna()
    pred_df_3way = pred_df[keep_mask].reset_index(drop=True)
    dropped_rows = int((~keep_mask).sum())

    summary_3way_a, per_class_3way_a, cm_3way_a, report_3way_a = evaluate_predictions(
        y_true=pred_df_3way["gold_label_3way_A"],
        y_pred=pred_df_3way["pred_label_3way_A"],
        label_order=THREE_WAY_LABEL_ORDER,
    )
    summary_3way_a["scheme_name"] = SCHEME_A_NAME
    summary_3way_a["dropped_rows"] = dropped_rows

    save_evaluation(
        output_dir=split_dir / "three_way" / SCHEME_A_NAME,
        summary=summary_3way_a,
        per_class_df=per_class_3way_a,
        cm_df=cm_3way_a,
        report_text=report_3way_a,
    )

    pred_df[
        [
            "sample_id",
            "headline",
            "body_first3sent",
            "gold_label_4way",
            "pred_label_4way",
            "gold_label_3way_A",
            "pred_label_3way_A",
            "decision_score",
        ]
    ].to_csv(split_dir / f"{split_name}_predictions.csv", index=False)

    pred_df_3way[
        [
            "sample_id",
            "headline",
            "body_first3sent",
            "gold_label_4way",
            "pred_label_4way",
            "gold_label_3way_A",
            "pred_label_3way_A",
            "decision_score",
        ]
    ].to_csv(split_dir / "three_way" / SCHEME_A_NAME / f"{split_name}_predictions_3way.csv", index=False)

    return summary_4way, summary_3way_a


def save_model_artifacts(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    config: BertExperimentConfig,
    output_dir: Path,
) -> None:
    model_root = output_dir / "model"
    model_dir = model_root / "hf_model"
    tokenizer_dir = model_root / "hf_tokenizer"

    model_root.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(tokenizer_dir)

    runtime_config = {
        "backend_name": "bert-base-uncased",
        "max_length": config.max_length,
        "label_order": FNC1_LABEL_ORDER,
        "tokenizer_dir": str(tokenizer_dir.resolve()),
        "model_dir": str(model_dir.resolve()),
    }
    (model_root / "runtime_config.json").write_text(
        json.dumps(runtime_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compare_with_svm_baseline(
    bert_test_4way: dict,
    bert_test_3way_a: dict,
    svm_baseline_dir: Path,
) -> dict:
    four_way_candidates = [
        svm_baseline_dir / "four_way" / "metrics_summary.json",
        svm_baseline_dir / "metrics_summary.json",
    ]

    svm_four_way = None
    for candidate in four_way_candidates:
        payload = load_json_if_exists(candidate)
        if payload is not None:
            svm_four_way = payload
            break

    svm_three_way_a = load_json_if_exists(
        svm_baseline_dir / "three_way" / SCHEME_A_NAME / "metrics_summary.json"
    )

    comparison = {
        "baseline_dir": str(svm_baseline_dir.resolve()),
        "four_way": {
            "bert": {
                "accuracy": bert_test_4way.get("accuracy"),
                "macro_f1": bert_test_4way.get("macro_f1"),
            },
            "svm": None,
            "delta_bert_minus_svm": None,
        },
        "three_way_scheme_a": {
            "bert": {
                "accuracy": bert_test_3way_a.get("accuracy"),
                "macro_f1": bert_test_3way_a.get("macro_f1"),
            },
            "svm": None,
            "delta_bert_minus_svm": None,
        },
    }

    if svm_four_way is not None:
        comparison["four_way"]["svm"] = {
            "accuracy": svm_four_way.get("accuracy"),
            "macro_f1": svm_four_way.get("macro_f1"),
        }
        comparison["four_way"]["delta_bert_minus_svm"] = {
            "accuracy": float(bert_test_4way.get("accuracy", 0.0) - svm_four_way.get("accuracy", 0.0)),
            "macro_f1": float(bert_test_4way.get("macro_f1", 0.0) - svm_four_way.get("macro_f1", 0.0)),
        }

    if svm_three_way_a is not None:
        comparison["three_way_scheme_a"]["svm"] = {
            "accuracy": svm_three_way_a.get("accuracy"),
            "macro_f1": svm_three_way_a.get("macro_f1"),
        }
        comparison["three_way_scheme_a"]["delta_bert_minus_svm"] = {
            "accuracy": float(
                bert_test_3way_a.get("accuracy", 0.0) - svm_three_way_a.get("accuracy", 0.0)
            ),
            "macro_f1": float(
                bert_test_3way_a.get("macro_f1", 0.0) - svm_three_way_a.get("macro_f1", 0.0)
            ),
        }

    return comparison


def train_and_evaluate(config: BertExperimentConfig) -> dict:
    set_seed(config.random_state)
    device = get_device()
    overall_start = time.perf_counter()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_train_val_test(config)

    print("===== FNC-1 BERT Upgrade: Training Started =====", flush=True)
    print(f"device: {device}", flush=True)
    print(
        f"splits -> train: {len(train_df)}, validation: {len(val_df)}, test: {len(test_df)}",
        flush=True,
    )
    print(
        f"hyperparams -> epochs: {config.num_train_epochs}, lr: {config.learning_rate}, "
        f"train_batch_size: {config.train_batch_size}, eval_batch_size: {config.eval_batch_size}, "
        f"max_length: {config.max_length}",
        flush=True,
    )

    label_to_id = {label: idx for idx, label in enumerate(FNC1_LABEL_ORDER)}
    id_to_label = {idx: label for idx, label in enumerate(FNC1_LABEL_ORDER)}

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(FNC1_LABEL_ORDER),
        label2id=label_to_id,
        id2label=id_to_label,
    )
    model.to(device)

    train_loader = build_dataloader(
        df=train_df,
        tokenizer=tokenizer,
        max_length=config.max_length,
        label_to_id=label_to_id,
        batch_size=config.train_batch_size,
        shuffle=True,
        with_labels=True,
    )

    optimizer = AdamW(
        params=model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    total_training_steps = max(1, len(train_loader) * config.num_train_epochs)
    warmup_steps = int(total_training_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    history: list[dict] = []
    best_val_macro_f1 = -1.0
    best_state_dict = None

    for epoch in range(1, config.num_train_epochs + 1):
        epoch_start = time.perf_counter()
        print(f"\n[epoch {epoch}/{config.num_train_epochs}] training...", flush=True)

        train_loss = run_training_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch_idx=epoch,
            num_epochs=config.num_train_epochs,
            log_every_steps=config.log_every_steps,
            disable_tqdm=config.disable_tqdm,
        )

        print(f"[epoch {epoch}/{config.num_train_epochs}] validating...", flush=True)
        val_pred_df = run_prediction(
            model=model,
            tokenizer=tokenizer,
            df=val_df,
            config=config,
            device=device,
            desc=f"eval val {epoch}/{config.num_train_epochs}",
        )
        val_summary_4way, _ = evaluate_and_save(
            pred_df=val_pred_df,
            split_name="validation_latest",
            output_dir=output_dir,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_summary_4way["accuracy"],
            "val_macro_f1": val_summary_4way["macro_f1"],
        }
        history.append(epoch_record)

        if val_summary_4way["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = float(val_summary_4way["macro_f1"])
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"[epoch {epoch}/{config.num_train_epochs}] new best checkpoint "
                f"(val_macro_f1={best_val_macro_f1:.4f})",
                flush=True,
            )

        epoch_seconds = time.perf_counter() - epoch_start
        print(
            f"[epoch {epoch}/{config.num_train_epochs}] done in {epoch_seconds:.1f}s | "
            f"train_loss={train_loss:.4f} | val_acc={val_summary_4way['accuracy']:.4f} | "
            f"val_macro_f1={val_summary_4way['macro_f1']:.4f}",
            flush=True,
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(device)
        print("\nLoaded best checkpoint by validation macro-F1.", flush=True)

    # Final evaluation with the best checkpoint.
    print("Running final validation and test evaluation...", flush=True)
    val_pred_df = run_prediction(
        model=model,
        tokenizer=tokenizer,
        df=val_df,
        config=config,
        device=device,
        desc="eval validation final",
    )
    test_pred_df = run_prediction(
        model=model,
        tokenizer=tokenizer,
        df=test_df,
        config=config,
        device=device,
        desc="eval test final",
    )

    val_4way, val_3way_a = evaluate_and_save(
        pred_df=val_pred_df,
        split_name="validation",
        output_dir=output_dir,
    )
    test_4way, test_3way_a = evaluate_and_save(
        pred_df=test_pred_df,
        split_name="test",
        output_dir=output_dir,
    )

    save_model_artifacts(
        model=model,
        tokenizer=tokenizer,
        config=config,
        output_dir=output_dir,
    )

    comparison = compare_with_svm_baseline(
        bert_test_4way=test_4way,
        bert_test_3way_a=test_3way_a,
        svm_baseline_dir=Path(config.svm_baseline_dir),
    )

    (output_dir / "run_config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "train_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "comparison_with_svm.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    runtime_seconds = time.perf_counter() - overall_start
    print(f"Total runtime (seconds): {runtime_seconds:.2f}", flush=True)

    return {
        "device": str(device),
        "runtime_seconds": float(runtime_seconds),
        "num_train": int(len(train_df)),
        "num_validation": int(len(val_df)),
        "num_test": int(len(test_df)),
        "validation": {
            "four_way": val_4way,
            "three_way_scheme_a": val_3way_a,
        },
        "test": {
            "four_way": test_4way,
            "three_way_scheme_a": test_3way_a,
        },
        "comparison_with_svm": comparison,
        "artifacts": {
            "output_dir": str(output_dir.resolve()),
            "model_runtime_config": str((output_dir / "model" / "runtime_config.json").resolve()),
            "validation_predictions": str((output_dir / "validation" / "validation_predictions.csv").resolve()),
            "test_predictions": str((output_dir / "test" / "test_predictions.csv").resolve()),
        },
    }


def load_inference_service_from_artifacts(output_dir: str | Path) -> BertStanceInferenceService:
    output_dir = Path(output_dir)
    runtime_config_path = output_dir / "model" / "runtime_config.json"
    if not runtime_config_path.exists():
        raise FileNotFoundError(f"Runtime config not found: {runtime_config_path}")

    runtime_config = json.loads(runtime_config_path.read_text(encoding="utf-8"))
    model_dir = Path(runtime_config["model_dir"])
    tokenizer_dir = Path(runtime_config["tokenizer_dir"])
    max_length = int(runtime_config["max_length"])

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)

    return BertStanceInferenceService(
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        device=device,
    )


_INFERENCE_CACHE: dict[str, BertStanceInferenceService] = {}


def predict_stance(
    claim: str,
    evidence_text: str,
    output_dir: str | Path = "outputs/fnc1_bert_upgrade",
) -> dict[str, float | str]:
    cache_key = str(Path(output_dir).resolve())
    if cache_key not in _INFERENCE_CACHE:
        _INFERENCE_CACHE[cache_key] = load_inference_service_from_artifacts(output_dir)
    return _INFERENCE_CACHE[cache_key].predict(claim=claim, evidence_text=evidence_text)


def main() -> None:
    config = parse_args()
    results = train_and_evaluate(config)
    runtime_seconds = float(results.get("runtime_seconds", 0.0))

    print("===== FNC-1 BERT Upgrade Finished =====")
    print(f"device: {results['device']}")
    print(f"runtime_seconds: {runtime_seconds:.2f}")
    print(f"runtime_minutes: {runtime_seconds / 60.0:.2f}")
    print(
        f"splits -> train: {results['num_train']}, "
        f"validation: {results['num_validation']}, test: {results['num_test']}"
    )
    print(
        "validation 4-way -> "
        f"accuracy: {results['validation']['four_way']['accuracy']:.4f}, "
        f"macro_f1: {results['validation']['four_way']['macro_f1']:.4f}"
    )
    print(
        "test 4-way -> "
        f"accuracy: {results['test']['four_way']['accuracy']:.4f}, "
        f"macro_f1: {results['test']['four_way']['macro_f1']:.4f}"
    )
    print(
        "test 3-way (scheme A) -> "
        f"accuracy: {results['test']['three_way_scheme_a']['accuracy']:.4f}, "
        f"macro_f1: {results['test']['three_way_scheme_a']['macro_f1']:.4f}, "
        f"dropped: {results['test']['three_way_scheme_a']['dropped_rows']}"
    )

    print("\nSaved artifacts:")
    print(f"output_dir: {results['artifacts']['output_dir']}")
    print(f"validation_predictions: {results['artifacts']['validation_predictions']}")
    print(f"test_predictions: {results['artifacts']['test_predictions']}")


if __name__ == "__main__":
    main()
