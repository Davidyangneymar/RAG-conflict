from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fnc1_stance_module import FNC1_LABEL_ORDER, first_n_sentences

DEFAULT_BERT_OUTPUT_DIR = "outputs/fnc1_bert_upgrade_full"
SCHEME_A_NAME = "scheme_a_filter_unrelated"
SCHEME_A_LABEL_MAP = {
    "agree": "support",
    "disagree": "oppose",
    "discuss": "neutral",
    "unrelated": None,
}


@dataclass
class BertRuntimeConfig:
    backend_name: str
    max_length: int
    label_order: list[str]
    tokenizer_dir: str
    model_dir: str


def load_runtime_config(output_dir: str | Path = DEFAULT_BERT_OUTPUT_DIR) -> BertRuntimeConfig:
    output_dir = Path(output_dir)
    runtime_config_path = output_dir / "model" / "runtime_config.json"
    if not runtime_config_path.exists():
        raise FileNotFoundError(f"Runtime config not found: {runtime_config_path}")

    payload = json.loads(runtime_config_path.read_text(encoding="utf-8"))

    required_keys = {"backend_name", "max_length", "label_order", "tokenizer_dir", "model_dir"}
    missing_keys = sorted(required_keys - set(payload.keys()))
    if missing_keys:
        raise ValueError(f"runtime_config.json missing keys: {missing_keys}")

    return BertRuntimeConfig(
        backend_name=str(payload["backend_name"]),
        max_length=int(payload["max_length"]),
        label_order=[str(v) for v in payload["label_order"]],
        tokenizer_dir=str(payload["tokenizer_dir"]),
        model_dir=str(payload["model_dir"]),
    )


def resolve_device(preferred_device: str | None = None) -> torch.device:
    if preferred_device:
        return torch.device(preferred_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preprocess_claim_and_evidence(claim: str, evidence_text: str) -> tuple[str, str]:
    claim_text = str(claim).strip()
    body_first3sent = first_n_sentences(str(evidence_text), n=3)
    return claim_text, body_first3sent


def map_four_to_three_scheme_a(label_4way: str) -> str | None:
    return SCHEME_A_LABEL_MAP.get(str(label_4way).strip().lower())


def map_four_to_three_scheme_a_with_filter(label_4way: str) -> str:
    mapped = map_four_to_three_scheme_a(label_4way)
    return mapped if mapped is not None else "filtered"


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    if logits.size == 0:
        return np.zeros_like(logits)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def build_prediction_output(
    claim: str,
    body_first3sent: str,
    pred_label_4way: str,
    decision_score: float,
) -> dict[str, str | float | bool | None]:
    pred_label_3way_a = map_four_to_three_scheme_a(pred_label_4way)
    return {
        "claim": claim,
        "body_first3sent": body_first3sent,
        "pred_label_4way": pred_label_4way,
        "pred_label_3way_a": pred_label_3way_a,
        "pred_label_3way_a_with_filter": (
            pred_label_3way_a if pred_label_3way_a is not None else "filtered"
        ),
        "is_filtered_3way_a": pred_label_3way_a is None,
        "decision_score": float(decision_score),
        "mapping_scheme": SCHEME_A_NAME,
    }


class BertStancePredictor:
    def __init__(
        self,
        model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        max_length: int,
        device: torch.device,
        label_order: list[str],
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.device = device
        self.label_order = [str(label) for label in label_order]

    @classmethod
    def from_output_dir(
        cls,
        output_dir: str | Path = DEFAULT_BERT_OUTPUT_DIR,
        preferred_device: str | None = None,
    ) -> "BertStancePredictor":
        runtime_cfg = load_runtime_config(output_dir=output_dir)

        tokenizer = AutoTokenizer.from_pretrained(runtime_cfg.tokenizer_dir)
        model = AutoModelForSequenceClassification.from_pretrained(runtime_cfg.model_dir)

        device = resolve_device(preferred_device=preferred_device)
        model.to(device)
        model.eval()

        label_order = runtime_cfg.label_order
        if len(label_order) != len(FNC1_LABEL_ORDER):
            label_order = FNC1_LABEL_ORDER

        return cls(
            model=model,
            tokenizer=tokenizer,
            max_length=runtime_cfg.max_length,
            device=device,
            label_order=label_order,
        )

    @torch.no_grad()
    def _predict_from_pairs(
        self,
        claims: list[str],
        body_first3sents: list[str],
    ) -> tuple[list[str], np.ndarray]:
        encoded = self.tokenizer(
            claims,
            body_first3sents,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        logits = self.model(**encoded).logits.detach().cpu().numpy()
        probs = softmax_numpy(logits)

        pred_ids = np.argmax(probs, axis=1) if len(probs) > 0 else np.zeros((0,), dtype=int)
        pred_labels = [self.label_order[int(pred_id)] for pred_id in pred_ids]
        decision_scores = probs[np.arange(len(pred_ids)), pred_ids] if len(pred_ids) > 0 else np.zeros((0,))

        return pred_labels, decision_scores

    def predict_stance(self, claim: str, evidence_text: str) -> dict[str, str | float | bool | None]:
        claim_text, body_first3sent = preprocess_claim_and_evidence(claim=claim, evidence_text=evidence_text)

        pred_labels, decision_scores = self._predict_from_pairs(
            claims=[claim_text],
            body_first3sents=[body_first3sent],
        )

        return build_prediction_output(
            claim=claim_text,
            body_first3sent=body_first3sent,
            pred_label_4way=str(pred_labels[0]),
            decision_score=float(decision_scores[0]),
        )

    def predict_stance_batch(
        self,
        claims: Iterable[str],
        evidence_texts: Iterable[str],
    ) -> list[dict[str, str | float | bool | None]]:
        claims_list = list(claims)
        evidence_list = list(evidence_texts)

        if len(claims_list) != len(evidence_list):
            raise ValueError(
                "claims and evidence_texts must have the same length. "
                f"Got {len(claims_list)} vs {len(evidence_list)}"
            )

        preprocessed = [
            preprocess_claim_and_evidence(claim=claim, evidence_text=evidence)
            for claim, evidence in zip(claims_list, evidence_list)
        ]
        processed_claims = [item[0] for item in preprocessed]
        body_first3sents = [item[1] for item in preprocessed]

        pred_labels, decision_scores = self._predict_from_pairs(
            claims=processed_claims,
            body_first3sents=body_first3sents,
        )

        return [
            build_prediction_output(
                claim=claim,
                body_first3sent=body_first3sent,
                pred_label_4way=pred_label_4way,
                decision_score=float(decision_score),
            )
            for claim, body_first3sent, pred_label_4way, decision_score in zip(
                processed_claims,
                body_first3sents,
                pred_labels,
                decision_scores,
            )
        ]

    def predict_stance_batch_as_dataframe(
        self,
        claims: Iterable[str],
        evidence_texts: Iterable[str],
    ) -> pd.DataFrame:
        return pd.DataFrame(self.predict_stance_batch(claims=claims, evidence_texts=evidence_texts))


_PREDICTOR_CACHE: dict[str, BertStancePredictor] = {}


def load_bert_stance_predictor(
    output_dir: str | Path = DEFAULT_BERT_OUTPUT_DIR,
    preferred_device: str | None = None,
) -> BertStancePredictor:
    cache_key = f"{Path(output_dir).resolve()}::{preferred_device or 'auto'}"
    if cache_key not in _PREDICTOR_CACHE:
        _PREDICTOR_CACHE[cache_key] = BertStancePredictor.from_output_dir(
            output_dir=output_dir,
            preferred_device=preferred_device,
        )
    return _PREDICTOR_CACHE[cache_key]


def predict_stance(
    claim: str,
    evidence_text: str,
    output_dir: str | Path = DEFAULT_BERT_OUTPUT_DIR,
    preferred_device: str | None = None,
) -> dict[str, str | float | bool | None]:
    predictor = load_bert_stance_predictor(output_dir=output_dir, preferred_device=preferred_device)
    return predictor.predict_stance(claim=claim, evidence_text=evidence_text)


def predict_stance_batch(
    claims: Iterable[str],
    evidence_texts: Iterable[str],
    output_dir: str | Path = DEFAULT_BERT_OUTPUT_DIR,
    preferred_device: str | None = None,
) -> list[dict[str, str | float | bool | None]]:
    predictor = load_bert_stance_predictor(output_dir=output_dir, preferred_device=preferred_device)
    return predictor.predict_stance_batch(claims=claims, evidence_texts=evidence_texts)
