"""Zero-shot NLI via an LLM (responses or chat-completions API).

Reuses the same environment variables and HTTP machinery as the LLM claim
extractor in `claim_extraction.py` so deployments share one credential set.

Output contract: each LLM call returns a JSON object
    {"label": "entailment|contradiction|neutral", "confidence": 0..1}
which we map to the standard `(entailment, contradiction, neutral)` score
triple by spreading the residual probability over the non-predicted classes.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from p1.observability import elapsed_ms, log_event
from p1.schemas import ClaimPair, NLIPairResult, NliLabel


logger = logging.getLogger(__name__)


_LABEL_ORDER: tuple[NliLabel, NliLabel, NliLabel] = (
    NliLabel.ENTAILMENT,
    NliLabel.CONTRADICTION,
    NliLabel.NEUTRAL,
)


SYSTEM_PROMPT = (
    "You are a careful natural language inference judge. "
    "Given a PREMISE and a HYPOTHESIS, decide whether the premise "
    "entails the hypothesis, contradicts it, or is neutral. "
    "Respond ONLY with a single JSON object of the form "
    '{"label": "entailment"|"contradiction"|"neutral", "confidence": 0..1}. '
    "Do not include explanations or markdown."
)


@dataclass
class LLMNLIModel:
    """NLI model backed by an LLM zero-shot classifier.

    The model name, base url, and api key default to the same env vars used
    by the LLM claim extractor (P1_LLM_*) so a single credential is enough.
    """

    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    api_style: str | None = None  # "responses" or "chat"
    bidirectional: bool = True
    temperature: float = 0.0
    timeout_seconds: float = 30.0
    cache_dir: Path | None = None
    enable_cache: bool = True

    _runtime: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._runtime = self._resolve_runtime()
        if self.cache_dir is None:
            self.cache_dir = Path(
                os.getenv("P1_LLM_NLI_CACHE_DIR", ".cache/p1_llm_nli")
            )
        if self.enable_cache:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                self.enable_cache = False

    # ------------------------------------------------------------------
    # NLIModel protocol
    # ------------------------------------------------------------------
    def predict(self, pair: ClaimPair) -> NLIPairResult:
        return self.predict_many([pair])[0]

    def predict_many(self, pairs: list[ClaimPair]) -> list[NLIPairResult]:
        results: list[NLIPairResult] = []
        if not pairs:
            return results
        started_at = time.perf_counter()
        cache_hits = 0
        for pair in pairs:
            forward = self._classify(pair.claim_a.text, pair.claim_b.text)
            if self.bidirectional:
                backward = self._classify(pair.claim_b.text, pair.claim_a.text)
                merged = tuple(
                    round((f + b) / 2, 6) for f, b in zip(forward["scores"], backward["scores"])
                )
                cache_hits += int(forward["cache_hit"]) + int(backward["cache_hit"])
                metadata = {
                    "model": self._runtime.get("model"),
                    "forward_scores": forward["scores"],
                    "backward_scores": backward["scores"],
                    "forward_label": forward["label"],
                    "backward_label": backward["label"],
                    "forward_cache": forward["cache_hit"],
                    "backward_cache": backward["cache_hit"],
                    "nli_batch_size": len(pairs),
                }
            else:
                merged = forward["scores"]
                cache_hits += int(forward["cache_hit"])
                metadata = {
                    "model": self._runtime.get("model"),
                    "forward_scores": forward["scores"],
                    "forward_label": forward["label"],
                    "forward_cache": forward["cache_hit"],
                    "nli_batch_size": len(pairs),
                }
            label = _scores_to_label(merged)
            results.append(
                NLIPairResult(
                    claim_a_id=pair.claim_a.claim_id,
                    claim_b_id=pair.claim_b.claim_id,
                    entailment_score=merged[0],
                    contradiction_score=merged[1],
                    neutral_score=merged[2],
                    label=label,
                    is_bidirectional=self.bidirectional,
                    metadata=metadata,
                )
            )
        log_event(
            logger,
            "p1.llm_nli.batch",
            pairs=len(pairs),
            bidirectional=self.bidirectional,
            cache_hits=cache_hits,
            duration_ms=elapsed_ms(started_at, time.perf_counter()),
        )
        return results

    # ------------------------------------------------------------------
    # Classification core
    # ------------------------------------------------------------------
    def _classify(self, premise: str, hypothesis: str) -> dict[str, Any]:
        cache_key = self._cache_key(premise, hypothesis)
        cached = self._cache_load(cache_key) if self.enable_cache else None
        if cached is not None:
            return {"scores": tuple(cached["scores"]), "label": cached["label"], "cache_hit": True}

        if self._runtime.get("error"):
            # Soft fallback: model not configured. Returns slight neutral
            # bias so argmax resolves to NEUTRAL even after bidirectional
            # averaging, and ensembles can proceed without crashing.
            scores = (0.33, 0.33, 0.34)
            return {"scores": scores, "label": NliLabel.NEUTRAL.value, "cache_hit": False}

        parsed = self._call_llm(premise, hypothesis)
        scores = _spread_confidence(parsed["label"], parsed["confidence"])
        if self.enable_cache:
            self._cache_store(cache_key, {"scores": list(scores), "label": parsed["label"]})
        return {"scores": scores, "label": parsed["label"], "cache_hit": False}

    def _call_llm(self, premise: str, hypothesis: str) -> dict[str, Any]:
        # Imported lazily so that test environments without the heavyweight
        # claim extractor module path still work.
        from p1.claim_extraction import (
            _post_chat_completion,
            _post_responses_api,
            _extract_text_from_responses_api,
            _extract_first_json_object,
        )

        user_prompt = (
            f"PREMISE: {premise.strip()}\n"
            f"HYPOTHESIS: {hypothesis.strip()}\n"
            'Answer with JSON only: {"label": "entailment|contradiction|neutral", "confidence": 0..1}'
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        api_style = self._runtime["api_style"]
        if api_style == "responses":
            response_json = _post_responses_api(
                api_key=str(self._runtime["api_key"]),
                base_url=str(self._runtime["base_url"]),
                model=str(self._runtime["model"]),
                messages=messages,
                timeout_seconds=self.timeout_seconds,
                temperature=self.temperature,
            )
            text = _extract_text_from_responses_api(response_json)
        else:
            response_json = _post_chat_completion(
                api_key=str(self._runtime["api_key"]),
                base_url=str(self._runtime["base_url"]),
                model=str(self._runtime["model"]),
                messages=messages,
                timeout_seconds=self.timeout_seconds,
                temperature=self.temperature,
            )
            choices = response_json.get("choices") or []
            if not choices:
                raise RuntimeError("llm_nli_empty_choices")
            content = (choices[0].get("message") or {}).get("content")
            if isinstance(content, list):
                text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            elif isinstance(content, str):
                text = content
            else:
                raise RuntimeError("llm_nli_missing_content")

        raw_json = _extract_first_json_object(text)
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"llm_nli_invalid_json:{exc.msg}") from exc
        return {
            "label": _normalize_label(parsed.get("label")),
            "confidence": _clip_confidence(parsed.get("confidence")),
        }

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _cache_key(self, premise: str, hypothesis: str) -> str:
        h = hashlib.sha256()
        h.update((self._runtime.get("model") or "_").encode("utf-8"))
        h.update(b"|")
        h.update(premise.encode("utf-8"))
        h.update(b"|")
        h.update(hypothesis.encode("utf-8"))
        return h.hexdigest()

    def _cache_path(self, key: str) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / f"{key}.json"

    def _cache_load(self, key: str) -> dict[str, Any] | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _cache_store(self, key: str, payload: dict[str, Any]) -> None:
        try:
            self._cache_path(key).write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def _resolve_runtime(self) -> dict[str, Any]:
        api_key = self.api_key or os.getenv("P1_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = (
            self.base_url
            or os.getenv("P1_LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        model = self.model or os.getenv("P1_LLM_NLI_MODEL") or os.getenv("P1_LLM_MODEL")
        api_style = (
            self.api_style or os.getenv("P1_LLM_API_STYLE") or "chat"
        ).strip().lower()
        if api_style not in {"responses", "chat"}:
            api_style = "chat"
        if not api_key:
            return {"error": "missing_api_key", "api_style": api_style, "model": model, "base_url": base_url, "api_key": None}
        if not model:
            return {"error": "missing_model", "api_style": api_style, "model": None, "base_url": base_url, "api_key": api_key}
        return {"error": None, "api_style": api_style, "model": model, "base_url": base_url, "api_key": api_key}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _normalize_label(raw: Any) -> str:
    if not isinstance(raw, str):
        return NliLabel.NEUTRAL.value
    text = raw.strip().lower()
    if text in {"entail", "entailment", "support", "supports", "agree"}:
        return NliLabel.ENTAILMENT.value
    if text in {"contradict", "contradiction", "refute", "refutes", "disagree"}:
        return NliLabel.CONTRADICTION.value
    return NliLabel.NEUTRAL.value


def _clip_confidence(raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.5
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _spread_confidence(label: str, confidence: float) -> tuple[float, float, float]:
    confidence = max(0.34, min(confidence, 0.99))  # never collapse to 0
    residual = (1.0 - confidence) / 2.0
    if label == NliLabel.ENTAILMENT.value:
        return (round(confidence, 6), round(residual, 6), round(residual, 6))
    if label == NliLabel.CONTRADICTION.value:
        return (round(residual, 6), round(confidence, 6), round(residual, 6))
    return (round(residual, 6), round(residual, 6), round(confidence, 6))


def _scores_to_label(scores: tuple[float, float, float]) -> NliLabel:
    best_index = max(range(3), key=lambda i: scores[i])
    return _LABEL_ORDER[best_index]
