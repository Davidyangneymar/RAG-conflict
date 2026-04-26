from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NormalizedRecord:
    sample_id: str
    dataset: str
    query: str
    gold_label: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionRecord:
    sample_id: str
    predicted_label: str | None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
