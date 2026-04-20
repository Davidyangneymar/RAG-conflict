"""
P1 -> P2 input adapter.

Parses the JSON payload emitted by P1 (claim extraction + candidate pair
blocking + NLI) into P2-internal dataclasses. Joins between claims,
candidate pairs, and NLI results are done strictly by id (never by list
order), so this adapter is robust to reordering upstream.

This module is intentionally model-agnostic: it does NOT import or touch
the BERT stance code. Downstream stance analysis and conflict typing
should consume the dataclasses defined here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class P1SchemaError(ValueError):
    """Raised when the P1 payload is missing required fields or is malformed."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Claim:
    claim_id: str
    text: str
    subject: Optional[str] = None
    relation: Optional[str] = None
    object: Optional[str] = None
    qualifier: Optional[str] = None
    time: Optional[str] = None
    role: Optional[str] = None  # flattened from source_metadata.role
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidatePair:
    claim_a_id: str
    claim_b_id: str
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple:
        return (self.claim_a_id, self.claim_b_id)


@dataclass
class NLIResult:
    claim_a_id: str
    claim_b_id: str
    label: str
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple:
        return (self.claim_a_id, self.claim_b_id)


@dataclass
class InputRecord:
    """One P1 sample, fully parsed and indexed for P2 consumption."""

    sample_id: str
    claims: List[Claim]
    candidate_pairs: List[CandidatePair]
    nli_results: List[NLIResult]
    raw: Dict[str, Any] = field(default_factory=dict)

    # indices built once at parse time, so P2 code doesn't rescan lists
    _claims_by_id: Dict[str, Claim] = field(default_factory=dict, repr=False)

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        return self._claims_by_id.get(claim_id)

    def has_claim(self, claim_id: str) -> bool:
        return claim_id in self._claims_by_id

    def iter_pair_with_nli(self):
        """
        Yield (CandidatePair, Claim_a, Claim_b, NLIResult or None).
        Joined by (claim_a_id, claim_b_id). NLI may be None if P1 did not
        emit an NLI result for that pair.
        """
        nli_by_key = {r.key: r for r in self.nli_results}
        for pair in self.candidate_pairs:
            yield (
                pair,
                self._claims_by_id.get(pair.claim_a_id),
                self._claims_by_id.get(pair.claim_b_id),
                nli_by_key.get(pair.key),
            )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _require(d: Dict[str, Any], key: str, ctx: str) -> Any:
    if not isinstance(d, dict):
        raise P1SchemaError(f"{ctx}: expected a JSON object, got {type(d).__name__}")
    if key not in d:
        raise P1SchemaError(f"{ctx}: missing required field '{key}'")
    value = d[key]
    if value is None or (isinstance(value, str) and value == ""):
        raise P1SchemaError(f"{ctx}: required field '{key}' is empty or null")
    return value


def _parse_claim(d: Dict[str, Any], ctx: str) -> Claim:
    claim_id = _require(d, "claim_id", ctx)
    text = _require(d, "text", ctx)

    source_metadata = d.get("source_metadata") or {}
    if not isinstance(source_metadata, dict):
        raise P1SchemaError(f"{ctx}: 'source_metadata' must be an object if present")
    role = source_metadata.get("role")

    return Claim(
        claim_id=str(claim_id),
        text=str(text),
        subject=d.get("subject"),
        relation=d.get("relation"),
        object=d.get("object"),
        qualifier=d.get("qualifier"),
        time=d.get("time"),
        role=role,
        source_metadata=source_metadata,
        raw=d,
    )


def _parse_candidate_pair(d: Dict[str, Any], ctx: str) -> CandidatePair:
    claim_a_id = _require(d, "claim_a_id", ctx)
    claim_b_id = _require(d, "claim_b_id", ctx)
    return CandidatePair(
        claim_a_id=str(claim_a_id),
        claim_b_id=str(claim_b_id),
        raw=d,
    )


def _parse_nli_result(d: Dict[str, Any], ctx: str) -> NLIResult:
    claim_a_id = _require(d, "claim_a_id", ctx)
    claim_b_id = _require(d, "claim_b_id", ctx)
    label = _require(d, "label", ctx)
    return NLIResult(
        claim_a_id=str(claim_a_id),
        claim_b_id=str(claim_b_id),
        label=str(label),
        raw=d,
    )


def _parse_one_record(data: Dict[str, Any], ctx: str) -> InputRecord:
    if not isinstance(data, dict):
        raise P1SchemaError(f"{ctx}: expected a JSON object for a sample")

    sample_id = _require(data, "sample_id", ctx)

    for key in ("claims", "candidate_pairs", "nli_results"):
        if key not in data:
            raise P1SchemaError(f"{ctx}: missing required field '{key}'")
        if not isinstance(data[key], list):
            raise P1SchemaError(f"{ctx}: field '{key}' must be a list")

    claims = [
        _parse_claim(c, f"{ctx}.claims[{i}]") for i, c in enumerate(data["claims"])
    ]

    # Catch duplicate claim_ids early — all joins assume ids are unique.
    seen_ids: Dict[str, int] = {}
    for i, c in enumerate(claims):
        if c.claim_id in seen_ids:
            raise P1SchemaError(
                f"{ctx}.claims[{i}]: duplicate claim_id '{c.claim_id}' "
                f"(also at index {seen_ids[c.claim_id]})"
            )
        seen_ids[c.claim_id] = i

    candidate_pairs = [
        _parse_candidate_pair(p, f"{ctx}.candidate_pairs[{i}]")
        for i, p in enumerate(data["candidate_pairs"])
    ]
    nli_results = [
        _parse_nli_result(n, f"{ctx}.nli_results[{i}]")
        for i, n in enumerate(data["nli_results"])
    ]

    record = InputRecord(
        sample_id=str(sample_id),
        claims=claims,
        candidate_pairs=candidate_pairs,
        nli_results=nli_results,
        raw=data,
        _claims_by_id={c.claim_id: c for c in claims},
    )
    return record


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_p1_payload(
    data: Union[Dict[str, Any], List[Dict[str, Any]]],
) -> List[InputRecord]:
    """
    Convert a dict (one sample) or list of dicts (batch of samples) into
    a list of InputRecord. Always returns a list so downstream code can
    treat single- and multi-sample payloads uniformly.
    """
    if isinstance(data, dict):
        return [_parse_one_record(data, ctx="payload")]
    if isinstance(data, list):
        return [
            _parse_one_record(item, ctx=f"payload[{i}]") for i, item in enumerate(data)
        ]
    raise P1SchemaError(
        f"payload: expected dict or list at top level, got {type(data).__name__}"
    )


def load_p1_payload(path: Union[str, Path]) -> List[InputRecord]:
    """Read a JSON file from disk and parse it into InputRecord objects."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"P1 payload file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return parse_p1_payload(data)
