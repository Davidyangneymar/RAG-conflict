"""
AVeriTeC -> P1-shape payload adapter.

WHY THIS LIVES IN P2:
---------------------
P1 is the team that does real claim extraction, blocking, and NLI on
retrieved evidence. Until P1's module runs end-to-end on AVeriTeC we
still need to exercise P2's stance + conflict typing on real data.

This adapter is therefore a *development-only* shim that converts an
AVeriTeC record into the same JSON shape P1 will eventually emit. When
P1 ships, this file can be deleted or kept as a fallback — P2 code
does not depend on it at runtime, only tests and benchmark scripts do.

AVeriTeC record shape we consume (subset):
    {
      "claim": "<main claim text>",
      "label": "Supported|Refuted|Not Enough Evidence|Conflicting Evidence/Cherrypicking",
      "claim_date": "YYYY-MM-DD",
      "speaker": "string",
      "questions": [
        {
          "question": "...",
          "answers": [
            {"answer": "...", "source_url": "...", "source_medium": "..."}
          ]
        }
      ]
    }

Produced P1-shape record (subset that P2 actually reads):
    {
      "sample_id": "...",
      "claims": [
        {"claim_id": "Q", "text": <main claim>, "time": <claim_date>,
         "source_metadata": {"role": "query", "speaker": ...}},
        {"claim_id": "E0", "text": <answer>, ...,
         "source_metadata": {"role": "retrieved_evidence",
                             "source_medium": ..., "question": ...}},
        ...
      ],
      "candidate_pairs": [{"claim_a_id": "Q", "claim_b_id": "E0"}, ...],
      "nli_results": []   # left empty; P2 falls back to stance-only
    }

nli_results is left empty on purpose. This is the honest behavior:
without a real NLI model we do NOT fabricate labels. The P2 fusion
layer already handles `nli_label=None` cleanly (stance-only path).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


AVERITEC_VERDICT_LABELS = (
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
)


_SOCIAL_REPORTING_RE = re.compile(
    r"facebook|twitter|tweeter|instagram|youtube|tiktok|whatsapp|social\s*media",
    flags=re.IGNORECASE,
)
_BLOG_REPORTING_RE = re.compile(r"blog|forum|reddit", flags=re.IGNORECASE)
_GOV_REPORTING_RE = re.compile(
    r"government|ministry|official|press\s*release|state\s*media",
    flags=re.IGNORECASE,
)
_NEWS_REPORTING_RE = re.compile(
    r"news|reuters|cnn|bbc|channel|television|newspaper|media",
    flags=re.IGNORECASE,
)


def _infer_query_source_medium(record: Dict[str, Any]) -> Optional[str]:
    """Infer a coarse source_medium for the query side from reporting_source."""
    reporting_source = str(record.get("reporting_source") or "").strip()
    if not reporting_source:
        return None

    if _SOCIAL_REPORTING_RE.search(reporting_source):
        return "social_media"
    if _BLOG_REPORTING_RE.search(reporting_source):
        return "blog"
    if _GOV_REPORTING_RE.search(reporting_source):
        return "government"
    if _NEWS_REPORTING_RE.search(reporting_source):
        return "news"
    return None


def averitec_record_to_p1_payload(
    record: Dict[str, Any],
    sample_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert one AVeriTeC record into a P1-shape dict."""
    claim_text = record.get("claim") or ""
    if not claim_text:
        raise ValueError("AVeriTeC record has no 'claim' field")

    sid = sample_id or str(record.get("sample_id") or record.get("id") or _hash_claim(claim_text))
    query_source_medium = _infer_query_source_medium(record)

    claims: List[Dict[str, Any]] = []
    claims.append(
        {
            "claim_id": "Q",
            "text": claim_text,
            "time": record.get("claim_date"),
            "source_metadata": {
                "role": "query",
                "speaker": record.get("speaker"),
                "claim_date": record.get("claim_date"),
                "reporting_source": record.get("reporting_source"),
                "source_medium": query_source_medium,
                "source_url": record.get("original_claim_url") or record.get("cached_original_claim_url"),
            },
        }
    )

    candidate_pairs: List[Dict[str, Any]] = []
    ev_idx = 0
    for q_block in record.get("questions") or []:
        question_text = q_block.get("question", "") if isinstance(q_block, dict) else ""
        for ans in q_block.get("answers") or []:
            answer_text = (ans.get("answer") or "").strip()
            if not answer_text:
                continue
            claim_id = f"E{ev_idx}"
            claims.append(
                {
                    "claim_id": claim_id,
                    "text": answer_text,
                    "source_metadata": {
                        "role": "retrieved_evidence",
                        "source_url": ans.get("source_url"),
                        "source_medium": ans.get("source_medium"),
                        "question": question_text,
                    },
                }
            )
            candidate_pairs.append({"claim_a_id": "Q", "claim_b_id": claim_id})
            ev_idx += 1

    return {
        "sample_id": sid,
        "claims": claims,
        "candidate_pairs": candidate_pairs,
        "nli_results": [],
        # benchmark passthrough — NOT part of P1 contract; ignored by
        # adapter but read by the AVeriTeC scorer script.
        "_averitec_gold": record.get("label"),
    }


def averitec_records_to_p1_payload(
    records: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [averitec_record_to_p1_payload(r, sample_id=f"averitec_{i}") for i, r in enumerate(records)]


def load_averitec_json(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load the standard AVeriTeC JSON file (array of records)."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON list of AVeriTeC records, got {type(data).__name__}")
    return data


def load_averitec_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load an AVeriTeC JSONL file (one record per line)."""
    p = Path(path)
    records: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _hash_claim(text: str) -> str:
    # stable short id for records that lack an explicit id
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
