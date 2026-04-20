from __future__ import annotations

import json
import re
from pathlib import Path

from p1.schemas import RetrievalInput, RetrievedChunk


def read_ramdocs_jsonl(path: str | Path, limit: int | None = None) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def normalize_ramdocs_record(raw: dict, *, sample_id: str) -> dict:
    documents = []
    for index, item in enumerate(raw.get("documents", []) or []):
        documents.append(
            {
                "doc_id": f"{sample_id}:doc{index}",
                "text": (item.get("text") or "").strip(),
                "type": (item.get("type") or "").strip().lower(),
                "answer": (item.get("answer") or "").strip(),
                "rank": index + 1,
            }
        )
    return {
        "sample_id": sample_id,
        "question": (raw.get("question") or "").strip(),
        "gold_answers": raw.get("gold_answers") or [],
        "wrong_answers": raw.get("wrong_answers") or [],
        "disambig_entity": raw.get("disambig_entity"),
        "documents": [doc for doc in documents if doc["text"]],
    }


def load_ramdocs_records(path: str | Path, limit: int | None = None) -> list[dict]:
    raw_records = read_ramdocs_jsonl(path, limit=limit)
    return [
        normalize_ramdocs_record(raw, sample_id=f"ramdocs:{index}")
        for index, raw in enumerate(raw_records)
    ]


def ramdocs_record_to_retrieval_input(record: dict, *, answer_aware: bool = False) -> RetrievalInput:
    return RetrievalInput(
        sample_id=record["sample_id"],
        query=record["question"],
        label=None,
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id=document["doc_id"],
                text=_maybe_append_answer_aware_sentence(
                    text=document["text"],
                    question=record["question"],
                    answer=document["answer"],
                    enabled=answer_aware,
                ),
                rank=document["rank"],
                retrieval_score=1.0 / document["rank"],
                metadata={
                    "dataset": "ramdocs",
                    "ramdocs_type": document["type"],
                    "answer": document["answer"],
                    "gold_answers": record["gold_answers"],
                    "wrong_answers": record["wrong_answers"],
                    "disambig_entity": record["disambig_entity"],
                    "answer_aware_enabled": answer_aware,
                    "answer_aware_sentence": _build_answer_aware_sentence(record["question"], document["answer"])
                    if answer_aware
                    else None,
                },
            )
            for document in record["documents"]
        ],
        metadata={
            "dataset": "ramdocs",
            "gold_answers": record["gold_answers"],
            "wrong_answers": record["wrong_answers"],
            "disambig_entity": record["disambig_entity"],
            "answer_aware_enabled": answer_aware,
        },
    )


def _maybe_append_answer_aware_sentence(
    *,
    text: str,
    question: str,
    answer: str,
    enabled: bool,
) -> str:
    if not enabled:
        return text
    answer_sentence = _build_answer_aware_sentence(question, answer)
    if answer_sentence is None:
        return text
    if answer_sentence.lower() in text.lower():
        return text
    return f"{text.rstrip()} {answer_sentence}"


def _build_answer_aware_sentence(question: str, answer: str) -> str | None:
    clean_answer = _clean_answer(answer)
    if not clean_answer:
        return None

    clean_question = re.sub(r"\s+", " ", question).strip().rstrip("?")
    if not clean_question:
        return None

    lowered = clean_question.lower()
    patterns = (
        (r"^what is the population of (?P<subject>.+)$", "The population of {subject} is {answer}."),
        (r"^what sport is (?P<subject>.+?) associated with$", "{subject} is associated with {answer}."),
        (r"^who are the directors of the film (?P<subject>.+)$", "The directors of the film {subject} are {answer}."),
        (r"^who is the director of the film (?P<subject>.+)$", "The director of the film {subject} is {answer}."),
        (r"^who directed (?P<subject>.+)$", "{subject} was directed by {answer}."),
        (r"^who is (?P<subject>.+)$", "{subject} is {answer}."),
        (r"^who was (?P<subject>.+)$", "{subject} was {answer}."),
        (r"^what is (?P<subject>.+)$", "{subject} is {answer}."),
        (r"^what are (?P<subject>.+)$", "{subject} are {answer}."),
        (r"^where is (?P<subject>.+)$", "{subject} is in {answer}."),
        (r"^when was (?P<subject>.+)$", "{subject} was in {answer}."),
    )
    for pattern, template in patterns:
        match = re.match(pattern, lowered, flags=re.IGNORECASE)
        if match:
            subject = _restore_subject_text(clean_question, match.group("subject"))
            return template.format(subject=subject, answer=clean_answer)

    return f"The answer to {clean_question} is {clean_answer}."


def _clean_answer(answer: str) -> str | None:
    cleaned = re.sub(r"\s+", " ", str(answer or "")).strip()
    if not cleaned or cleaned.lower() in {"unknown", "none", "n/a", "null"}:
        return None
    return cleaned.rstrip(".")


def _restore_subject_text(question: str, lowered_subject: str) -> str:
    normalized_subject = re.sub(r"\s+", " ", lowered_subject).strip()
    start = question.lower().find(normalized_subject.lower())
    if start < 0:
        return normalized_subject.strip(" \"'")
    return question[start : start + len(normalized_subject)].strip(" \"'")
