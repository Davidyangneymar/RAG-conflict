from __future__ import annotations

import json
from pathlib import Path

from p1.schemas import ChunkInput, RetrievalInput, RetrievedChunk


AVERITEC_LABEL_TO_NLI = {
    "Supported": "entailment",
    "Refuted": "contradiction",
    "Not Enough Evidence": "neutral",
    "Conflicting Evidence/Cherrypicking": "neutral",
}


def read_averitec_json(path: str | Path) -> list[dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("data", "records", "examples"):
            if isinstance(raw.get(key), list):
                return raw[key]
    raise ValueError(f"Unsupported AVeriTeC payload at {path}")


def normalize_averitec_record(
    raw: dict,
    *,
    sample_id: str,
    split: str,
    source_path: str | Path,
) -> dict:
    questions = []
    for question_index, question_item in enumerate(raw.get("questions", []) or []):
        answers = []
        for answer_index, answer_item in enumerate(question_item.get("answers", []) or []):
            answers.append(
                {
                    "answer_id": f"{sample_id}:q{question_index}:a{answer_index}",
                    "answer": (answer_item.get("answer") or "").strip(),
                    "answer_type": answer_item.get("answer_type"),
                    "source_url": answer_item.get("source_url"),
                    "source_medium": answer_item.get("source_medium"),
                    "cached_source_url": answer_item.get("cached_source_url"),
                }
            )
        questions.append(
            {
                "question_id": f"{sample_id}:q{question_index}",
                "question": (question_item.get("question") or "").strip(),
                "answers": [answer for answer in answers if answer["answer"]],
            }
        )

    label = (raw.get("label") or "").strip()
    return {
        "sample_id": sample_id,
        "split": split,
        "claim": (raw.get("claim") or "").strip(),
        "label": label,
        "nli_label": AVERITEC_LABEL_TO_NLI.get(label, "neutral"),
        "justification": (raw.get("justification") or "").strip(),
        "speaker": raw.get("speaker"),
        "reporting_source": raw.get("reporting_source"),
        "claim_date": raw.get("claim_date"),
        "location_iso_code": raw.get("location_ISO_code"),
        "claim_types": raw.get("claim_types") or [],
        "fact_checking_strategies": raw.get("fact_checking_strategies") or [],
        "fact_checking_article": raw.get("fact_checking_article"),
        "original_claim_url": raw.get("original_claim_url"),
        "cached_original_claim_url": raw.get("cached_original_claim_url"),
        "required_reannotation": bool(raw.get("required_reannotation", False)),
        "questions": questions,
        "question_count": len(questions),
        "answer_count": sum(len(item["answers"]) for item in questions),
        "source_path": str(source_path),
    }


def load_averitec_records(path: str | Path, limit: int | None = None) -> list[dict]:
    source = Path(path)
    split = source.stem
    raw_records = read_averitec_json(source)
    if limit is not None:
        raw_records = raw_records[:limit]
    return [
        normalize_averitec_record(
            raw_record,
            sample_id=f"averitec:{split}:{index}",
            split=split,
            source_path=source,
        )
        for index, raw_record in enumerate(raw_records)
    ]


def averitec_record_to_claim_chunk(record: dict) -> ChunkInput:
    return ChunkInput(
        doc_id=record["sample_id"],
        chunk_id="claim",
        text=record["claim"],
        metadata={
            "dataset": "averitec",
            "split": record["split"],
            "label": record["label"],
            "speaker": record["speaker"],
            "claim_date": record["claim_date"],
            "reporting_source": record["reporting_source"],
        },
    )


def averitec_record_to_question_chunks(
    record: dict,
    *,
    include_questions: bool = False,
    include_answers: bool = True,
    max_questions: int = 2,
    max_answers_per_question: int = 2,
) -> list[ChunkInput]:
    chunks: list[ChunkInput] = []
    for question_index, question_item in enumerate(record.get("questions", [])[:max_questions]):
        if include_questions and question_item.get("question"):
            chunks.append(
                ChunkInput(
                    doc_id=record["sample_id"],
                    chunk_id=f"question-{question_index}",
                    text=question_item["question"],
                    metadata={
                        "dataset": "averitec",
                        "split": record["split"],
                        "label": record["label"],
                        "role": "question",
                        "question_id": question_item["question_id"],
                    },
                )
            )
        if include_answers:
            for answer_index, answer_item in enumerate(question_item.get("answers", [])[:max_answers_per_question]):
                answer_text = (answer_item.get("answer") or "").strip()
                if not answer_text:
                    continue
                chunks.append(
                    ChunkInput(
                        doc_id=record["sample_id"],
                        chunk_id=f"answer-{question_index}-{answer_index}",
                        text=answer_text,
                        metadata={
                            "dataset": "averitec",
                            "split": record["split"],
                            "label": record["label"],
                            "role": "answer",
                            "question_id": question_item["question_id"],
                            "answer_type": answer_item.get("answer_type"),
                            "source_medium": answer_item.get("source_medium"),
                            "source_url": answer_item.get("source_url"),
                        },
                    )
                )
    return chunks


def averitec_record_to_chunks(
    record: dict,
    *,
    include_questions: bool = False,
    include_answers: bool = True,
    max_questions: int = 2,
    max_answers_per_question: int = 2,
) -> list[ChunkInput]:
    chunks = [averitec_record_to_claim_chunk(record)]
    chunks.extend(
        averitec_record_to_question_chunks(
            record,
            include_questions=include_questions,
            include_answers=include_answers,
            max_questions=max_questions,
            max_answers_per_question=max_answers_per_question,
        )
    )
    return chunks


def averitec_record_to_retrieval_input(
    record: dict,
    *,
    max_questions: int = 2,
    max_answers_per_question: int = 2,
) -> RetrievalInput:
    retrieved_chunks: list[RetrievedChunk] = []
    rank = 1
    for question_item in record.get("questions", [])[:max_questions]:
        question_text = (question_item.get("question") or "").strip()
        for answer_item in question_item.get("answers", [])[:max_answers_per_question]:
            answer_text = (answer_item.get("answer") or "").strip()
            if not answer_text:
                continue
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=f"retrieved-{rank}",
                    text=answer_text,
                    rank=rank,
                    retrieval_score=round(1.0 / rank, 4),
                    source_url=answer_item.get("source_url"),
                    source_medium=answer_item.get("source_medium"),
                    metadata={
                        "question": question_text,
                        "answer_type": answer_item.get("answer_type"),
                        "cached_source_url": answer_item.get("cached_source_url"),
                    },
                )
            )
            rank += 1

    return RetrievalInput(
        sample_id=record["sample_id"],
        query=record["claim"],
        label=record["label"],
        retrieved_chunks=retrieved_chunks,
        metadata={
            "dataset": "averitec",
            "split": record["split"],
            "speaker": record["speaker"],
            "claim_date": record["claim_date"],
            "reporting_source": record["reporting_source"],
        },
    )
