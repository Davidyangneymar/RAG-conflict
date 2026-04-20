from __future__ import annotations

import random
from pathlib import Path

from p1.data.fnc1 import read_jsonl, sample_to_retrieval_input
from p1.schemas import RetrievalInput, RetrievedChunk


def build_fnc1_distractor_pool(
    path: str | Path,
    *,
    limit: int = 500,
    body_mode: str = "top2_span",
    stance_labels: set[str] | None = None,
) -> list[str]:
    records = read_jsonl(path)
    distractors: list[str] = []
    allowed = {"unrelated"} if stance_labels is None else {label.strip().lower() for label in stance_labels}
    for record in records:
        stance_label = record.get("stance_label")
        if stance_label not in allowed:
            continue
        retrieval_input = sample_to_retrieval_input(record, body_mode=body_mode)
        for chunk in retrieval_input.retrieved_chunks:
            text = chunk.text.strip()
            if text:
                distractors.append(text)
        if len(distractors) >= limit:
            break
    return distractors[:limit]


def inject_distractor_chunks(
    retrieval_input: RetrievalInput,
    distractor_pool: list[str],
    *,
    distractor_count: int = 2,
    seed: int = 13,
) -> RetrievalInput:
    rng = random.Random(f"{seed}:{retrieval_input.sample_id}")
    chosen = rng.sample(distractor_pool, k=min(distractor_count, len(distractor_pool)))
    retrieved_chunks = list(retrieval_input.retrieved_chunks)
    start_rank = len(retrieved_chunks) + 1
    for offset, text in enumerate(chosen):
        retrieved_chunks.append(
            RetrievedChunk(
                chunk_id=f"noise-{offset + 1}",
                text=text,
                rank=start_rank + offset,
                retrieval_score=0.0,
                source_medium="noise_pool",
                metadata={"role_hint": "distractor", "noise_source": "fnc1_unrelated"},
            )
        )
    return RetrievalInput(
        sample_id=retrieval_input.sample_id,
        query=retrieval_input.query,
        label=retrieval_input.label,
        retrieved_chunks=retrieved_chunks,
        metadata={**retrieval_input.metadata, "stress_mode": "noisy_retrieval"},
    )
