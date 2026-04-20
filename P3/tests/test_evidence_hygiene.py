from src.config import RetrievalConfig
from src.schemas.retrieval import RetrievalResponse, RetrievedEvidence
from src.services.evidence_hygiene import apply_evidence_hygiene, assess_evidence_hygiene


def test_evidence_hygiene_penalizes_fragment_like_evidence() -> None:
    config = RetrievalConfig(
        enable_evidence_hygiene=True,
        evidence_hygiene_penalty_weight=0.5,
        evidence_hygiene_skip_threshold=0.85,
        evidence_hygiene_min_tokens=6,
    )
    evidence = RetrievedEvidence(
        query="Did Drake Bell voice Spider-Man?",
        chunk_id="doc-1::chunk-0",
        doc_id="doc-1",
        dataset="toy",
        source_name="Wikipedia",
        text="Spider-Man on Disney XD .",
        score_hybrid=0.9,
        rank=1,
    )

    assessment = assess_evidence_hygiene(evidence, config)

    assert assessment.penalty > 0.0
    assert "ultra_short_non_propositional" in assessment.flags


def test_evidence_hygiene_reranks_cleaner_evidence_first() -> None:
    config = RetrievalConfig(
        enable_evidence_hygiene=True,
        evidence_hygiene_penalty_weight=0.5,
        evidence_hygiene_skip_threshold=0.85,
        evidence_hygiene_min_tokens=6,
    )
    response = RetrievalResponse(
        query="Did Drake Bell voice Spider-Man?",
        mode="hybrid",
        results=[
            RetrievedEvidence(
                query="Did Drake Bell voice Spider-Man?",
                chunk_id="doc-1::chunk-0",
                doc_id="doc-1",
                dataset="toy",
                source_name="Wikipedia",
                text="Spider-Man on Disney XD .",
                score_hybrid=0.9,
                rank=1,
            ),
            RetrievedEvidence(
                query="Did Drake Bell voice Spider-Man?",
                chunk_id="doc-1::chunk-1",
                doc_id="doc-1",
                dataset="toy",
                source_name="Wikipedia",
                text="Bell was the voice of Peter Parker / Spider-Man in the animated series Ultimate Spider-Man on Disney XD .",
                score_hybrid=0.8,
                rank=2,
            ),
        ],
    )

    reranked = apply_evidence_hygiene(response, config, top_k=2)

    assert reranked.results[0].chunk_id == "doc-1::chunk-1"
    assert reranked.results[0].metadata["hygiene_penalty"] < reranked.results[1].metadata["hygiene_penalty"]
