from src.schemas.documents import ClaimRecord
from src.schemas.retrieval import RetrievalResponse, RetrievedEvidence
from src.services.handoff_adapter import claim_and_response_to_p1_record, retrieval_response_to_p1_record


def test_retrieval_response_to_p1_record_preserves_contract_fields() -> None:
    response = RetrievalResponse(
        query="Fox 2000 Pictures released the film Soul Food.",
        mode="hybrid",
        results=[
            RetrievedEvidence(
                query="Fox 2000 Pictures released the film Soul Food.",
                chunk_id="doc-1::chunk-0",
                doc_id="doc-1",
                dataset="fever_wiki_sample",
                source_name="Wikipedia",
                source_url="https://example.org/doc-1",
                title="Soul Food",
                published_at=None,
                text="Soul Food was released by Fox 2000 Pictures.",
                score_hybrid=0.87,
                rank=1,
                metadata={"language": "en"},
            )
        ],
    )

    record = retrieval_response_to_p1_record(
        response,
        sample_id="137334",
        label="SUPPORTS",
        metadata={"dataset": "fever", "split": "dev"},
    )

    assert record.sample_id == "137334"
    assert record.query == response.query
    assert record.retrieved_chunks[0].chunk_id == "doc-1::chunk-0"
    assert record.retrieved_chunks[0].retrieval_score == 0.87
    assert record.retrieved_chunks[0].source_medium == "Wikipedia"
    assert record.retrieved_chunks[0].metadata["title"] == "Soul Food"
    assert record.retrieved_chunks[0].metadata["source_doc_id"] == "doc-1"


def test_claim_and_response_to_p1_record_uses_claim_metadata() -> None:
    claim = ClaimRecord(
        claim_id="111897",
        dataset="fever",
        query="Telemundo is a English-language television network.",
        label="REFUTES",
    )
    response = RetrievalResponse(query=claim.query, mode="hybrid", results=[])

    record = claim_and_response_to_p1_record(claim, response, split="dev")

    assert record.sample_id == "111897"
    assert record.label == "REFUTES"
    assert record.metadata["dataset"] == "fever"
    assert record.metadata["split"] == "dev"


def test_p1_handoff_contract_payload_contains_required_fields() -> None:
    response = RetrievalResponse(
        query="Fox 2000 Pictures released the film Soul Food.",
        mode="hybrid",
        results=[
            RetrievedEvidence(
                query="Fox 2000 Pictures released the film Soul Food.",
                chunk_id="Soul_Food_-LRB-film-RRB-::chunk-0",
                doc_id="Soul_Food_-LRB-film-RRB-",
                dataset="fever_wiki_sample",
                source_name="Wikipedia",
                source_url="https://en.wikipedia.org/wiki/Soul_Food_(film)",
                title="Soul Food (film)",
                published_at=None,
                text="Soul Food was released by Fox 2000 Pictures.",
                score_sparse=0.31,
                score_dense=0.52,
                score_hybrid=0.81,
                score_rerank=0.93,
                rank=1,
                metadata={"language": "en"},
            )
        ],
    )

    payload = retrieval_response_to_p1_record(
        response,
        sample_id="137334",
        label="SUPPORTS",
        metadata={"dataset": "fever", "split": "dev"},
    ).model_dump()

    assert "sample_id" in payload
    assert "query" in payload
    assert "retrieved_chunks" in payload
    assert isinstance(payload["retrieved_chunks"], list)
    assert len(payload["retrieved_chunks"]) == 1

    chunk = payload["retrieved_chunks"][0]
    assert "chunk_id" in chunk
    assert "text" in chunk
    assert "retrieval_score" in chunk
    assert "source_url" in chunk
    assert "metadata" in chunk
    assert "title" in chunk["metadata"]
    assert "source_doc_id" in chunk["metadata"]

    assert payload["sample_id"] == "137334"
    assert payload["query"] == response.query
    assert chunk["chunk_id"] == "Soul_Food_-LRB-film-RRB-::chunk-0"
    assert chunk["retrieval_score"] == 0.93
    assert chunk["source_url"] == "https://en.wikipedia.org/wiki/Soul_Food_(film)"
    assert chunk["metadata"]["title"] == "Soul Food (film)"
    assert chunk["metadata"]["source_doc_id"] == "Soul_Food_-LRB-film-RRB-"


def test_p1_handoff_contract_survives_reranker_ablation_metadata() -> None:
    response = RetrievalResponse(
        query="Does coffee cause cancer?",
        mode="hybrid",
        results=[
            RetrievedEvidence(
                query="Does coffee cause cancer?",
                chunk_id="doc-bge::chunk-0",
                doc_id="doc-bge",
                dataset="fever_wiki_sample",
                source_name="Wikipedia",
                source_url="https://example.org/bge",
                title="Coffee",
                text="Coffee has been linked to reduced cancer risk.",
                score_hybrid=0.42,
                score_rerank=0.99,
                rank=1,
                metadata={"reranker_backend": "bge", "pre_rerank_rank": 3},
            ),
            RetrievedEvidence(
                query="Does coffee cause cancer?",
                chunk_id="doc-none::chunk-0",
                doc_id="doc-none",
                dataset="fever_wiki_sample",
                source_name="Wikipedia",
                source_url="https://example.org/none",
                title="Coffee baseline",
                text="Coffee was reviewed by health agencies.",
                score_hybrid=0.41,
                rank=2,
                metadata={"reranker_backend": "none", "pre_rerank_rank": 2},
            ),
        ],
    )

    payload = retrieval_response_to_p1_record(response, sample_id="coffee-1").model_dump()

    assert payload["sample_id"] == "coffee-1"
    assert payload["query"] == "Does coffee cause cancer?"
    assert len(payload["retrieved_chunks"]) == 2
    for chunk in payload["retrieved_chunks"]:
        assert "chunk_id" in chunk
        assert "text" in chunk
        assert "retrieval_score" in chunk
        assert "source_url" in chunk
        assert "metadata" in chunk
        assert "title" in chunk["metadata"]
        assert "source_doc_id" in chunk["metadata"]

    assert payload["retrieved_chunks"][0]["retrieval_score"] == 0.99
    assert payload["retrieved_chunks"][1]["retrieval_score"] == 0.41
    assert payload["retrieved_chunks"][0]["metadata"]["reranker_backend"] == "bge"
    assert payload["retrieved_chunks"][1]["metadata"]["reranker_backend"] == "none"
