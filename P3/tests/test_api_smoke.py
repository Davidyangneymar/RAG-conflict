from pathlib import Path

import pytest
import yaml

pytest.importorskip("fastapi")
pytest.importorskip("qdrant_client")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.app.api import get_retrieval_service, router
from src.config import load_retrieval_config
from src.ingestion.build_index import IndexBuilder
from src.schemas.documents import DocumentRecord
from src.services.retrieval_service import RetrievalService


def _prepare_service(tmp_path: Path) -> RetrievalService:
    config_data = {
        "embedding_backend": "hash",
        "reranker_backend": "heuristic",
        "qdrant_collection_name": "api_test_collection",
        "qdrant_path": str(tmp_path / "qdrant"),
        "chunk_store_path": str(tmp_path / "chunks.jsonl"),
        "bm25_store_path": str(tmp_path / "bm25.json"),
        "chunk_size": 20,
        "chunk_overlap": 5,
        "top_n_before_rerank": 8,
        "top_k_after_rerank": 4,
        "hybrid_alpha": 0.5,
        "max_per_source": 2,
        "max_per_doc": 1,
        "default_mode": "hybrid",
        "enable_rerank": True,
        "enable_diversify": True,
    }
    config_path = tmp_path / "retrieval.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    config = load_retrieval_config(config_path)

    documents = [
        DocumentRecord(
            doc_id="doc-1",
            dataset="toy",
            source_name="Source A",
            title="Foxes",
            full_text="Foxes are omnivorous mammals in the dog family.",
        ),
        DocumentRecord(
            doc_id="doc-2",
            dataset="toy",
            source_name="Source B",
            title="Wolves",
            full_text="Wolves are social canines that hunt in packs.",
        ),
    ]

    IndexBuilder(config).build(documents)
    return RetrievalService(config_path)


def test_api_smoke(tmp_path: Path) -> None:
    app = FastAPI()
    app.include_router(router)
    service = _prepare_service(tmp_path)
    app.dependency_overrides[get_retrieval_service] = lambda: service

    client = TestClient(app)
    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}

    retrieve_response = client.post(
        "/retrieve",
        json={
            "query": "social canines",
            "top_k": 1,
            "mode": "hybrid",
            "use_rerank": True,
            "use_diversify": True,
            "prefer_recent": False,
            "min_unique_sources": 1,
            "max_per_source": 1,
            "filters": {},
        },
    )
    assert retrieve_response.status_code == 200
    payload = retrieve_response.json()
    assert payload["results"][0]["title"] == "Wolves"
