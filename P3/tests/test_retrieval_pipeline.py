from pathlib import Path

import pytest
import yaml

pytest.importorskip("qdrant_client")

from src.config import RetrievalConfig, load_retrieval_config
from src.ingestion.build_index import IndexBuilder
from src.retrieval.pipeline import RetrievalPipeline
from src.schemas.documents import DocumentRecord
from src.schemas.retrieval import RetrievalQuery


def _write_test_config(tmp_path: Path) -> Path:
    config_data = {
        "embedding_backend": "hash",
        "reranker_backend": "heuristic",
        "qdrant_collection_name": "test_collection",
        "qdrant_path": str(tmp_path / "qdrant"),
        "chunk_store_path": str(tmp_path / "chunks.jsonl"),
        "bm25_store_path": str(tmp_path / "bm25.json"),
        "chunk_size": 20,
        "chunk_overlap": 5,
        "top_n_before_rerank": 8,
        "top_k_after_rerank": 4,
        "hybrid_alpha": 0.5,
        "max_per_source": 1,
        "max_per_doc": 1,
        "default_mode": "hybrid",
        "enable_rerank": True,
        "enable_diversify": True,
    }
    config_path = tmp_path / "retrieval.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    return config_path


def test_retrieval_pipeline_smoke(tmp_path: Path) -> None:
    config_path = _write_test_config(tmp_path)
    config: RetrievalConfig = load_retrieval_config(config_path)

    documents = [
        DocumentRecord(
            doc_id="doc-cats",
            dataset="toy",
            source_name="Source A",
            title="Cats",
            full_text="Cats hunt mice and live with humans in many homes.",
        ),
        DocumentRecord(
            doc_id="doc-dogs",
            dataset="toy",
            source_name="Source B",
            title="Dogs",
            full_text="Dogs are domesticated animals and often live with humans.",
        ),
        DocumentRecord(
            doc_id="doc-lions",
            dataset="toy",
            source_name="Source C",
            title="Lions",
            full_text="Lions are wild cats that hunt prey in groups.",
        ),
    ]

    IndexBuilder(config).build(documents)
    pipeline = RetrievalPipeline.from_artifacts(config)
    response = pipeline.retrieve(
        RetrievalQuery(
            query="cats hunt prey",
            top_k=2,
            mode="hybrid",
            min_unique_sources=2,
            max_per_source=1,
        )
    )

    assert len(response.results) == 2
    assert response.results[0].title in {"Cats", "Lions"}
    assert len({result.source_name for result in response.results}) == 2
