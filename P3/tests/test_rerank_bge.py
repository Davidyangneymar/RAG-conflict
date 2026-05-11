"""Mock-based tests for the BGE reranker backend."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from src.config import RetrievalConfig
from src.retrieval.rerank import Reranker
from src.schemas.retrieval import RetrievedEvidence


def _make_candidates(n: int) -> list[RetrievedEvidence]:
    return [
        RetrievedEvidence(
            query="test query",
            chunk_id=f"doc-{i}::chunk-0",
            doc_id=f"doc-{i}",
            dataset="toy",
            source_name=f"Source {i}",
            text=f"Candidate text number {i}",
            rank=i + 1,
            metadata={},
        )
        for i in range(n)
    ]


def _mock_flag_module(scores):
    """Create a mock FlagEmbedding module with configurable scores."""
    mock_module = ModuleType("FlagEmbedding")
    mock_reranker_cls = MagicMock()
    mock_instance = MagicMock()
    mock_instance.compute_score.return_value = scores
    mock_reranker_cls.return_value = mock_instance
    mock_module.FlagReranker = mock_reranker_cls
    return mock_module


def test_bge_reranker_sorts_by_score():
    mock_module = _mock_flag_module([0.3, 0.9, 0.6])

    with patch.dict(sys.modules, {"FlagEmbedding": mock_module}):
        config = RetrievalConfig(reranker_backend="bge")
        reranker = Reranker(config)
        results = reranker.rerank("test query", _make_candidates(3), top_k=3)

    assert len(results) == 3
    assert results[0].score_rerank == 0.9
    assert results[1].score_rerank == 0.6
    assert results[2].score_rerank == 0.3
    assert results[0].rank == 1
    assert results[1].rank == 2
    assert results[2].rank == 3


def test_bge_reranker_can_reorder_candidates():
    mock_module = _mock_flag_module([0.1, 0.2, 0.95])

    with patch.dict(sys.modules, {"FlagEmbedding": mock_module}):
        config = RetrievalConfig(reranker_backend="bge")
        reranker = Reranker(config)
        results = reranker.rerank("test query", _make_candidates(3), top_k=3)

    assert [result.chunk_id for result in results] == [
        "doc-2::chunk-0",
        "doc-1::chunk-0",
        "doc-0::chunk-0",
    ]
    assert results[0].metadata["pre_rerank_rank"] == 3
    assert results[0].metadata["reranker_backend"] == "bge"


def test_none_reranker_preserves_candidate_order():
    config = RetrievalConfig(reranker_backend="none")
    reranker = Reranker(config)
    candidates = _make_candidates(3)
    candidates[0].score_hybrid = 0.80
    candidates[1].score_hybrid = 0.70
    candidates[2].score_hybrid = 0.60

    results = reranker.rerank("test query", candidates, top_k=3)

    assert [result.chunk_id for result in results] == [
        "doc-0::chunk-0",
        "doc-1::chunk-0",
        "doc-2::chunk-0",
    ]
    assert [result.rank for result in results] == [1, 2, 3]
    assert [result.score_hybrid for result in results] == [0.80, 0.70, 0.60]
    assert all(result.score_rerank is None for result in results)
    assert all(result.metadata["reranker_backend"] == "none" for result in results)
    assert all(result.metadata["reranker_model"] is None for result in results)
    assert all(result.metadata["reranker_backend"] != "bge" for result in results)


def test_bge_reranker_top_k_truncation():
    mock_module = _mock_flag_module([0.3, 0.9, 0.6])

    with patch.dict(sys.modules, {"FlagEmbedding": mock_module}):
        config = RetrievalConfig(reranker_backend="bge")
        reranker = Reranker(config)
        results = reranker.rerank("test query", _make_candidates(3), top_k=2)

    assert len(results) == 2
    assert results[0].score_rerank == 0.9
    assert results[1].score_rerank == 0.6


def test_bge_reranker_metadata_enrichment():
    mock_module = _mock_flag_module([0.3, 0.9, 0.6])

    with patch.dict(sys.modules, {"FlagEmbedding": mock_module}):
        config = RetrievalConfig(reranker_backend="bge")
        reranker = Reranker(config)
        results = reranker.rerank("test query", _make_candidates(3), top_k=3)

    for result in results:
        assert result.metadata["reranker_backend"] == "bge"
        assert result.metadata["reranker_model"] == "BAAI/bge-reranker-v2-m3"
        assert "pre_rerank_rank" in result.metadata
        assert "rerank_score" in result.metadata
        assert "fallback_reason" not in result.metadata

    top_result = results[0]
    assert top_result.metadata["pre_rerank_rank"] == 2
    assert top_result.metadata["rerank_score"] == 0.9


def test_bge_reranker_fallback_on_import_error():
    config = RetrievalConfig(reranker_backend="bge", allow_model_fallback=True)
    reranker = Reranker(config)

    with patch.dict(sys.modules, {"FlagEmbedding": None}):
        reranker._bge_model = None
        results = reranker.rerank("cats hunt mice", _make_candidates(3), top_k=3)

    assert len(results) == 3
    assert config.reranker_backend == "heuristic"
    assert all(r.score_rerank is not None for r in results)
    for r in results:
        assert r.metadata["reranker_backend"] == "heuristic_fallback"
        assert r.metadata["reranker_model"] == "heuristic"
        assert r.metadata["requested_reranker_model"] == "BAAI/bge-reranker-v2-m3"
        assert r.metadata["fallback_reason"] == "bge_load_failed"


def test_bge_reranker_no_fallback_raises():
    config = RetrievalConfig(reranker_backend="bge", allow_model_fallback=False)
    reranker = Reranker(config)

    with patch.dict(sys.modules, {"FlagEmbedding": None}):
        reranker._bge_model = None
        with pytest.raises(Exception):
            reranker.rerank("test query", _make_candidates(3), top_k=3)


def test_bge_reranker_inference_failure_fallback():
    """Model loads but compute_score raises — should fallback to heuristic."""
    mock_module = _mock_flag_module([0.5, 0.5, 0.5])
    mock_instance = mock_module.FlagReranker.return_value
    mock_instance.compute_score.side_effect = RuntimeError("tokenizer error")

    with patch.dict(sys.modules, {"FlagEmbedding": mock_module}):
        config = RetrievalConfig(reranker_backend="bge", allow_model_fallback=True)
        reranker = Reranker(config)
        results = reranker.rerank("cats hunt mice", _make_candidates(3), top_k=3)

    assert len(results) == 3
    assert config.reranker_backend == "heuristic"
    assert all(r.score_rerank is not None for r in results)
    for r in results:
        assert r.metadata["reranker_backend"] == "heuristic_fallback"
        assert r.metadata["reranker_model"] == "heuristic"
        assert r.metadata["requested_reranker_model"] == "BAAI/bge-reranker-v2-m3"
        assert r.metadata["fallback_reason"] == "bge_inference_failed"


def test_bge_reranker_inference_failure_no_fallback_raises():
    """Model loads but compute_score raises — should raise if fallback disabled."""
    mock_module = _mock_flag_module([0.5, 0.5, 0.5])
    mock_instance = mock_module.FlagReranker.return_value
    mock_instance.compute_score.side_effect = RuntimeError("tokenizer error")

    with patch.dict(sys.modules, {"FlagEmbedding": mock_module}):
        config = RetrievalConfig(reranker_backend="bge", allow_model_fallback=False)
        reranker = Reranker(config)
        with pytest.raises(RuntimeError, match="tokenizer error"):
            reranker.rerank("test query", _make_candidates(3), top_k=3)


def test_bge_reranker_single_candidate():
    mock_module = _mock_flag_module(0.85)

    with patch.dict(sys.modules, {"FlagEmbedding": mock_module}):
        config = RetrievalConfig(reranker_backend="bge")
        reranker = Reranker(config)
        results = reranker.rerank("test query", _make_candidates(1), top_k=1)

    assert len(results) == 1
    assert results[0].score_rerank == 0.85
    assert results[0].rank == 1
    assert results[0].metadata["reranker_backend"] == "bge"


def test_bge_reranker_empty_candidates():
    config = RetrievalConfig(reranker_backend="bge")
    reranker = Reranker(config)
    results = reranker.rerank("test query", [], top_k=5)
    assert results == []


def test_heuristic_backend_adds_metadata():
    config = RetrievalConfig(reranker_backend="heuristic")
    reranker = Reranker(config)
    results = reranker.rerank("cats hunt mice", _make_candidates(2), top_k=2)

    for result in results:
        assert result.metadata["reranker_backend"] == "heuristic"
        assert result.metadata["reranker_model"] == "heuristic"
        assert "pre_rerank_rank" in result.metadata
        assert "rerank_score" in result.metadata
        assert "fallback_reason" not in result.metadata
