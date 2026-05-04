from __future__ import annotations

from pathlib import Path

from scripts.check_retrieval_ready import (
    CONFIG_MISMATCH,
    NOT_READY_CORPUS_MISSING,
    READY,
    CheckResult,
    check_corpus,
    choose_status,
    expected_embedding_dim,
)


def test_check_corpus_reports_missing_file(tmp_path: Path) -> None:
    result = check_corpus(tmp_path / "missing.jsonl")

    assert result.ok is False
    assert "not found" in result.message


def test_choose_status_prioritizes_missing_corpus() -> None:
    status = choose_status(
        [CheckResult("corpus", False, "missing")],
        corpus_missing=True,
        collection_points=None,
    )

    assert status == NOT_READY_CORPUS_MISSING


def test_choose_status_reports_ready_when_all_checks_pass() -> None:
    status = choose_status(
        [CheckResult("config", True, "ok"), CheckResult("corpus", True, "ok")],
        corpus_missing=False,
        collection_points=3,
    )

    assert status == READY


def test_choose_status_reports_config_mismatch_for_vector_failure() -> None:
    status = choose_status(
        [CheckResult("vector_size", False, "mismatch")],
        corpus_missing=False,
        collection_points=3,
    )

    assert status == CONFIG_MISMATCH


def test_expected_embedding_dim_uses_hash_fallback_dim() -> None:
    assert expected_embedding_dim({"embedding_backend": "hash", "fallback_embedding_dim": 256}) == 256
