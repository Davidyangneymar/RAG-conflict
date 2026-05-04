"""Check whether local P3 retrieval artifacts are ready for querying."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


READY = "READY"
NOT_READY_CORPUS_MISSING = "NOT_READY_CORPUS_MISSING"
NOT_READY_INDEX_MISSING = "NOT_READY_INDEX_MISSING"
NOT_READY_COLLECTION_EMPTY = "NOT_READY_COLLECTION_EMPTY"
CONFIG_MISMATCH = "CONFIG_MISMATCH"
UNKNOWN_ERROR = "UNKNOWN_ERROR"


@dataclass
class CheckResult:
    """Human-readable readiness check result."""

    name: str
    ok: bool
    message: str
    suggestion: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


def resolve_project_path(value: str | Path, *, root: Path = ROOT) -> Path:
    """Resolve repository-relative paths against the project root."""
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def load_yaml_config(config_path: Path) -> tuple[dict[str, Any], str | None]:
    """Load YAML config without hiding dependency errors."""
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - depends on environment
        return {}, f"PyYAML is not available: {exc}. Run `pip install -e .[dev]` first."

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}, None
    except Exception as exc:
        return {}, f"Failed to read config: {exc}"


def first_config_value(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present config value across compatible key names."""
    for key in keys:
        if key in config and config[key] not in (None, ""):
            return config[key]
    return default


def expected_embedding_dim(config: dict[str, Any]) -> int | None:
    """Infer expected vector dimension when the config declares it safely."""
    explicit = first_config_value(config, "embedding_dim", "vector_size", default=None)
    if explicit is not None:
        return int(explicit)

    backend = str(first_config_value(config, "embedding_backend", default="")).lower()
    if backend == "hash":
        return int(first_config_value(config, "fallback_embedding_dim", default=256))
    return None


def vector_size_from_collection(collection: Any) -> int | None:
    """Read vector size from qdrant-client collection metadata across versions."""
    config = getattr(collection, "config", None)
    params = getattr(config, "params", None)
    vectors = getattr(params, "vectors", None)
    if vectors is None and isinstance(collection, dict):
        vectors = (
            collection.get("config", {})
            .get("params", {})
            .get("vectors")
        )

    if isinstance(vectors, dict):
        if "size" in vectors:
            return int(vectors["size"])
        first_vector = next(iter(vectors.values()), None)
        if isinstance(first_vector, dict) and "size" in first_vector:
            return int(first_vector["size"])
        if hasattr(first_vector, "size"):
            return int(first_vector.size)

    if hasattr(vectors, "size"):
        return int(vectors.size)
    return None


def check_config(config_path: Path) -> tuple[dict[str, Any], CheckResult]:
    if not config_path.exists():
        return {}, CheckResult(
            "config",
            False,
            f"Config file not found: {config_path}",
            "Pass --config-path or run from the P3 project root.",
        )

    config, error = load_yaml_config(config_path)
    if error:
        return {}, CheckResult("config", False, error, "Install project dependencies and retry.")
    return config, CheckResult("config", True, f"Config found: {config_path}")


def check_corpus(corpus_path: Path | None) -> CheckResult:
    if corpus_path is None:
        return CheckResult(
            "corpus",
            False,
            "No corpus path provided.",
            "Pass --corpus-path, for example `FEVER Dataset/wiki_pages_matched_sample.jsonl`.",
        )
    if not corpus_path.exists():
        return CheckResult(
            "corpus",
            False,
            f"Corpus file not found: {corpus_path}",
            "Place the matched wiki JSONL corpus locally before ingestion.",
        )
    if corpus_path.stat().st_size == 0:
        return CheckResult("corpus", False, f"Corpus file is empty: {corpus_path}")
    return CheckResult(
        "corpus",
        True,
        f"Corpus exists and is non-empty: {corpus_path}",
        details={"bytes": corpus_path.stat().st_size},
    )


def check_artifact_file(name: str, path: Path, suggestion: str) -> CheckResult:
    if not path.exists():
        return CheckResult(name, False, f"Missing artifact: {path}", suggestion)
    if path.is_file() and path.stat().st_size == 0:
        return CheckResult(name, False, f"Artifact exists but is empty: {path}", suggestion)
    return CheckResult(name, True, f"Artifact exists: {path}")


def inspect_qdrant(
    *,
    qdrant_path: Path,
    collection_name: str,
    expected_dim: int | None,
) -> tuple[list[CheckResult], int | None]:
    results: list[CheckResult] = []
    point_count: int | None = None

    if not qdrant_path.exists():
        return [
            CheckResult(
                "qdrant_path",
                False,
                f"Qdrant path not found: {qdrant_path}",
                "Run scripts/ingest_corpus.py to build the local vector index.",
            )
        ], point_count

    try:
        from qdrant_client import QdrantClient
    except Exception as exc:  # pragma: no cover - depends on environment
        return [
            CheckResult(
                "qdrant_client",
                False,
                f"qdrant-client is not available: {exc}",
                "Run `pip install -e .[dev]` and retry.",
            )
        ], point_count

    try:
        client = QdrantClient(path=str(qdrant_path))
        if not client.collection_exists(collection_name):
            return [
                CheckResult(
                    "qdrant_collection",
                    False,
                    f"Collection not found: {collection_name}",
                    "Check qdrant_collection_name or rerun ingestion.",
                )
            ], point_count

        results.append(CheckResult("qdrant_collection", True, f"Collection exists: {collection_name}"))

        collection = client.get_collection(collection_name)
        actual_dim = vector_size_from_collection(collection)
        if expected_dim is not None and actual_dim is not None and actual_dim != expected_dim:
            results.append(
                CheckResult(
                    "vector_size",
                    False,
                    f"Vector size mismatch: collection={actual_dim}, config={expected_dim}",
                    "Use a new collection name or rerun ingestion after changing embedding backend/model/dim.",
                    details={"actual_dim": actual_dim, "expected_dim": expected_dim},
                )
            )
        elif actual_dim is not None:
            results.append(
                CheckResult(
                    "vector_size",
                    True,
                    f"Vector size is compatible: {actual_dim}",
                    details={"actual_dim": actual_dim, "expected_dim": expected_dim},
                )
            )
        else:
            results.append(
                CheckResult(
                    "vector_size",
                    True,
                    "Vector size check skipped because qdrant-client did not expose it.",
                    details={"expected_dim": expected_dim},
                )
            )

        count_result = client.count(collection_name=collection_name, exact=True)
        point_count = int(getattr(count_result, "count", 0))
        if point_count <= 0:
            results.append(
                CheckResult(
                    "point_count",
                    False,
                    "Collection exists but contains 0 points.",
                    "Rerun ingestion and confirm corpus rows were loaded.",
                    details={"points": point_count},
                )
            )
        else:
            results.append(
                CheckResult(
                    "point_count",
                    True,
                    f"Collection contains {point_count} points.",
                    details={"points": point_count},
                )
            )
    except Exception as exc:
        results.append(
            CheckResult(
                "qdrant",
                False,
                f"Failed to inspect Qdrant index: {exc}",
                "Check qdrant_path, collection name, and whether another local process is locking the index.",
            )
        )

    return results, point_count


def run_sample_query(config_path: Path, query: str) -> CheckResult:
    try:
        from src.config import load_retrieval_config
        from src.retrieval.pipeline import RetrievalPipeline
        from src.schemas.retrieval import RetrievalQuery

        config = load_retrieval_config(config_path)
        pipeline = RetrievalPipeline.from_artifacts(config)
        response = pipeline.retrieve(RetrievalQuery(query=query, top_k=3, mode=config.default_mode))
        if not response.results:
            return CheckResult(
                "sample_query",
                False,
                "Sample query returned 0 results.",
                "Confirm ingestion completed and BM25/Qdrant artifacts are aligned with config.",
            )
        return CheckResult(
            "sample_query",
            True,
            f"Sample query returned {len(response.results)} result(s).",
            details={"query": query, "top_chunk_id": response.results[0].chunk_id},
        )
    except Exception as exc:
        return CheckResult(
            "sample_query",
            False,
            f"Sample query failed: {exc}",
            "Use static checks above first; then rerun ingestion if artifacts are missing or mismatched.",
        )


def choose_status(results: list[CheckResult], *, corpus_missing: bool, collection_points: int | None) -> str:
    if all(result.ok for result in results):
        return READY
    failed_names = {result.name for result in results if not result.ok}
    if corpus_missing:
        return NOT_READY_CORPUS_MISSING
    if "vector_size" in failed_names:
        return CONFIG_MISMATCH
    if collection_points == 0:
        return NOT_READY_COLLECTION_EMPTY
    if failed_names & {"chunk_store", "bm25_store", "qdrant_path", "qdrant_collection", "point_count"}:
        return NOT_READY_INDEX_MISSING
    return UNKNOWN_ERROR


def result_to_dict(result: CheckResult) -> dict[str, Any]:
    payload = {
        "name": result.name,
        "ok": result.ok,
        "message": result.message,
    }
    if result.suggestion:
        payload["suggestion"] = result.suggestion
    if result.details:
        payload["details"] = result.details
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether local P3 retrieval is ready.")
    parser.add_argument("--config-path", default="config/retrieval.yaml", help="Path to retrieval YAML config.")
    parser.add_argument("--corpus-path", default=None, help="Optional local corpus JSONL path to verify.")
    parser.add_argument("--sample-query", default=None, help="Optional query to run through the retrieval pipeline.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless final status is READY.")
    parser.add_argument("--json-output", default=None, help="Optional path to write structured check results.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_project_path(args.config_path)
    corpus_path = resolve_project_path(args.corpus_path) if args.corpus_path else None

    results: list[CheckResult] = []
    config, config_result = check_config(config_path)
    results.append(config_result)

    corpus_result = check_corpus(corpus_path)
    results.append(corpus_result)

    collection_points: int | None = None
    if config_result.ok:
        qdrant_path = resolve_project_path(
            first_config_value(config, "qdrant_path", "qdrant_dir", default="data/processed/qdrant")
        )
        collection_name = str(
            first_config_value(config, "qdrant_collection_name", "collection_name", default="conflict_aware_rag_dev")
        )
        chunk_store_path = resolve_project_path(
            first_config_value(config, "chunk_store_path", "chunks_path", default="data/processed/chunks.jsonl")
        )
        bm25_store_path = resolve_project_path(
            first_config_value(config, "bm25_store_path", "bm25_path", default="data/processed/bm25_corpus.json")
        )

        results.append(
            check_artifact_file(
                "chunk_store",
                chunk_store_path,
                "Run scripts/ingest_corpus.py to generate chunk records.",
            )
        )
        results.append(
            check_artifact_file(
                "bm25_store",
                bm25_store_path,
                "Run scripts/ingest_corpus.py to generate the sparse corpus.",
            )
        )
        qdrant_results, collection_points = inspect_qdrant(
            qdrant_path=qdrant_path,
            collection_name=collection_name,
            expected_dim=expected_embedding_dim(config),
        )
        results.extend(qdrant_results)

        if args.sample_query:
            results.append(run_sample_query(config_path, args.sample_query))

    status = choose_status(
        results,
        corpus_missing=not corpus_result.ok,
        collection_points=collection_points,
    )
    payload = {
        "status": status,
        "ready": status == READY,
        "checks": [result_to_dict(result) for result in results],
    }

    print(f"P3 retrieval readiness: {status}")
    print("=" * 80)
    for result in results:
        marker = "OK" if result.ok else "FAIL"
        print(f"[{marker}] {result.name}: {result.message}")
        if result.suggestion and not result.ok:
            print(f"      next: {result.suggestion}")
    print("=" * 80)
    if status == READY:
        print("Local retrieval artifacts look ready. You can run retrieval/export commands.")
    else:
        print("Local retrieval is not ready yet. Follow the failed check suggestions above.")

    if args.json_output:
        output_path = resolve_project_path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote JSON readiness report to {output_path}")

    return 0 if status == READY or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())
