# services/p3_runner.py
import logging
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
from typing import List, Optional

logger = logging.getLogger(__name__)

# ------------------------------
# Local BM25 implementation (fallback)
# ------------------------------
_BM25_INDEX = None

def _load_bm25_index():
    global _BM25_INDEX
    if _BM25_INDEX is None:
        index_path = Path("P3/data/processed/bm25_corpus.json")
        if not index_path.exists():
            index_path = Path("data/processed/bm25_corpus.json")
        if not index_path.exists():
            logger.warning("BM25 index not found")
            return None
        with open(index_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if "chunk_ids" not in data or "tokenized_corpus" not in data:
            logger.error("BM25 index missing required keys")
            return None
        _BM25_INDEX = data
        logger.info(f"BM25 index loaded: {len(data['chunk_ids'])} chunks")
    return _BM25_INDEX

def _tokenize(text: str):
    return re.findall(r'\b[a-z0-9]+\b', text.lower())

def _bm25_score(query_tokens, doc_tokens, avg_doc_len, doc_len, k1=1.5, b=0.75):
    score = 0.0
    term_freq = Counter(doc_tokens)
    for term in set(query_tokens):
        if term not in term_freq:
            continue
        tf = term_freq[term]
        idf = math.log((len(doc_tokens) - tf + 0.5) / (tf + 0.5) + 1.0)
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * doc_len / avg_doc_len)
        score += idf * numerator / denominator
    return score

def _bm25_search(query: str, top_k: int):
    index = _load_bm25_index()
    if index is None:
        return []
    query_tokens = _tokenize(query)
    chunk_ids = index["chunk_ids"]
    tokenized_corpus = index["tokenized_corpus"]
    doc_lengths = [len(doc) for doc in tokenized_corpus]
    avg_doc_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1.0
    scores = []
    for i, doc_tokens in enumerate(tokenized_corpus):
        score = _bm25_score(query_tokens, doc_tokens, avg_doc_len, len(doc_tokens))
        scores.append((score, i))
    scores.sort(key=lambda x: x[0], reverse=True)
    results = []
    for rank, (score, idx) in enumerate(scores[:top_k], start=1):
        results.append(SimpleNamespace(
            chunk_id=chunk_ids[idx],
            text=" ".join(tokenized_corpus[idx]),
            rank=rank,
            retrieval_score=score,
            source_url=None,
            source_medium="bm25_local",
            metadata={"index": idx}
        ))
    return results

# ------------------------------
# Real P3 RetrievalService (if available)
# ------------------------------
_REAL_RETRIEVAL = None

def _init_real_p3():
    global _REAL_RETRIEVAL
    if _REAL_RETRIEVAL is not None:
        return _REAL_RETRIEVAL
    try:
        import sys
        from pathlib import Path
        # Add P3/src to path temporarily
        p3_src = Path(__file__).parent.parent.parent / "P3" / "src"
        if p3_src.exists() and str(p3_src) not in sys.path:
            sys.path.insert(0, str(p3_src))
        from services.retrieval_service import RetrievalService
        config_path = Path(__file__).parent.parent.parent / "P3" / "config" / "retrieval.yaml"
        if config_path.exists():
            _REAL_RETRIEVAL = RetrievalService(str(config_path))
            logger.info("Real P3 retrieval service initialized")
            return _REAL_RETRIEVAL
    except Exception as e:
        logger.warning(f"Real P3 retrieval init failed: {e}")
    return None

def _p3_search(query: str, top_k: int):
    service = _init_real_p3()
    if service is None:
        return None
    try:
        from schemas.retrieval import RetrievalQuery
        req = RetrievalQuery(query=query, top_k=top_k, mode="bm25")  # force bm25
        resp = service.retrieve(req)
        chunks = []
        for ev in resp.results:
            chunks.append(SimpleNamespace(
                chunk_id=ev.chunk_id,
                text=ev.text,
                rank=ev.rank,
                retrieval_score=ev.score_hybrid or ev.score_bm25 or 0.5,
                source_url=ev.source_url,
                source_medium=ev.source_medium or "p3",
                metadata=ev.metadata
            ))
        return chunks
    except Exception as e:
        logger.error(f"P3 search failed: {e}")
        return None

# ------------------------------
# Public API
# ------------------------------
async def retrieve(query: str, top_k: int = 5) -> List[SimpleNamespace]:
    """Main entry point for retrieval. Tries real P3 first, then falls back to local BM25."""
    # Attempt real P3
    chunks = _p3_search(query, top_k)
    if chunks is not None:
        logger.info(f"Retrieved {len(chunks)} chunks via real P3")
        return chunks
    # Fallback to local BM25
    chunks = _bm25_search(query, top_k)
    if chunks:
        logger.info(f"Retrieved {len(chunks)} chunks via local BM25")
        return chunks
    # Ultimate fallback: mock
    logger.warning("No retrieval method available, using mock")
    return _mock_chunks(query, top_k)

def _mock_chunks(query: str, top_k: int):
    return [
        SimpleNamespace(
            chunk_id=f"mock_{i}",
            text=f"Mock evidence for: {query}",
            rank=i+1,
            retrieval_score=0.5,
            source_url=None,
            source_medium="mock",
            metadata={}
        ) for i in range(top_k)
    ]