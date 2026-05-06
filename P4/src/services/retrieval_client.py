# services/retrieval_client.py
import asyncio
import json
import logging
import math
import re
from pathlib import Path
from types import SimpleNamespace
from collections import Counter

logger = logging.getLogger(__name__)

# ---------- Local BM25 implementation (fallback) ----------
_BM25_INDEX = None

def _load_bm25_index():
    """Load BM25 index from predefined locations, preferring P4/data_backup."""
    global _BM25_INDEX
    if _BM25_INDEX is None:
        # Priority 1: P4/data_backup/bm25_corpus.json (user-built Wikipedia index)
        index_path = Path(__file__).parent.parent / "data_backup" / "bm25_corpus.json"
        # Priority 2: project root data/processed
        if not index_path.exists():
            index_path = Path("data/processed/bm25_corpus.json")
        # Priority 3: P3/data/processed
        if not index_path.exists():
            index_path = Path("P3/data/processed/bm25_corpus.json")
        if not index_path.exists():
            logger.warning("BM25 index not found, local BM25 will be unavailable")
            return None
        with open(index_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if "chunk_ids" not in data or "tokenized_corpus" not in data:
            logger.error("BM25 index missing required keys")
            return None
        _BM25_INDEX = data
        logger.info(f"Loaded BM25 index with {len(data['chunk_ids'])} chunks")
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

async def _local_bm25_search(query: str, top_k: int) -> list:
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

    chunks = []
    for rank, (score, idx) in enumerate(scores[:top_k], start=1):
        text = " ".join(tokenized_corpus[idx])
        chunk = SimpleNamespace(
            chunk_id=chunk_ids[idx],
            text=text,
            rank=rank,
            retrieval_score=score,
            source_url=None,
            source_medium="bm25_local",
            metadata={"index": idx}
        )
        chunks.append(chunk)
    return chunks

# ---------- Main retrieval function (prioritizes P3 service) ----------
async def retrieve_chunks(query: str, top_k: int = 5, retrieval_service=None) -> list:
    """
    Try to use P3 retrieval service if available, otherwise fallback to local BM25.
    """
    if retrieval_service is not None:
        try:
            from schemas.retrieval import RetrievalQuery
            req = RetrievalQuery(
                query=query,
                top_k=top_k,
                mode="hybrid",          # or "bm25" to avoid dense issues
                use_rerank=True,
                use_diversify=True,
                prefer_recent=False,
                min_unique_sources=2,
                max_per_source=2,
            )
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, retrieval_service.retrieve, req)
            chunks = []
            for ev in response.results:
                # Safely extract score with fallback chain
                score = (
                    getattr(ev, 'score_hybrid', None) or
                    getattr(ev, 'score_sparse', None) or
                    getattr(ev, 'score_dense', None) or
                    0.0
                )
                source_medium = getattr(ev, 'source_name', None)
                ns = SimpleNamespace(
                    chunk_id=getattr(ev, 'chunk_id', ''),
                    text=getattr(ev, 'text', ''),
                    rank=getattr(ev, 'rank', 0),
                    retrieval_score=score,
                    source_url=getattr(ev, 'source_url', None),
                    source_medium=source_medium,
                    metadata=getattr(ev, 'metadata', {}),
                )
                chunks.append(ns)
            logger.info(f"Retrieved {len(chunks)} chunks via P3 service")
            return chunks
        except Exception as e:
            logger.error(f"P3 retrieval failed: {e}, falling back to local BM25")
    # Fallback to local BM25
    logger.warning("Using local BM25 retrieval")
    return await _local_bm25_search(query, top_k)