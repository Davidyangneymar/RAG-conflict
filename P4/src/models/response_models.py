# models/response_models.py
from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    text: str
    top_k: int = 5

class QueryResponse(BaseModel):
    sample_id: str
    answer: str
    abstained: bool
    confidence: float
    evidence: List[str]
    conflict_type: str
    resolution_policy: str
    retrieval_source: Optional[str] = None   # "qdrant" or "bm25"