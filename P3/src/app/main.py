"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from src.app.api import router
from src.config import load_retrieval_config
from src.utils.logging import setup_logging

config = load_retrieval_config()
setup_logging(config.log_level)

app = FastAPI(title="Conflict-Aware RAG Retrieval API", version="0.1.0")
app.include_router(router)
