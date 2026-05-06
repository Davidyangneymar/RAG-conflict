# src/dependencies.py
import os
import tempfile
import yaml
import atexit
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from pathlib import Path
from dotenv import load_dotenv

from p1.pipeline import build_pipeline
from p2.pipeline import run_full_p2_pipeline_from_records
from p6 import build_answer_plans

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
dotenv_path = PROJECT_ROOT / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
    logger.info(f"Loaded .env from {dotenv_path}")
else:
    logger.warning(f".env not found at {dotenv_path}")

# default: use Qdrant (dense/hybrid) unless user sets P4_RETRIEVAL_MODE=local_bm25
RETRIEVAL_MODE = os.getenv("P4_RETRIEVAL_MODE", "qdrant").strip().lower()
logger.info(f"P4_RETRIEVAL_MODE = {RETRIEVAL_MODE}")

class AppState:
    def __init__(self):
        self.retrieval_service = None
        self.p1_pipeline = None
        self.p2_func = run_full_p2_pipeline_from_records
        self.p6_func = build_answer_plans
        self.retrieval_mode = "unknown"

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing real modules...")
    state = AppState()

    # 1. P1 pipeline
    try:
        state.p1_pipeline = build_pipeline()
        logger.info("P1 pipeline loaded")
    except Exception as e:
        logger.error(f"Failed to load P1 pipeline: {e}")

    # 2. Qdrant retrieval service (dense/hybrid) by default,
    #    unless user explicitly requested local BM25.
    temp_config_path = None
    if RETRIEVAL_MODE == "local_bm25":
        logger.info("Using local BM25 retrieval (fast start).")
        state.retrieval_mode = "local_bm25"
        state.retrieval_service = None
    else:
        original_config = PROJECT_ROOT / "P3" / "config" / "retrieval.yaml"
        if original_config.exists():
            try:
                with open(original_config, "r", encoding="utf-8") as f:
                    raw_cfg = yaml.safe_load(f)

                # Overwrite paths to point inside P3/
                raw_cfg["chunk_store_path"] = "P3/data/processed/chunks.jsonl"
                raw_cfg["qdrant_path"] = "P3/data/processed/qdrant"
                raw_cfg["bm25_store_path"] = "P3/data/processed/bm25_corpus.json"

                fd, temp_config_path = tempfile.mkstemp(suffix=".yaml", text=True)
                with os.fdopen(fd, "w") as tmp:
                    yaml.dump(raw_cfg, tmp)

                logger.info(f"Created temporary Qdrant config at {temp_config_path}")

                from services.retrieval_service import RetrievalService
                state.retrieval_service = RetrievalService(temp_config_path)
                logger.info("Qdrant retrieval service initialized (dense + hybrid).")
                state.retrieval_mode = "qdrant (dense/hybrid)"

                def cleanup_temp():
                    if temp_config_path and os.path.exists(temp_config_path):
                        os.unlink(temp_config_path)
                        logger.info(f"Removed temporary config {temp_config_path}")
                atexit.register(cleanup_temp)

            except Exception as e:
                logger.error(f"Failed to initialize Qdrant retrieval service: {e}")
                logger.warning("Falling back to local BM25 (Qdrant disabled).")
                state.retrieval_mode = "local_bm25 (fallback from Qdrant error)"
                state.retrieval_service = None
        else:
            logger.warning("Qdrant config not found, using local BM25.")
            state.retrieval_mode = "local_bm25 (config missing)"
            state.retrieval_service = None

    app.state = state
    yield
    logger.info("Shutting down...")

def get_app_state(request: Request) -> AppState:
    return request.app.state