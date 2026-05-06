# P4 - Conflict-Aware RAG Backend API

P4 is the backend API layer of a conflict-aware Retrieval-Augmented Generation (RAG) system. It orchestrates:

- **P3** – retrieval (Qdrant hybrid search or local BM25 fallback)
- **P1** – claim decomposition + natural language inference (NLI)
- **P2** – conflict typing (stance detection model)
- **P6** – answer planning & generation (optional LLM)

It provides REST endpoints for single‑query fact checking and batch FEVER‑style evaluation.

---

## Architecture

The request flow is as follows:

1. **User Request** -> P4 API
2. P3 retrieves relevant evidence
3. P1 decomposes the claim and runs NLI
4. P2 detects conflict type
5. P6 generates the final answer
6. Response returned to user

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- (Optional) Qdrant index built by P3 (`ingest_corpus.py`). If not available, use `local_bm25` mode.

### 2. Install Dependencies

From the **project root** (the directory containing `P3/`, `P4/`, etc.):

    cd /path/to/RAG-conflict-main
    pip install -e P4

If you want to use P1, P2, P3, P6 as editable packages, uncomment the corresponding lines in `P4/pyproject.toml`.

### 3. Environment Variables (`.env`)

Create a `.env` file in the **project root** (the parent directory of `P4/`). Example:

    # HuggingFace mirror for faster download (optional)
    HF_ENDPOINT=https://hf-mirror.com

    # LLM configuration for answer generation (OpenAI-compatible)
    LLM_API_KEY=your-api-key-here
    LLM_BASE_URL=https://api.openai.com/v1
    LLM_MODEL=gpt-3.5-turbo

    # Retrieval mode (default: qdrant)
    P4_RETRIEVAL_MODE=qdrant   # or local_bm25

If `LLM_*` variables are not set, P6 will skip LLM calls and return answers based on the strongest NLI signal (may produce "I don't know").

### 4. Build BM25 Index (only for `local_bm25` mode)

From project root:

    python P4/scripts/build_wiki_bm25.py

The script reads `FEVER Dataset/wiki_pages_matched_sample.jsonl` (relative to project root) and writes `P4/data_backup/bm25_corpus.json`.

### 5. Start the Server

You can run `p4ctl.py` from **any directory** – it will automatically locate the project root and set the correct environment. From the project root or any other location:

    python /path/to/RAG-conflict-main/P4/p4ctl.py start           # uses Qdrant (default)
    python /path/to/RAG-conflict-main/P4/p4ctl.py start local_bm25 # uses local BM25

If you are already inside the project root, you can simply:

    python P4/p4ctl.py start local_bm25

#### Useful p4ctl commands

    # Stop all Python processes and clean Qdrant lock files
    python P4/p4ctl.py stop

    # Manually clean Qdrant lock files (without stopping processes)
    python P4/p4ctl.py clean

    # Show server status (requires server running)
    python P4/p4ctl.py status

> The `stop` command automatically runs the `clean` operation, so you do not need a separate `clean` command after stopping.

#### Alternative: using uvicorn directly

From the project root:

    uvicorn P4.src.main:app --reload

The retrieval mode is controlled by the `P4_RETRIEVAL_MODE` environment variable.

---

## API Endpoints

| Method | Path                           | Description                          | Example Request Body                       |
|--------|--------------------------------|--------------------------------------|--------------------------------------------|
| POST   | `/api/v1/query`                | Single claim fact checking            | `{"text":"...", "top_k":3}`                |
| POST   | `/api/v1/benchmark/export`     | Batch FEVER‑style evaluation          | See `P4/tests/test_p5.json`                |
| GET    | `/health`                      | Health check                          | (none)                                     |

### Example Response (`/api/v1/query`)

    {
      "sample_id": "550e8400-e29b-41d4-a716-446655440000",
      "answer": "SUPPORTS",
      "abstained": false,
      "confidence": 0.87,
      "evidence": [
        "Soul Food is a 1997 American film released by Fox 2000 Pictures."
      ],
      "conflict_type": "none",
      "resolution_policy": "pass_through",
      "retrieval_source": "qdrant"
    }

---

## Testing

### Unit / API Tests

From project root:

    pytest P4/tests/

### Batch Query Test (using `P4/tests/queries.txt`)

    python P4/tests/test_queries.py

Output is written to `P4/tests/queries_output_{retrieval_source}.jsonl`.

### FEVER Benchmark Test

    python P4/tests/test_p5_api.py

Results are saved to `P4/tests/p5_output_from_python.jsonl`.

---

## Troubleshooting

| Issue                                                | Solution                                                                  |
|------------------------------------------------------|---------------------------------------------------------------------------|
| Qdrant lock error: `Storage folder ... is already accessed` | Run `python P4/p4ctl.py stop` (which also cleans locks)          |
| P3 retrieval not available, falling back to local BM25 | Verify `P3/data/processed/qdrant` exists and is not corrupted; re-run P3 ingestion |
| BM25 index not found                                 | Run `python P4/scripts/build_wiki_bm25.py`                                |
| LLM call fails (401/403)                             | Check `LLM_API_KEY`, `LLM_BASE_URL` in `.env`                             |
| Import error for P1/P2/P3/P6                         | Ensure sibling submodules exist and are installed as editable packages (`pip install -e ../P1`, etc.) or set `PYTHONPATH` correctly (handled by `p4ctl.py`) |

---

## Project Structure

    P4/
    ├── config/                -> api_config.yaml
    ├── data_backup/           -> bm25_corpus.json
    ├── scripts/               -> build_wiki_bm25.py, start_server.sh
    ├── src/
    │   ├── dependencies.py    -> app state, lifespan, Qdrant/BM25 init
    │   ├── main.py            -> FastAPI entry point
    │   ├── models/            -> response_models.py
    │   ├── routers/           -> query.py, benchmark.py
    │   └── services/          -> p1_runner, p2_runner, p3_runner, p6_runner, retrieval_client
    ├── tests/                 -> test scripts and sample data
    ├── p4ctl.py               -> command-line control utility
    ├── pyproject.toml
    └── README.md