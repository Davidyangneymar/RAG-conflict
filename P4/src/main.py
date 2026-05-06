import sys
from pathlib import Path

# ---- Add submodule src directories to sys.path ----
_current_file = Path(__file__).resolve()
P4_ROOT = _current_file.parent.parent
PROJECT_ROOT = P4_ROOT.parent

for sub_name in ['P1', 'P2', 'P3', 'P6']:
    sub_src = PROJECT_ROOT / sub_name / 'src'
    if sub_src.exists() and str(sub_src) not in sys.path:
        sys.path.insert(0, str(sub_src))

for root_name in ['P2']:  
    p_root = PROJECT_ROOT / root_name
    if p_root.exists() and str(p_root) not in sys.path:
        sys.path.insert(0, str(p_root))

# ---- FastAPI app ----
from fastapi import FastAPI
from .routers import query, benchmark
from .dependencies import lifespan

app = FastAPI(title="Conflict-Aware RAG API", version="0.1.0", lifespan=lifespan)
app.include_router(query.router)
app.include_router(benchmark.router)

@app.get("/health")
async def health():
    return {"status": "ok"}