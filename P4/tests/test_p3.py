import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "P3" / "src"))

from services.retrieval_service import RetrievalService
from schemas.retrieval import RetrievalQuery

def main():
    rs = RetrievalService(str(PROJECT_ROOT / "P3" / "config" / "retrieval.yaml"))
    query = "Fox 2000 Pictures released the film Soul Food."
    resp = rs.retrieve(RetrievalQuery(query=query, top_k=3))
    if resp.results:
        print("P3 OK")
        for r in resp.results:
            print(r.text[:100])
    else:
        print("P3 no results")

if __name__ == "__main__":
    main()