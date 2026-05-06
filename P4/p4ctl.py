#!/usr/bin/env python
"""
P4 command-line control utility (works from any directory, keeps working dir at project root).

Commands:
  start [qdrant|local_bm25]   Start server with optional retrieval mode (default: qdrant)
  stop                        Terminate all Python processes (clean exit) and clean locks.
  clean                       Remove Qdrant lock files without stopping processes.
  status                      Show current mode (if server running).
"""

import os
import sys
import subprocess
from pathlib import Path

def get_project_root() -> Path:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        if (parent / "P3").is_dir() and (parent / "P4").is_dir():
            return parent
    raise RuntimeError("Cannot locate project root (missing P3/ or P4/ directory)")

PROJECT_ROOT = get_project_root()
P4_ROOT = PROJECT_ROOT / "P4"

def setup_environment(mode: str):
    """Set environment variables and ensure project root is in sys.path."""
    os.environ["P4_RETRIEVAL_MODE"] = mode
    # Add project root to PYTHONPATH so that `P4.src.main` can be imported
    pythonpath = os.environ.get("PYTHONPATH", "")
    if str(PROJECT_ROOT) not in pythonpath:
        if pythonpath:
            pythonpath = str(PROJECT_ROOT) + os.pathsep + pythonpath
        else:
            pythonpath = str(PROJECT_ROOT)
        os.environ["PYTHONPATH"] = pythonpath

def start_server(mode: str):
    """Launch uvicorn with project root as working directory."""
    setup_environment(mode)
    # Change working directory to project root (not into P4)
    os.chdir(PROJECT_ROOT)
    print(f"Working directory: {os.getcwd()}")
    print(f"Starting P4 with retrieval mode: {mode}")
    # Use module-style import: P4.src.main:app
    uvicorn_args = [
        sys.executable, "-m", "uvicorn",
        "P4.src.main:app", "--reload"
    ]
    subprocess.run(uvicorn_args)

def stop_all_python():
    print("Stopping all Python processes...")
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/IM", "python.exe"], shell=True, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(["pkill", "-f", "python"], stderr=subprocess.DEVNULL)
    print("All Python processes terminated.")

def clean_qdrant_locks():
    print("Cleaning Qdrant lock files...")
    lock_dirs = [
        PROJECT_ROOT / "P3" / "data" / "processed" / "qdrant",
        PROJECT_ROOT / "data" / "processed" / "qdrant"
    ]
    for d in lock_dirs:
        if d.exists():
            for lock in d.glob("*.lock"):
                lock.unlink()
                print(f"  Removed: {lock}")
    print("Lock files cleaned.")

def show_status():
    try:
        import requests
        r = requests.get("http://localhost:8000/p4/api/v1/status", timeout=2)  # adjust endpoint if needed
        if r.status_code == 200:
            data = r.json()
            print(f"Retrieval mode: {data.get('retrieval_mode', 'unknown')}")
            print(f"P1 loaded: {data.get('p1_pipeline_loaded')}")
            print(f"P2 ready: {data.get('p2_func_ready')}")
            print(f"P6 ready: {data.get('p6_func_ready')}")
            print(f"Service available: {data.get('retrieval_service_available')}")
        else:
            print(f"Status endpoint returned {r.status_code}")
    except Exception as e:
        print(f"Cannot connect to server: {e}")

def main():
    if len(sys.argv) < 2:
        start_server("qdrant")
        return

    cmd = sys.argv[1].lower()
    if cmd == "stop":
        stop_all_python()
        clean_qdrant_locks()
        print("Stop and cleanup completed.")
    elif cmd == "clean":
        clean_qdrant_locks()
    elif cmd == "status":
        show_status()
    elif cmd in ("start", "--mode"):
        mode = "qdrant"
        if len(sys.argv) >= 3:
            arg = sys.argv[2].lower()
            if arg in ("qdrant", "local_bm25"):
                mode = arg
            elif arg.startswith("--") and len(sys.argv) >= 4:
                mode = sys.argv[3].lower()
        start_server(mode)
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)

if __name__ == "__main__":
    main()