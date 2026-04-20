from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the local P1 runtime for Phase 1.1.")
    parser.add_argument("--require-accelerator", action="store_true")
    return parser.parse_args()


def import_status(module_name: str) -> dict[str, str | bool | None]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {
            "available": False,
            "version": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "available": True,
        "version": str(getattr(module, "__version__", "unknown")),
        "error": None,
    }


def torch_status() -> dict[str, object]:
    status = import_status("torch")
    if not status["available"]:
        return status

    import torch

    cuda_available = bool(torch.cuda.is_available())
    mps_available = bool(
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    )
    accelerator = "cuda" if cuda_available else "mps" if mps_available else "cpu"
    status.update(
        {
            "cuda_available": cuda_available,
            "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
            "mps_available": mps_available,
            "recommended_device": accelerator,
        }
    )
    return status


def main() -> None:
    args = parse_args()
    torch = torch_status()
    payload = {
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "packages": {
            "torch": torch,
            "transformers": import_status("transformers"),
            "sentence_transformers": import_status("sentence_transformers"),
            "spacy": import_status("spacy"),
        },
        "phase_1_1_ready": bool(
            torch.get("available")
            and import_status("transformers")["available"]
            and import_status("sentence_transformers")["available"]
        ),
        "accelerator_available": bool(
            torch.get("cuda_available") or torch.get("mps_available")
        ),
    }
    if args.require_accelerator and not payload["accelerator_available"]:
        payload["phase_1_1_ready"] = False
        payload["error"] = "accelerator_required_but_not_available"
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
