from __future__ import annotations

import json
import logging
import re
from typing import Any


SAFE_VALUE_RE = re.compile(r"^[A-Za-z0-9_.:/@-]+$")


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    payload = " ".join(
        f"{key}={_format_value(value)}"
        for key, value in fields.items()
        if value is not None
    )
    message = f"event={event}"
    if payload:
        message = f"{message} {payload}"
    logger.log(level, message)


def elapsed_ms(started_at: float, ended_at: float) -> float:
    return round((ended_at - started_at) * 1000, 3)


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    text = str(value)
    if SAFE_VALUE_RE.fullmatch(text):
        return text
    return json.dumps(text, ensure_ascii=False)
