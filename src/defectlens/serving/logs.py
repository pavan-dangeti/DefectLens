"""Structured JSON request logs.

One line per request, carrying the request ID that is also returned in the ``X-Request-ID``
header, so a client report can be matched to the server log. Client addresses are deliberately
not logged: they are personal data and nothing in the inspection path needs them.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

ACCESS_LOGGER = "defectlens.access"
_EXTRA_FIELDS = ("request_id", "method", "route", "status", "duration_ms")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for field in _EXTRA_FIELDS:
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        return json.dumps(payload, default=str)


def configure_json_logging() -> None:
    """Send the access log to stderr as JSON. Idempotent, so repeated calls add no handlers."""
    logger = logging.getLogger(ACCESS_LOGGER)
    if any(isinstance(h.formatter, JsonFormatter) for h in logger.handlers):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
