"""Input and traffic limits for the inspection endpoints.

The limits are chosen so that a single request cannot exhaust memory or the inference slots, and
so that overload is answered with an explicit status code rather than slow silent degradation.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

# Pillow refuses to decode past 2x its own limit, but a 25 MP cap keeps a single decode well inside
# memory on the 16 GB target and matches the largest VisA image with room to spare.
MAX_PIXELS = 25_000_000
MIN_SIDE = 32
MAX_BATCH = 16


class SlidingWindowLimiter:
    """Per-client request budget over a rolling window.

    In-process only: a deployment with several workers needs a shared store to enforce the budget
    across them. The single-process uvicorn deployment in the Dockerfile is what this is sized for.
    """

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._hits: defaultdict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits[key]
            while hits and current - hits[0] >= self._window:
                hits.popleft()
            if len(hits) >= self._max:
                return False
            hits.append(current)
            return True
