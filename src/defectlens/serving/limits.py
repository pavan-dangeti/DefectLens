"""Input and traffic limits for the inspection endpoints.

The limits are chosen so that a single request cannot exhaust memory or the inference slots, and
so that overload is answered with an explicit status code rather than slow silent degradation.
"""

from __future__ import annotations

import ipaddress
import threading
import time
from collections import deque

# Pillow refuses to decode past 2x its own limit, but a 25 MP cap keeps a single decode well inside
# memory on the 16 GB target and matches the largest VisA image with room to spare.
MAX_PIXELS = 25_000_000
MIN_SIDE = 32
MAX_BATCH = 16


def client_key(host: str) -> str:
    """Bucket IPv6 clients by /64, since one host commonly controls a whole /64 of addresses."""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host
    if isinstance(address, ipaddress.IPv6Address):
        return str(ipaddress.ip_network(f"{address}/64", strict=False))
    return str(address)


class SlidingWindowLimiter:
    """Per-client budget of image units over a rolling window.

    A batch is charged once per image, so a 16-image request costs 16 units and cannot be used to
    get around a per-request budget. Keys with no recent activity are dropped, so the map stays
    bounded by the number of clients active within one window.

    In-process only: a deployment with several workers needs a shared store to enforce the budget
    across them. The single-process uvicorn deployment in the Dockerfile is what this is sized for.
    """

    _MIN_SWEEP_AT = 4096

    def __init__(self, max_units: int, window_seconds: float, max_keys: int = 100_000) -> None:
        self._max = max_units
        self._window = window_seconds
        self._max_keys = max_keys
        self._hits: dict[str, deque[float]] = {}
        # Sweeping on every call would make each request O(map size) once the map is large.
        # Sweeping when the map has doubled since the last sweep keeps the cost amortised O(1).
        self._next_sweep = self._MIN_SWEEP_AT
        self._lock = threading.Lock()

    def allow(self, key: str, cost: int = 1, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        with self._lock:
            if len(self._hits) >= self._next_sweep:
                self._sweep(current)
            if key not in self._hits and len(self._hits) >= self._max_keys:
                # Fail closed: a flood of new keys must not grow memory without limit.
                return False
            hits = self._hits.setdefault(key, deque())
            self._expire(hits, current)
            if len(hits) + cost > self._max:
                if not hits:
                    del self._hits[key]
                return False
            hits.extend([current] * cost)
            return True

    def exhausted(self, key: str, now: float | None = None) -> bool:
        """True when the key has no budget left, without spending any."""
        current = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                return False
            self._expire(hits, current)
            if not hits:
                del self._hits[key]
                return False
            return len(hits) >= self._max

    def _expire(self, hits: deque[float], current: float) -> None:
        while hits and current - hits[0] >= self._window:
            hits.popleft()

    def _sweep(self, current: float) -> None:
        for key in list(self._hits):
            hits = self._hits[key]
            self._expire(hits, current)
            if not hits:
                del self._hits[key]
        self._next_sweep = max(self._MIN_SWEEP_AT, 2 * len(self._hits))
