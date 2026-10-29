"""Drift monitoring on the anomaly-score stream: is production still looking like the calibration set?

Two signals, per category, over the most recent ``window`` scored images:

- score drift: the two-sample Kolmogorov-Smirnov statistic between the window and the calibration
  scores stored in the artefact. It alerts at the 0.1% level, using the large-sample critical value.
- decision rate: the share of window images above the threshold. The expected share is the target
  false-reject rate, because the threshold was calibrated to reject that share of good parts. It alerts
  when the observed share leaves a band of ``DECISION_Z`` standard errors around that expectation.

Assumption: the mix of good and defective parts in production is roughly stable. A sustained change in
the flag rate is then a reason to investigate, not proof that the model has failed.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

KS_CRITICAL = 1.95  # c(alpha) at alpha = 0.001 for the large-sample two-sample KS test
DECISION_Z = 4.0


@dataclass(frozen=True)
class DriftStatus:
    ready: bool
    ks: float | None
    ks_alert: bool
    flag_rate: float | None
    rate_alert: bool


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Largest gap between the two empirical CDFs; both inputs must be sorted."""
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


class DriftMonitor:
    def __init__(
        self,
        reference: np.ndarray,
        threshold: float,
        target_frr: float = 0.01,
        window: int = 200,
    ) -> None:
        if reference.size < 2:
            raise ValueError("drift monitoring needs at least two reference scores")
        if window < 2:
            raise ValueError("the drift window must hold at least two scores")
        self._reference = np.sort(reference.astype(np.float64).ravel())
        self._threshold = threshold
        self._expected_rate = target_frr
        self._window = window
        self._scores: deque[float] = deque(maxlen=window)

    def observe(self, score: float) -> DriftStatus:
        self._scores.append(score)
        return self.status()

    def status(self) -> DriftStatus:
        if len(self._scores) < self._window:
            return DriftStatus(
                ready=False, ks=None, ks_alert=False, flag_rate=None, rate_alert=False
            )
        window = np.fromiter(self._scores, dtype=np.float64, count=self._window)
        n, m = self._reference.size, window.size
        ks = _ks_statistic(self._reference, np.sort(window))
        ks_critical = KS_CRITICAL * math.sqrt((n + m) / (n * m))
        flag_rate = float(np.mean(window > self._threshold))
        standard_error = math.sqrt(self._expected_rate * (1.0 - self._expected_rate) / m)
        rate_alert = abs(flag_rate - self._expected_rate) > DECISION_Z * standard_error
        return DriftStatus(
            ready=True,
            ks=ks,
            ks_alert=ks > ks_critical,
            flag_rate=flag_rate,
            rate_alert=rate_alert,
        )
