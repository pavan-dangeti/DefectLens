"""Prometheus metrics for the serving surface, kept on a registry of their own.

A private registry means the test apps and the real app never collide on metric names, and
``/metrics`` exposes exactly what this service records and nothing from the process defaults.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry()

REQUESTS = Counter(
    "defectlens_requests_total",
    "HTTP requests by route and status code.",
    ["route", "status"],
    registry=REGISTRY,
)
REQUEST_SECONDS = Histogram(
    "defectlens_request_seconds",
    "HTTP request duration by route.",
    ["route"],
    registry=REGISTRY,
)
INFERENCE_SECONDS = Histogram(
    "defectlens_inference_seconds",
    "Per-image model forward and scoring time.",
    ["category"],
    registry=REGISTRY,
)
DECISIONS = Counter(
    "defectlens_decisions_total",
    "Decisions returned, by category.",
    ["category", "decision"],
    registry=REGISTRY,
)
# Score distribution is the input to drift alerts, so its buckets must cover the range seen in
# calibration as well as the tail above threshold.
ANOMALY_SCORE = Histogram(
    "defectlens_anomaly_score",
    "Image anomaly scores, by category.",
    ["category"],
    buckets=(5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100, 150, 200),
    registry=REGISTRY,
)
LOW_CONFIDENCE = Counter(
    "defectlens_low_confidence_total",
    "Inputs flagged low confidence, by reason.",
    ["reason"],
    registry=REGISTRY,
)
REJECTED = Counter(
    "defectlens_rejected_total",
    "Inputs refused before scoring, by reason.",
    ["reason"],
    registry=REGISTRY,
)
MODELS_LOADED = Gauge(
    "defectlens_models_loaded",
    "Models loaded and ready to score.",
    registry=REGISTRY,
)
DRIFT_KS = Gauge(
    "defectlens_score_drift_ks",
    "Kolmogorov-Smirnov statistic of recent anomaly scores against the calibration scores.",
    ["category"],
    registry=REGISTRY,
)
FLAG_RATE_WINDOW = Gauge(
    "defectlens_flag_rate_window",
    "Share of the most recent images flagged as anomalous.",
    ["category"],
    registry=REGISTRY,
)
DRIFT_ALERT = Gauge(
    "defectlens_drift_alert",
    "1 while a drift alert is active, by kind: score (KS) or decision (flag rate).",
    ["category", "kind"],
    registry=REGISTRY,
)
REVIEW_PENDING = Gauge(
    "review_cases_pending",
    "Flagged cases awaiting a human review label.",
    registry=REGISTRY,
)
