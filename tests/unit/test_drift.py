from __future__ import annotations

import numpy as np
import pytest

from defectlens.serving.drift import DriftMonitor


def _reference(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 1.0, size=500)


def test_no_status_until_the_window_is_full() -> None:
    monitor = DriftMonitor(_reference(), threshold=2.0, window=50)
    for value in np.random.default_rng(1).normal(size=49):
        status = monitor.observe(float(value))
    assert status.ready is False and status.ks is None


def test_same_distribution_raises_no_alert() -> None:
    monitor = DriftMonitor(_reference(), threshold=10.0, window=200)
    for value in np.random.default_rng(2).normal(size=200):
        status = monitor.observe(float(value))
    assert status.ready and status.ks_alert is False and status.rate_alert is False


def test_shifted_scores_trip_the_score_drift_alert() -> None:
    monitor = DriftMonitor(_reference(), threshold=10.0, window=200)
    for value in np.random.default_rng(3).normal(loc=1.5, size=200):
        status = monitor.observe(float(value))
    assert status.ks_alert is True
    assert status.rate_alert is False


def test_a_flag_rate_far_from_the_target_trips_the_decision_alert() -> None:
    # Scores from the reference distribution, but a threshold that flags about half of them.
    monitor = DriftMonitor(_reference(), threshold=0.0, window=200)
    for value in np.random.default_rng(4).normal(size=200):
        status = monitor.observe(float(value))
    assert status.flag_rate is not None and status.flag_rate > 0.3
    assert status.rate_alert is True
    assert status.ks_alert is False


def test_rate_at_the_target_raises_no_decision_alert() -> None:
    reference = _reference()
    threshold = float(np.quantile(reference, 0.99))
    monitor = DriftMonitor(reference, threshold=threshold, target_frr=0.01, window=200)
    for value in np.random.default_rng(5).normal(size=200):
        status = monitor.observe(float(value))
    assert status.rate_alert is False


@pytest.mark.parametrize("reference", [np.array([1.0]), np.array([])])
def test_tiny_reference_is_refused(reference: np.ndarray) -> None:
    with pytest.raises(ValueError, match="reference"):
        DriftMonitor(reference, threshold=1.0)


def test_serving_publishes_the_drift_gauges() -> None:
    import io

    import torch
    from fastapi.testclient import TestClient
    from PIL import Image

    from defectlens.serving.app import create_app

    class _ConstantModel:
        def predict(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            n = tensor.shape[0]
            return torch.full((n,), 0.5), torch.zeros(n, 1, 224, 224)

    monitor = DriftMonitor(_reference(), threshold=0.6, window=5)
    client = TestClient(
        create_app(
            models={"bottle": {"model": _ConstantModel(), "threshold": 0.6, "drift": monitor}}
        )
    )
    buf = io.BytesIO()
    Image.new("RGB", (224, 224)).save(buf, format="PNG")
    for _ in range(5):
        response = client.post(
            "/inspect",
            params={"category": "bottle"},
            files={"file": ("x.png", buf.getvalue(), "image/png")},
        )
        assert response.status_code == 200
    text = client.get("/metrics").text
    assert 'defectlens_flag_rate_window{category="bottle"} 0.0' in text
    assert 'defectlens_drift_alert{category="bottle",kind="decision"} 0.0' in text
    assert 'defectlens_score_drift_ks{category="bottle"}' in text
