from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from defectlens.data.splits import make_splits


def test_splits_are_disjoint_and_total() -> None:
    s = make_splits(100, seed=42)
    assert set(s.fit).isdisjoint(s.calibration)
    assert set(s.fit).isdisjoint(s.holdout)
    assert set(s.calibration).isdisjoint(s.holdout)
    assert sorted(s.fit + s.calibration + s.holdout) == list(range(100))


def test_splits_are_deterministic() -> None:
    assert make_splits(50, seed=7) == make_splits(50, seed=7)


def test_different_seeds_give_different_membership() -> None:
    a = make_splits(50, seed=1)
    b = make_splits(50, seed=2)
    assert a.fit != b.fit or a.calibration != b.calibration


def test_ratios_are_approximately_respected() -> None:
    s = make_splits(1000, seed=3)
    assert abs(len(s.fit) / 1000 - 0.8) < 0.05
    assert abs(len(s.calibration) / 1000 - 0.1) < 0.05
    assert abs(len(s.holdout) / 1000 - 0.1) < 0.05


def test_too_few_items_raises() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        make_splits(2, seed=1)


def test_few_shot_draws_only_from_fit() -> None:
    s = make_splits(200, seed=11)
    few_shot_indices = s.fit[:5]
    assert not (set(few_shot_indices) & set(s.calibration))
    assert not (set(few_shot_indices) & set(s.holdout))


@settings(max_examples=50, deadline=None)
@given(n=st.integers(min_value=3, max_value=500), seed=st.integers(min_value=0, max_value=10_000))
def test_splits_invariant_for_all_valid_inputs(n: int, seed: int) -> None:
    s = make_splits(n, seed)
    combined = s.fit + s.calibration + s.holdout
    assert len(combined) == n
    assert sorted(combined) == list(range(n))
    assert set(s.fit).isdisjoint(s.calibration)
    assert set(s.fit).isdisjoint(s.holdout)
    assert set(s.calibration).isdisjoint(s.holdout)
