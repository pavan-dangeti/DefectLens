from __future__ import annotations

import random
from dataclasses import dataclass

_RATIOS = {"fit": 0.8, "calibration": 0.1, "holdout": 0.1}


@dataclass(frozen=True)
class Splits:
    fit: tuple[int, ...]
    calibration: tuple[int, ...]
    holdout: tuple[int, ...]

    def __post_init__(self) -> None:
        disjoint = (
            not (set(self.fit) & set(self.calibration))
            and not (set(self.fit) & set(self.holdout))
            and not (set(self.calibration) & set(self.holdout))
        )
        if not disjoint:
            raise ValueError("splits must be disjoint")


def make_splits(
    n: int, seed: int, ratios: dict[str, float] | None = None
) -> Splits:
    """Deterministically partition ``n`` items into fit / calibration / holdout.

    The split is stable for a given ``(n, seed, ratios)``, disjoint, and total: every index in
    ``range(n)`` appears in exactly one split. Few-shot experiments draw from ``fit`` only, so a
    few-shot model can never leak into its own calibration set.
    """
    if n < 3:
        raise ValueError(f"need at least 3 items for a three-way split, got {n}")
    ratios = ratios or dict(_RATIOS)
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("ratios must sum to a positive value")
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    fit_count = round(n * ratios["fit"] / total)
    cal_count = round(n * ratios["calibration"] / total)
    fit = tuple(sorted(indices[:fit_count]))
    calibration = tuple(sorted(indices[fit_count : fit_count + cal_count]))
    holdout = tuple(sorted(indices[fit_count + cal_count :]))
    if len(fit) + len(calibration) + len(holdout) != n:
        raise AssertionError("split counts do not sum to n")
    return Splits(fit=fit, calibration=calibration, holdout=holdout)
