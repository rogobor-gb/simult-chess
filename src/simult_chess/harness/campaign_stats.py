"""Pre-registered estimand statistics for the Phase 11b campaign (docs/
DEVELOPMENT_addendum_v1.1.md §11b). Kept separate from `campaign.py` so the
formulas are independently testable against known reference values.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest, norm


@dataclass(frozen=True, slots=True)
class WilsonInterval:
    """A Wilson score confidence interval for a binomial proportion."""

    point: float
    low: float
    high: float


def wilson_interval(successes: int, n: int, *, z: float = 1.96) -> WilsonInterval:
    """95%-by-default Wilson score interval (closed form, no `n->inf` normal
    approximation pathology at small `n` or extreme `p`, spec estimand 1/6).
    """
    if n == 0:
        return WilsonInterval(point=float("nan"), low=float("nan"), high=float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return WilsonInterval(
        point=p, low=(center - spread) / denom, high=(center + spread) / denom
    )


@dataclass(frozen=True, slots=True)
class MedianIQR:
    median: float
    q1: float
    q3: float


def median_iqr(values: Sequence[float]) -> MedianIQR:
    if not values:
        return MedianIQR(median=float("nan"), q1=float("nan"), q3=float("nan"))
    arr = np.asarray(values, dtype=float)
    q1, median, q3 = np.percentile(arr, [25, 50, 75])
    return MedianIQR(median=float(median), q1=float(q1), q3=float(q3))


def volatility(material_trajectory: Sequence[float]) -> float:
    """Stdev of per-phase material-difference increments (estimand 5)."""
    if len(material_trajectory) < 2:
        return float("nan")
    increments = np.diff(np.asarray(material_trajectory, dtype=float))
    return float(np.std(increments))


@dataclass(frozen=True, slots=True)
class ColorSymmetryResult:
    white_wins: int
    decisive: int
    p_value: float
    wilson: WilsonInterval


def color_symmetry_test(white_wins: int, decisive: int) -> ColorSymmetryResult:
    """Exact two-sided binomial test of H0: p_White = 1/2 among decisive games
    (spec estimand 6). Same primitive as `tests/unit/test_solver_minimatch.py`'s
    strength check, applied here as a balance/symmetry audit instead.
    """
    if decisive == 0:
        return ColorSymmetryResult(
            white_wins=0,
            decisive=0,
            p_value=float("nan"),
            wilson=wilson_interval(0, 0),
        )
    result = binomtest(white_wins, decisive, p=0.5, alternative="two-sided")
    return ColorSymmetryResult(
        white_wins=white_wins,
        decisive=decisive,
        p_value=float(result.pvalue),
        wilson=wilson_interval(white_wins, decisive),
    )


def minimum_detectable_effect(
    n_per_arm: int, baseline_rate: float, *, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Minimal detectable absolute effect `delta` for a two-proportion
    comparison at the given per-arm sample size (spec §11b: "document the
    minimal detectable effect at that n" for each A/B arm), via the standard
    normal-approximation sample-size formula solved for `delta`:
    `n = (z_a2 + z_1mb)^2 * p(1-p) / delta^2`.
    """
    z_alpha2 = float(norm.ppf(1 - alpha / 2))
    z_power = float(norm.ppf(power))
    p = baseline_rate
    numerator = (z_alpha2 + z_power) ** 2 * p * (1 - p)
    return math.sqrt(numerator / n_per_arm)
