"""Learning diagnostics (Phase 13b, design §6.4): stage-policy entropy,
value-head calibration, and a color-symmetry spot-check.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from simult_chess.harness.campaign_stats import (
    ColorSymmetryResult,
    color_symmetry_test,
)
from simult_chess.learn.net import SimultChessNet
from simult_chess.learn.selfplay import SelfPlayGame, TrainingExample


def entropy(distribution: dict[int, float]) -> float:
    """Shannon entropy (nats) of a probability distribution. 0 for a pure
    (one-hot) strategy; `log(n)` for uniform over `n` legal actions."""
    total = sum(distribution.values())
    if total <= 0.0:
        return 0.0
    result = 0.0
    for weight in distribution.values():
        p = weight / total
        if p > 0.0:
            result -= p * math.log(p)
    return result


@dataclass(frozen=True, slots=True)
class EntropySummary:
    """Mean slot-1 search-strategy entropy over a set of self-play games
    (design §6.4: "does self-play discover genuinely mixed play?"). Tracked
    separately per color since the two heads train independently."""

    mean_white: float
    mean_black: float
    n_phases: int


def stage_policy_entropy(games: Sequence[SelfPlayGame]) -> EntropySummary:
    """Mean entropy of the recorded slot-1 search-average-strategy targets
    (`PhaseRecord.white_slot1_target`/`black_slot1_target`) over every phase
    of `games` -- no re-search needed, since the search-derived targets are
    already the mixed strategies §6.4 asks about."""
    white_entropies: list[float] = []
    black_entropies: list[float] = []
    for game in games:
        for phase in game.phases:
            white_entropies.append(entropy(phase.white_slot1_target))
            black_entropies.append(entropy(phase.black_slot1_target))
    n = len(white_entropies)
    if n == 0:
        return EntropySummary(mean_white=0.0, mean_black=0.0, n_phases=0)
    return EntropySummary(
        mean_white=sum(white_entropies) / n,
        mean_black=sum(black_entropies) / n,
        n_phases=n,
    )


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    predicted_mean: float
    actual_mean: float
    count: int


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    bins: tuple[CalibrationBin, ...]
    mean_absolute_error: float


def value_calibration(
    net: SimultChessNet,
    examples: Sequence[TrainingExample],
    *,
    n_bins: int = 10,
    device: torch.device | None = None,
) -> CalibrationResult:
    """Bin the network's predicted value against the actual Monte-Carlo
    return `z` from the same self-play run (design §6.4). `net` is run in
    `eval()` mode; callers pass a checkpoint snapshot, not a training-mode
    net (the standard "load the checkpoint, eval() it" pattern)."""
    net.eval()
    device = device or next(net.parameters()).device
    planes = torch.from_numpy(
        np.stack([ex.phase.planes for ex in examples])
    ).to(device)
    scalars = torch.from_numpy(
        np.stack([ex.phase.scalars for ex in examples])
    ).to(device)
    with torch.no_grad():
        _, _, predicted, _ = net(planes, scalars)
    predicted_np = predicted.cpu().numpy()
    actual_np = np.array([ex.z for ex in examples], dtype=np.float64)

    edges = np.linspace(-1.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(predicted_np, edges[1:-1]), 0, n_bins - 1)
    bins: list[CalibrationBin] = []
    for b in range(n_bins):
        mask = bin_indices == b
        count = int(mask.sum())
        if count == 0:
            continue
        bins.append(
            CalibrationBin(
                predicted_mean=float(predicted_np[mask].mean()),
                actual_mean=float(actual_np[mask].mean()),
                count=count,
            )
        )
    mae = float(np.mean(np.abs(predicted_np - actual_np))) if len(examples) else 0.0
    return CalibrationResult(bins=tuple(bins), mean_absolute_error=mae)


def color_symmetry_spot_check(games: Sequence[SelfPlayGame]) -> ColorSymmetryResult:
    """Cheap M3 hygiene, reused from Phase 11b (design §6.4): among decisive
    self-play games, White-win fraction against a two-sided exact binomial
    vs 1/2. Since Phi is provably chi-symmetric (M3), any rejection localizes
    to a learned-agent asymmetry, never the rules -- delegates entirely to
    `harness.campaign_stats.color_symmetry_test`, the same primitive Phase
    11b's own campaign used, for one source of truth."""
    white_wins = sum(1 for g in games if g.outcome == "white_wins")
    decisive = sum(1 for g in games if g.outcome in ("white_wins", "black_wins"))
    return color_symmetry_test(white_wins, decisive)
