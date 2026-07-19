"""Evaluation harness (Phase 13b, design §6): relative-Elo ladder, the
primary strength gate (§6.2), exploitability/NashConv (§6.3), and learning
diagnostics (§6.4).

This module is a `learn/` submodule but, unlike the rest of the package, is
allowed to import `simult_chess.solver` too (numpy/scipy, no torch) --
evaluation genuinely needs `matrix_1ply` as a ladder opponent (§6.1) and
`solver.lp.solve_zero_sum` for exact NashConv (§6.3); this does not
compromise the core torch quarantine (`tests/unit/test_learn_quarantine.py`
only asserts `core`/`rules`/`referee`/agents/harness stay torch-free, which
this module's solver dependency doesn't touch).
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from simult_chess.agents.base import Agent
from simult_chess.referee.match import play_match
from simult_chess.referee.setup import standard_starting_state
from simult_chess.rules.ruleset import RuleSet

_UNDECIDED_SCORE = 0.5
"""A phase-limit-reached match is scored as a draw (0.5): the match neither
side clearly won, matching the standard chess-rating convention that an
undecided result contributes no net evidence of relative strength."""


@dataclass(frozen=True, slots=True)
class LadderMatchResult:
    """One seeded ladder match set: `agent_a`'s per-game score (1.0 win,
    0.5 draw/undecided, 0.0 loss) and raw outcome, colour-balanced."""

    scores: tuple[float, ...]
    outcomes: tuple[str, ...]

    @property
    def win_rate(self) -> float:
        return sum(self.scores) / len(self.scores)

    @property
    def decisive_games(self) -> int:
        return sum(1 for s in self.scores if s in (0.0, 1.0))

    @property
    def wins(self) -> int:
        return sum(1 for s in self.scores if s == 1.0)


def play_ladder_match(
    agent_a: Agent,
    agent_b: Agent,
    ruleset: RuleSet,
    n_games: int,
    base_seed: int,
    *,
    max_phases: int = 500,
) -> LadderMatchResult:
    """`n_games` seeded games, colour-balanced (`agent_a` plays White in the
    even-indexed games, Black in the odd-indexed ones -- M3's own operator-
    level symmetry means colour shouldn't matter, so balancing removes a
    confound from the strength measurement rather than relying on it)."""
    scores: list[float] = []
    outcomes: list[str] = []
    for i in range(n_games):
        seed = base_seed + i
        rng_a = random.Random(seed)
        rng_b = random.Random(seed ^ 0x5EED)
        a_is_white = i % 2 == 0
        if a_is_white:
            result = play_match(
                standard_starting_state(),
                agent_a,
                agent_b,
                ruleset,
                rng_white=rng_a,
                rng_black=rng_b,
                max_phases=max_phases,
            )
        else:
            result = play_match(
                standard_starting_state(),
                agent_b,
                agent_a,
                ruleset,
                rng_white=rng_b,
                rng_black=rng_a,
                max_phases=max_phases,
            )
        outcomes.append(result.outcome)
        if result.outcome == "draw" or result.outcome == "phase_limit_reached":
            scores.append(_UNDECIDED_SCORE)
        elif (result.outcome == "white_wins") == a_is_white:
            scores.append(1.0)
        else:
            scores.append(0.0)
    return LadderMatchResult(scores=tuple(scores), outcomes=tuple(outcomes))


def elo_diff_from_score(score_rate: float) -> float:
    """The standard logistic Elo-difference-from-expected-score transform
    (score_rate = (wins + 0.5*draws) / games). Clamped away from {0, 1} to
    avoid +-inf at a shutout, which a finite match sample can produce
    without implying an infinite rating gap."""
    clamped = min(max(score_rate, 1e-6), 1.0 - 1e-6)
    return -400.0 * math.log10(1.0 / clamped - 1.0)


@dataclass(frozen=True, slots=True)
class EloEstimate:
    point: float
    lower: float
    upper: float


def bootstrap_elo(
    scores: Sequence[float],
    rng: random.Random,
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
) -> EloEstimate:
    """Percentile-bootstrap CI on the Elo difference implied by `scores`
    (design §6.1: "Report relative Elo with bootstrap CIs")."""
    n = len(scores)
    point = elo_diff_from_score(sum(scores) / n)
    resampled = []
    for _ in range(n_resamples):
        resample_rate = sum(scores[rng.randrange(n)] for _ in range(n)) / n
        resampled.append(elo_diff_from_score(resample_rate))
    resampled.sort()
    tail = (1.0 - confidence) / 2.0
    lower = resampled[int(tail * n_resamples)]
    upper = resampled[int((1.0 - tail) * n_resamples) - 1]
    return EloEstimate(point=point, lower=lower, upper=upper)


def strength_gate_passes(
    ladder_result: LadderMatchResult, *, alpha: float = 0.01
) -> tuple[bool, float]:
    """The 13b DoD's primary strength gate (design §6.2): the checkpoint
    defeats `matrix_1ply` at one-sided exact binomial p < `alpha` -- the
    same test Phase 10 used for `matrix_1ply` vs `random_legal`
    (`tests/unit/test_solver_minimatch.py`), reused for continuity. Returns
    `(passes, p_value)`; `decisive_games == 0` (no wins or losses at all)
    cannot pass by construction (nothing to test)."""
    from scipy.stats import binomtest

    decisive = ladder_result.decisive_games
    wins = ladder_result.wins
    if decisive == 0:
        return False, 1.0
    p_value = float(
        binomtest(wins, decisive, p=0.5, alternative="greater").pvalue
    )
    return p_value < alpha, p_value
