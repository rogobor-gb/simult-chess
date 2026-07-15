"""`matrix_1ply`: an equilibrium-mixing agent over a pruned stage-matrix
support (spec §8.4, A7). Lives in `solver/`, not `agents/`, so `agents/`
stays standard-library only (`solver/` is the one place `numpy`/`scipy` may
be imported).
"""

from __future__ import annotations

import random

from simult_chess.core.types import Color, Program, State
from simult_chess.rules.ruleset import RuleSet
from simult_chess.solver.lp import solve_zero_sum
from simult_chess.solver.stage_matrix import build_stage_matrix
from simult_chess.solver.supports import enumerate_support


def matrix_1ply(
    state: State, color: Color, ruleset: RuleSet, rng: random.Random
) -> Program:
    """Sample a program from this phase's approximate stage-game equilibrium.

    Builds a small, seeded restricted support for *both* colors (spec §8.1's
    stage game is defined over both sides' action sets at the same public
    state — modeling the opponent's likely support is not privileged
    information, since both engines see the same `state`), solves the
    resulting matrix by LP (`solver/lp.py`), and samples a program from
    `color`'s side of the equilibrium mixed strategy.
    """
    support_white = enumerate_support(state, Color.WHITE, ruleset, rng)
    support_black = enumerate_support(state, Color.BLACK, ruleset, rng)
    own_support = support_white if color is Color.WHITE else support_black
    if not own_support:
        return ()
    if not support_white or not support_black:
        return rng.choice(own_support)

    matrix = build_stage_matrix(state, support_white, support_black, ruleset)
    solution = solve_zero_sum(matrix)
    strategy = solution.row_strategy if color is Color.WHITE else solution.col_strategy
    # Clip tiny negative LP noise and renormalize before sampling.
    weights = [max(0.0, w) for w in strategy]
    return rng.choices(own_support, weights=weights, k=1)[0]
