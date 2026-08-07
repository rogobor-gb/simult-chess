"""Double oracle: the certificate `solve_exact`'s own top-k-by-prior
restriction (row-sketch's program pool) can never carry (v3 18c.3,
docs/LEARNING_ROADMAP_v3.md). A restricted stage matrix's value is
neither an upper nor a lower bound on the true game's value -- solving a
*shrinking* subset of a zero-sum game can favour either side arbitrarily.
Double oracle fixes this: solve the restricted game, find a best response
*outside* the current support, add it, repeat. Termination with no
improving best response for either side certifies an equilibrium of the
*full* (exhaustive) game -- and if terminated early, the largest
unresolved best-response gap is a genuine exploitability bound, not an
estimate.

`u(a, b)` is computed directly via `phi()` + a terminal check, at the
same `max_depth=1` depth `solver.exact.solve_exact` uses (any state still
"ongoing" gets the same `0.0` "undecided -> neutral" fallback that
module, and `learn.selfplay.ReplayBuffer`, already established) --
deliberately *not* by first running `solve_exact` itself, which would pay
the full exhaustive-solve cost before double oracle even starts, defeating
its own purpose (avoiding full stage-matrix evaluation). A `phi()` call
is only ever spent on a candidate cell double oracle actually touches,
memoized so repeat cells across iterations are free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from simult_chess.core.phi import phi
from simult_chess.core.types import Color, Program, State
from simult_chess.interop.openspiel_adapter import enumerate_legal_programs
from simult_chess.rules.ruleset import RuleSet
from simult_chess.solver.lp import solve_zero_sum

FloatArray = npt.NDArray[np.float64]

_TERMINAL_PAYOFF: dict[str, float] = {
    "white_wins": 1.0,
    "black_wins": -1.0,
    "draw": 0.0,
}


@dataclass(frozen=True, slots=True)
class DoubleOracleResult:
    """`exploitability_bound` is `0.0` exactly on certified convergence
    (neither side has an improving out-of-support best response) -- not
    an approximation reported as zero. If `max_iterations` is hit first,
    it is the largest unresolved best-response gap found, a genuine
    (not merely plausible) bound on how exploitable `value` still is."""

    value: float
    white_support: tuple[Program, ...]
    black_support: tuple[Program, ...]
    iterations: int
    exploitability_bound: float
    converged: bool


def solve_by_double_oracle(
    state: State,
    ruleset: RuleSet,
    *,
    initial_support_size: int = 1,
    max_iterations: int = 50,
    tol: float = 1e-9,
) -> DoubleOracleResult:
    white_pool = enumerate_legal_programs(state, Color.WHITE, ruleset)
    black_pool = enumerate_legal_programs(state, Color.BLACK, ruleset)
    if not white_pool or not black_pool:
        raise ValueError("double oracle needs at least one legal program per side")

    cache: dict[tuple[Program, Program], float] = {}

    def u(white_program: Program, black_program: Program) -> float:
        key = (white_program, black_program)
        cached = cache.get(key)
        if cached is not None:
            return cached
        result = phi(state, white_program, black_program, ruleset)
        value = _TERMINAL_PAYOFF.get(result.outcome, 0.0)
        cache[key] = value
        return value

    white_support = list(white_pool[:initial_support_size])
    black_support = list(black_pool[:initial_support_size])

    value = 0.0
    white_gap = 0.0
    black_gap = 0.0
    iteration = 0
    for iteration in range(max_iterations):
        matrix: FloatArray = np.array(
            [[u(a, b) for b in black_support] for a in white_support], dtype=np.float64
        )
        solution = solve_zero_sum(matrix)
        value = solution.value

        # White's best response to Black's current mixed strategy, over
        # the *full* exhaustive pool, not just the current support.
        white_best_program = white_pool[0]
        white_best_value = float("-inf")
        for candidate in white_pool:
            candidate_value = float(
                sum(
                    p * u(candidate, b)
                    for p, b in zip(solution.col_strategy, black_support, strict=True)
                )
            )
            if candidate_value > white_best_value:
                white_best_value = candidate_value
                white_best_program = candidate

        # Black's best response to White's current mixed strategy --
        # Black minimizes, over the full exhaustive pool.
        black_best_program = black_pool[0]
        black_best_value = float("inf")
        for candidate in black_pool:
            candidate_value = float(
                sum(
                    p * u(a, candidate)
                    for p, a in zip(solution.row_strategy, white_support, strict=True)
                )
            )
            if candidate_value < black_best_value:
                black_best_value = candidate_value
                black_best_program = candidate

        white_gap = white_best_value - value
        black_gap = value - black_best_value

        white_improves = white_gap > tol and white_best_program not in white_support
        black_improves = black_gap > tol and black_best_program not in black_support
        if not white_improves and not black_improves:
            return DoubleOracleResult(
                value=value,
                white_support=tuple(white_support),
                black_support=tuple(black_support),
                iterations=iteration + 1,
                exploitability_bound=0.0,
                converged=True,
            )
        if white_improves:
            white_support.append(white_best_program)
        if black_improves:
            black_support.append(black_best_program)

    return DoubleOracleResult(
        value=value,
        white_support=tuple(white_support),
        black_support=tuple(black_support),
        iterations=iteration + 1,
        exploitability_bound=max(white_gap, black_gap, 0.0),
        converged=False,
    )
