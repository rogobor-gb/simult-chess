"""The strategic-depth profile functionals (Phase 18b.4,
docs/LEARNING_ROADMAP_v3.md, Vol. I §2.7/§4.5).

Each functional is a pure function of a solved stage matrix (and, where
noted, its max-entropy equilibrium) -- at most one extra LP/optimisation
once `solver.exact.solve_exact` and `solver.entropy.solve_max_entropy_
equilibrium` already exist, per the roadmap's own cost note.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from simult_chess.core.types import Cancel, Program, Reserve
from simult_chess.solver.entropy import (
    MaxEntropyEquilibrium,
    solve_max_entropy_equilibrium,
)
from simult_chess.solver.lp import solve_zero_sum

FloatArray = npt.NDArray[np.float64]


def gamma(matrix: FloatArray) -> float:
    """`min_j max_i U_ij - max_i min_j U_ij`: the value of move order --
    how much simultaneity itself matters. Identically 0 in any alternating
    game (where one side would get to react to the other's already-
    revealed choice); the game's strategic-depth signature."""
    return float(matrix.max(axis=0).min() - matrix.min(axis=1).max())


def mu(matrix: FloatArray, value: float) -> float:
    """`val(U) - max_i min_j U_ij`: the value of being allowed to
    randomise, over the best deterministic (pure) guarantee."""
    return float(value - matrix.min(axis=1).max())


def support_size(distribution: FloatArray, *, atol: float = 1e-6) -> int:
    """`k*(s)`: how many actions carry non-negligible mass in an
    equilibrium strategy."""
    return int((distribution > atol).sum())


def entropy_of(distribution: FloatArray) -> float:
    """`H*(s)`: Shannon entropy of an equilibrium strategy (nats)."""
    positive = distribution[distribution > 1e-300]
    return float(-(positive * np.log(positive)).sum())


def phi_functional(
    matrix: FloatArray, programs: tuple[Program, ...]
) -> float | None:
    """`val(U) - max over product-form x of min_y x^T U y`: how much
    intra-program slot coordination is worth -- the gap between the true
    (correlated, joint-program) equilibrium value and the best a player
    restricted to independently-mixing its own slots could guarantee.

    **Returns `None` (not computed), not a fake 0.0, whenever `programs`
    contains no genuine 2-slot program.** Reporting 0.0 in that case would
    be silently indistinguishable from "coordination was measured and
    found worthless" -- exactly the false-negative this functional exists
    to rule out (v3's own framing: `phi` and `delta` are the first
    quantitative statement about whether the reservation mechanic carries
    any weight at all; a fabricated 0.0 would corrupt that reading). When
    every program in `programs` is single-action, phi is *trivially* and
    *correctly* 0 (no second slot exists to coordinate), so that one case
    legitimately returns 0.0; every other case (multi-slot programs
    present) needs the caller to actually solve the constrained product-
    form optimisation, not implemented here yet -- left as an explicit gap
    rather than guessed at, matching this session's practice elsewhere."""
    if all(len(program) <= 1 for program in programs):
        return 0.0
    return None


def delta_functional(value_with_reserve: float, value_without_reserve: float) -> float:
    """`val(U) - val(U with White's Reserve programs deleted)`: the
    shadow price of the commitment mechanic -- how much value White loses
    by being unable to declare a Reserve at all."""
    return float(value_with_reserve - value_without_reserve)


def reservation_usage(
    programs: tuple[Program, ...], equilibrium: FloatArray
) -> tuple[float, float]:
    """`(rho_res, rho_can)`: equilibrium mass on programs containing a
    `Reserve`/`Cancel` action respectively -- whether the mechanics are
    used at equilibrium at all."""
    rho_res = 0.0
    rho_can = 0.0
    for program, mass in zip(programs, equilibrium, strict=True):
        if any(isinstance(action, Reserve) for action in program):
            rho_res += float(mass)
        if any(isinstance(action, Cancel) for action in program):
            rho_can += float(mass)
    return rho_res, rho_can


@dataclass(frozen=True, slots=True)
class DepthProfile:
    fixture_name: str
    value: float
    gamma: float
    mu: float
    k_star_white: int
    k_star_black: int
    h_star_white: float
    h_star_black: float
    phi: float | None
    """`None` means not computed (genuine multi-slot programs present but
    the constrained product-form optimisation isn't implemented) -- see
    `phi_functional`. Never a fabricated 0.0."""
    rho_res_white: float
    rho_can_white: float
    rho_res_black: float
    rho_can_black: float
    delta: float | None = None
    """`None` when the caller didn't supply a Reserve-deleted comparison
    matrix (see `compute_depth_profile`'s `matrix_without_white_reserve`)."""


def compute_depth_profile(
    fixture_name: str,
    matrix: FloatArray,
    white_programs: tuple[Program, ...],
    black_programs: tuple[Program, ...],
    *,
    matrix_without_white_reserve: FloatArray | None = None,
    equilibrium: MaxEntropyEquilibrium | None = None,
) -> DepthProfile:
    """The full depth profile of one solved stage matrix. `equilibrium`,
    if not supplied, is computed here (the max-entropy one, per 18b.2's
    own fixture-format decision -- canonical, not an arbitrary LP vertex)."""
    solution = solve_zero_sum(matrix)
    eq = equilibrium or solve_max_entropy_equilibrium(matrix)
    delta = None
    if matrix_without_white_reserve is not None:
        without = solve_zero_sum(matrix_without_white_reserve)
        delta = delta_functional(solution.value, without.value)
    rho_res_w, rho_can_w = reservation_usage(white_programs, eq.row_strategy)
    rho_res_b, rho_can_b = reservation_usage(black_programs, eq.col_strategy)
    return DepthProfile(
        fixture_name=fixture_name,
        value=solution.value,
        gamma=gamma(matrix),
        mu=mu(matrix, solution.value),
        k_star_white=support_size(eq.row_strategy),
        k_star_black=support_size(eq.col_strategy),
        h_star_white=entropy_of(eq.row_strategy),
        h_star_black=entropy_of(eq.col_strategy),
        phi=phi_functional(matrix, white_programs),
        rho_res_white=rho_res_w,
        rho_can_white=rho_can_w,
        rho_res_black=rho_res_b,
        rho_can_black=rho_can_b,
        delta=delta,
    )
