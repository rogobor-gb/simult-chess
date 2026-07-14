"""Zero-sum matrix-game value and optimal mixed strategies by LP (spec §8.1,
scipy/HiGHS), with the standard duality-pair certificate check.

Numerical tolerance policy, stated once and used everywhere in the solver
layer (spec §8.1): ``ATOL = 1e-9``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import linprog

FloatArray = npt.NDArray[np.float64]

ATOL = 1e-9


@dataclass(frozen=True, slots=True)
class MatrixGameSolution:
    """A solved zero-sum matrix game: row player (`support_white`, index i)
    maximizes, column player (`support_black`, index j) minimizes."""

    value: float
    row_strategy: FloatArray
    col_strategy: FloatArray


def _solve_row_player(matrix: FloatArray) -> tuple[FloatArray, float]:
    """max_x min_j (x^T U)_j  s.t. sum(x)=1, x>=0 -- the row (maximizing)
    player's optimal mixed strategy and the game value it guarantees."""
    m, n = matrix.shape
    # Variables z = [x_1..x_m, v]; minimize -v (i.e. maximize v).
    c = np.zeros(m + 1)
    c[-1] = -1.0
    # For every column j: v <= sum_i x_i * U[i,j]  =>  -U[:,j]·x + v <= 0
    a_ub = np.zeros((n, m + 1))
    a_ub[:, :m] = -matrix.T
    a_ub[:, m] = 1.0
    b_ub = np.zeros(n)
    a_eq = np.zeros((1, m + 1))
    a_eq[0, :m] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0.0, None)] * m + [(None, None)]

    result = linprog(
        c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )
    if not result.success:
        raise ValueError(f"row-player LP failed to solve: {result.message}")
    x = result.x[:m]
    v = result.x[m]
    return x, float(v)


def _solve_col_player(matrix: FloatArray) -> tuple[FloatArray, float]:
    """min_y max_i (U y)_i  s.t. sum(y)=1, y>=0 -- the column (minimizing)
    player's optimal mixed strategy and the game value it guarantees."""
    m, n = matrix.shape
    # Variables z = [y_1..y_n, w]; minimize w directly.
    c = np.zeros(n + 1)
    c[-1] = 1.0
    # For every row i: sum_j U[i,j] y_j <= w  =>  U[i,:]·y - w <= 0
    a_ub = np.zeros((m, n + 1))
    a_ub[:, :n] = matrix
    a_ub[:, n] = -1.0
    b_ub = np.zeros(m)
    a_eq = np.zeros((1, n + 1))
    a_eq[0, :n] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0.0, None)] * n + [(None, None)]

    result = linprog(
        c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )
    if not result.success:
        raise ValueError(f"column-player LP failed to solve: {result.message}")
    y = result.x[:n]
    w = result.x[n]
    return y, float(w)


def solve_zero_sum(matrix: FloatArray, *, atol: float = ATOL) -> MatrixGameSolution:
    """Solve the zero-sum matrix game `matrix` by the standard primal/dual LP
    pair, verified by the certificate :math:`x^\\top U y` agreeing with both
    LP objective values within `atol` (spec §8.1's duality pair)."""
    row_strategy, row_value = _solve_row_player(matrix)
    col_strategy, col_value = _solve_col_player(matrix)
    certificate = float(row_strategy @ matrix @ col_strategy)

    # LP numerical slack is typically far tighter than 1e-9 for the small,
    # well-scaled matrices this layer solves; if this ever fires in
    # practice it means the LPs disagree on the game's value, not just
    # floating-point noise, and should be investigated rather than loosened.
    if abs(certificate - row_value) > atol or abs(certificate - col_value) > atol:
        raise ValueError(
            "LP duality certificate failed: "
            f"x^T U y={certificate}, row_value={row_value}, col_value={col_value}"
        )
    return MatrixGameSolution(
        value=(row_value + col_value) / 2.0,
        row_strategy=row_strategy,
        col_strategy=col_strategy,
    )
