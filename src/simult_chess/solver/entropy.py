"""The maximum-entropy equilibrium of a zero-sum matrix game (Phase 18b,
docs/LEARNING_ROADMAP_v3.md).

`solver.lp.solve_zero_sum`'s LP returns *an* optimal vertex -- exact and
correct for the game's *value* (unique), but not canonical for its
*strategies* when the optimal face has positive dimension (payoff ties are
common in a game with terminal payoffs in `{-1, 0, +1}`): which vertex
HiGHS returns depends on pivot order, so it is not stable across solver
versions (v3 Prop. 2.6). The **maximum-entropy equilibrium** is the
canonical `tau -> 0` limit of the entropy-regularised saddle point

    val_tau(U) = max_x min_y [x^T U y + tau*H(x) - tau*H(y)]

(v3 Thm. 2.9) -- the unique, interior, solver-upgrade-stable selection
among the optimal face.

**Numerical approach, and why it isn't the obvious one.** The naive
approach -- iterate mutual logit best response, `x <- softmax(Uy/tau)`,
`y <- softmax(-U^Tx/tau)` -- was tried first and found to *not* reliably
converge on cyclic games (rock-paper-scissors-shaped payoff structures):
even damped/averaged, it stalled well short of the known answer. This
module instead exploits that, for *fixed* `tau`, each player's problem
decouples via the standard closed form for an entropy-regularised linear
objective over a simplex (`argmin_y [c^T y + tau*(-H(y))] = softmax(-c/tau)`,
value `-tau*logsumexp(-c/tau)`): substituting the opponent's closed-form
best response turns each player's *own* problem into a smooth, single-
player concave (row) / convex (column) optimisation in that player's own
strategy alone (reparametrised via `x = softmax(z)` to turn the simplex
constraint into an unconstrained one) -- solved independently for each
side via `scipy.optimize.minimize`, not derived from the other side's
(numerically noisy, and catastrophically amplified by a tiny `tau`)
residual. Solving both sides independently, rather than deriving one from
the other, is what avoids the numerical-noise blowup: deriving `y` from
`softmax(-U^T x* / tau)` using the row solve's own (slightly imprecise)
`x*` amplifies floating-point noise by `1/tau`, which is catastrophic once
`tau` reaches the ~1e-8 range the annealing schedule needs -- confirmed
empirically (see `test_solver_entropy.py`'s fixtures, each with a hand-
verifiable exact answer) before this design was trusted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from scipy.special import logsumexp

FloatArray = npt.NDArray[np.float64]

_DEFAULT_TAU_SCHEDULE: tuple[float, ...] = tuple(1.0 * (0.6**k) for k in range(50))
"""1.0 down to ~1e-11, each stage warm-started from the previous -- the
continuation-method schedule `test_solver_entropy.py` validates against."""


def _softmax(z: FloatArray) -> FloatArray:
    shifted = z - z.max()
    weights = np.exp(shifted)
    result: FloatArray = weights / weights.sum()
    return result


def _entropy(distribution: FloatArray) -> float:
    positive = distribution[distribution > 1e-300]
    return float(-(positive * np.log(positive)).sum())


def _solve_row(matrix: FloatArray, tau: float, z0: FloatArray) -> FloatArray:
    """`argmax_x [tau*H(x) - tau*logsumexp(-U^Tx/tau)]` (the row player's
    own regularised problem, the opponent's closed-form best response
    already substituted in) -- reparametrised as `x = softmax(z)` to drop
    the simplex constraint."""

    def negative_objective(z: FloatArray) -> float:
        x = _softmax(z)
        value = tau * _entropy(x) - tau * logsumexp(-matrix.T @ x / tau)
        return -float(value)

    result = minimize(negative_objective, z0, method="L-BFGS-B")
    z: FloatArray = result.x
    return z


def _solve_col(matrix: FloatArray, tau: float, w0: FloatArray) -> FloatArray:
    """The column player's symmetric counterpart:
    `argmin_y [tau*logsumexp(Uy/tau) - tau*H(y)]`."""

    def objective(w: FloatArray) -> float:
        y = _softmax(w)
        value = tau * logsumexp(matrix @ y / tau) - tau * _entropy(y)
        return float(value)

    result = minimize(objective, w0, method="L-BFGS-B")
    w: FloatArray = result.x
    return w


@dataclass(frozen=True, slots=True)
class MaxEntropyEquilibrium:
    row_strategy: FloatArray
    col_strategy: FloatArray
    value: float
    """`x^T U y` under the returned strategies."""
    row_entropy: float
    col_entropy: float


def solve_max_entropy_equilibrium(
    matrix: FloatArray,
    *,
    tau_schedule: tuple[float, ...] = _DEFAULT_TAU_SCHEDULE,
) -> MaxEntropyEquilibrium:
    """The max-entropy equilibrium of the zero-sum game `matrix` (row player
    maximizes, column player minimizes -- `solver.lp.solve_zero_sum`'s own
    convention). Solves each side's own smooth regularised problem
    independently at each `tau` in the (largest-first) schedule, warm-
    starting every stage from the previous (smaller `tau` is a small
    perturbation of the previous stage's solution, so this continuation
    approach converges reliably where a cold start at a tiny `tau` would
    not)."""
    m, n = matrix.shape
    z: FloatArray = np.zeros(m)
    w: FloatArray = np.zeros(n)
    for tau in tau_schedule:
        z = _solve_row(matrix, tau, z)
        w = _solve_col(matrix, tau, w)
    x, y = _softmax(z), _softmax(w)
    return MaxEntropyEquilibrium(
        row_strategy=x,
        col_strategy=y,
        value=float(x @ matrix @ y),
        row_entropy=_entropy(x),
        col_entropy=_entropy(y),
    )
