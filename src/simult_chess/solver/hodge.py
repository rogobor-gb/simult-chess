"""Combinatorial Hodge decomposition of a pairwise-comparison ("who beats
whom") matrix (v3 18f.1, docs/LEARNING_ROADMAP_v3.md), used to separate the
transitive component of a population's strength (a single scalar rating
recovers it) from the genuinely cyclic component (a rating cannot, by
definition -- that mass is exactly what a scalar Elo throws away).

Nothing like this exists elsewhere in the repo -- new module, following
the HodgeRank framework (Jiang, Lim, Yao & Ye 2011). The general
decomposition of a skew-symmetric pairwise matrix on a graph has three
components (gradient + harmonic + curl); on a **complete** graph (every
pair compared, exactly the shape a full round-robin tournament produces)
the harmonic component vanishes identically, leaving the two-term split
the roadmap itself names: ``P = grad(r) + rot``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def least_squares_rating(matrix: FloatArray) -> FloatArray:
    """The gradient (transitive) component's scalar rating: the `r`
    minimizing `sum_{i,j} (matrix[i,j] - (r[i]-r[j]))**2` over a
    **complete** comparison graph (`matrix` skew-symmetric, `matrix[i,i]
    == 0`).

    Closed form, not an iterative solve: setting the objective's gradient
    to zero gives `n*r[i] - sum_j r[j] = sum_j matrix[i,j]`, and since `r`
    is only determined up to an additive constant (shifting every rating
    by the same amount leaves every difference `r[i]-r[j]` unchanged),
    fixing that constant via the natural mean-zero normalization
    (`sum_j r[j] = 0`) collapses this to `r[i] = mean_j(matrix[i,j])` --
    each agent's average margin against the rest of the population.
    """
    rating: FloatArray = np.asarray(matrix, dtype=np.float64).mean(axis=1)
    return rating


@dataclass(frozen=True, slots=True)
class HodgeDecomposition:
    """`matrix == gradient + curl` exactly (an algebraic identity, not an
    approximation) -- `gradient` is what a single scalar rating explains,
    `curl` is the residual no rating can, and `cyclicity_ratio` is how
    much of the matrix's total magnitude that residual accounts for."""

    rating: FloatArray
    gradient: FloatArray
    curl: FloatArray
    cyclicity_ratio: float


def hodge_decompose(matrix: FloatArray) -> HodgeDecomposition:
    """Decompose a skew-symmetric pairwise-comparison `matrix` (`matrix ==
    -matrix.T`, `matrix[i,i] == 0`) into its gradient (transitive) and
    curl (cyclic) components. `cyclicity_ratio` is `‖curl‖_F / ‖matrix‖_F`
    (Frobenius norm) -- 0 for a purely transitive population (a scalar
    rating explains everything), closer to 1 the more the "beats"
    relation cycles; 0 identically for the degenerate all-zero matrix
    (nothing to explain, not "fully cyclic")."""
    matrix = np.asarray(matrix, dtype=np.float64)
    rating = least_squares_rating(matrix)
    gradient = rating[:, None] - rating[None, :]
    curl = matrix - gradient
    matrix_norm = float(np.linalg.norm(matrix))
    cyclicity_ratio = (
        float(np.linalg.norm(curl)) / matrix_norm if matrix_norm > 0.0 else 0.0
    )
    return HodgeDecomposition(
        rating=rating, gradient=gradient, curl=curl, cyclicity_ratio=cyclicity_ratio
    )
