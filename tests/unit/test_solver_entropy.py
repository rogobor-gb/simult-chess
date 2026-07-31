"""v3 18b: `solver.entropy.solve_max_entropy_equilibrium`, validated against
fixtures with a hand-verifiable exact answer -- including one deliberately
constructed to have a *degenerate* (positive-dimensional) optimal face, to
check the solver actually picks the canonical max-entropy point on that
face rather than an arbitrary one, which is the entire reason this module
exists instead of just reusing `solver.lp.solve_zero_sum`.
"""

from __future__ import annotations

import numpy as np
import pytest

from simult_chess.solver.entropy import solve_max_entropy_equilibrium
from simult_chess.solver.lp import solve_zero_sum

_ATOL = 1e-4


def test_matching_pennies_is_uniform() -> None:
    matrix = np.array([[1.0, -1.0], [-1.0, 1.0]])
    result = solve_max_entropy_equilibrium(matrix)
    assert result.row_strategy == pytest.approx([0.5, 0.5], abs=_ATOL)
    assert result.col_strategy == pytest.approx([0.5, 0.5], abs=_ATOL)
    assert result.value == pytest.approx(0.0, abs=_ATOL)


def test_weighted_rps_matches_the_known_unique_equilibrium() -> None:
    """v3 18a'.1's own weighted-RPS fixture: a *unique*, fully-mixed
    equilibrium (1/4, 1/2, 1/4) -- since it's unique, the max-entropy
    selection has no freedom to differ from it, a useful sanity check that
    the solver converges to the *correct* fixed point at all (not just
    *some* point on a degenerate face)."""
    matrix = np.array([[0.0, -1.0, 2.0], [1.0, 0.0, -1.0], [-2.0, 1.0, 0.0]])
    result = solve_max_entropy_equilibrium(matrix)
    expected = [0.25, 0.5, 0.25]
    assert result.row_strategy == pytest.approx(expected, abs=_ATOL)
    assert result.col_strategy == pytest.approx(expected, abs=_ATOL)
    assert result.value == pytest.approx(0.0, abs=_ATOL)


def test_duplicated_row_picks_the_max_entropy_point_on_a_degenerate_face() -> None:
    """Matching Pennies with row 0 split into two *identical* copies
    (rows 0 and 1 both (1,-1); row 2 is (-1,1)) -- the true optimal row
    strategy is any split of 0.5 mass between the two identical rows, plus
    0.5 on row 2: a genuinely positive-dimensional optimal face. The LP
    (`solve_zero_sum`) returns an arbitrary vertex of that face (e.g. all
    mass on one of the two duplicates); the max-entropy selection must
    split the tied mass *evenly* between them, (0.25, 0.25, 0.5) -- this is
    the concrete case this whole module exists for."""
    matrix = np.array([[1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]])
    lp = solve_zero_sum(matrix)
    assert lp.row_strategy[0] == pytest.approx(0.0, abs=_ATOL) or lp.row_strategy[
        1
    ] == pytest.approx(0.0, abs=_ATOL)  # the LP vertex is degenerate, as expected

    result = solve_max_entropy_equilibrium(matrix)
    assert result.row_strategy == pytest.approx([0.25, 0.25, 0.5], abs=_ATOL)
    assert result.col_strategy == pytest.approx([0.5, 0.5], abs=_ATOL)


def test_fully_degenerate_all_zero_matrix_is_uniform() -> None:
    """Every entry 0: every strategy is optimal for both players. The
    max-entropy selection is uniform on both sides -- the maximum-entropy
    point of the *entire* simplex, since the entire simplex is the optimal
    face."""
    matrix = np.array([[0.0, 0.0], [0.0, 0.0]])
    result = solve_max_entropy_equilibrium(matrix)
    assert result.row_strategy == pytest.approx([0.5, 0.5], abs=_ATOL)
    assert result.col_strategy == pytest.approx([0.5, 0.5], abs=_ATOL)


def test_matches_lp_value_and_a_dominated_action_gets_zero_mass() -> None:
    """The 18a'.1 unequal-support fixture (val != 0, a genuinely dominated
    column action) -- value must match the LP exactly, and the dominated
    action's mass must vanish, not just shrink."""
    matrix = np.array([[3.0, -1.0, -2.0], [-2.0, 2.0, 1.0]]) / 3.0
    lp = solve_zero_sum(matrix)
    result = solve_max_entropy_equilibrium(matrix)
    assert result.value == pytest.approx(lp.value, abs=_ATOL)
    assert result.row_strategy == pytest.approx(lp.row_strategy, abs=_ATOL)
    assert result.col_strategy[1] == pytest.approx(0.0, abs=1e-3)


def test_entropy_fields_are_nonnegative_and_bounded_by_log_of_support_size() -> None:
    matrix = np.array([[0.0, -1.0, 2.0], [1.0, 0.0, -1.0], [-2.0, 1.0, 0.0]])
    result = solve_max_entropy_equilibrium(matrix)
    assert 0.0 <= result.row_entropy <= np.log(3) + 1e-9
    assert 0.0 <= result.col_entropy <= np.log(3) + 1e-9
