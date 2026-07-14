from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")

import numpy as np  # noqa: E402

from simult_chess.solver.lp import ATOL, solve_zero_sum  # noqa: E402


def test_solve_zero_sum_on_a_single_entry_matrix() -> None:
    matrix = np.array([[5.0]])
    solution = solve_zero_sum(matrix)
    assert solution.value == pytest.approx(5.0)
    assert solution.row_strategy == pytest.approx([1.0])
    assert solution.col_strategy == pytest.approx([1.0])


def test_solve_zero_sum_on_matching_pennies() -> None:
    # Classic no-pure-equilibrium game (spec section 8.2's own example):
    # value 0, uniform mixing on both sides.
    matrix = np.array([[1.0, -1.0], [-1.0, 1.0]])
    solution = solve_zero_sum(matrix)
    assert solution.value == pytest.approx(0.0, abs=1e-6)
    assert solution.row_strategy == pytest.approx([0.5, 0.5], abs=1e-6)
    assert solution.col_strategy == pytest.approx([0.5, 0.5], abs=1e-6)


def test_solve_zero_sum_certificate_matches_both_lp_objectives() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.uniform(-9.0, 9.0, size=(4, 5))
    solution = solve_zero_sum(matrix)
    certificate = float(solution.row_strategy @ matrix @ solution.col_strategy)
    assert certificate == pytest.approx(solution.value, abs=1e-6)


def test_solve_zero_sum_strategies_are_valid_probability_distributions() -> None:
    rng = np.random.default_rng(1)
    matrix = rng.uniform(-5.0, 5.0, size=(3, 3))
    solution = solve_zero_sum(matrix)
    assert np.all(solution.row_strategy >= -ATOL)
    assert np.all(solution.col_strategy >= -ATOL)
    assert solution.row_strategy.sum() == pytest.approx(1.0, abs=1e-6)
    assert solution.col_strategy.sum() == pytest.approx(1.0, abs=1e-6)


def test_solve_zero_sum_dominant_strategy_case() -> None:
    # Row 0 dominates row 1 in every column -> row player always plays it,
    # column player is then indifferent only insofar as it minimizes.
    matrix = np.array([[2.0, 3.0], [-1.0, 0.0]])
    solution = solve_zero_sum(matrix)
    assert solution.value == pytest.approx(2.0, abs=1e-6)
    assert solution.row_strategy == pytest.approx([1.0, 0.0], abs=1e-6)
