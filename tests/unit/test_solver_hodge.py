"""v3 18f.1: the combinatorial Hodge decomposition (`solver.hodge`),
validated against two hand-built, hand-verified fixtures before being
trusted on a real tournament -- this session's established practice.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")

import numpy as np  # noqa: E402

from simult_chess.solver.hodge import hodge_decompose  # noqa: E402


def test_decomposition_reconstructs_the_original_matrix_exactly() -> None:
    # gradient + curl == matrix is an algebraic identity by construction,
    # not a behavioural expectation -- pin it directly, on an arbitrary
    # (still skew-symmetric) matrix.
    matrix = np.array(
        [
            [0.0, 0.3, -0.6, 0.1],
            [-0.3, 0.0, 0.2, 0.5],
            [0.6, -0.2, 0.0, -0.4],
            [-0.1, -0.5, 0.4, 0.0],
        ]
    )
    decomposition = hodge_decompose(matrix)
    reconstructed = decomposition.gradient + decomposition.curl
    assert reconstructed == pytest.approx(matrix, abs=1e-9)


def test_purely_cyclic_population_has_a_flat_rating_and_full_cyclicity() -> None:
    # A beats B beats C beats A, each 100% -- the RPS-shaped case. No
    # scalar rating can explain "everyone beats someone and loses to
    # someone else identically," so the least-squares rating should be
    # exactly flat and the entire matrix should be curl.
    matrix = np.array(
        [
            [0.0, 1.0, -1.0],
            [-1.0, 0.0, 1.0],
            [1.0, -1.0, 0.0],
        ]
    )
    decomposition = hodge_decompose(matrix)
    assert decomposition.rating == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    assert decomposition.curl == pytest.approx(matrix, abs=1e-9)
    assert decomposition.cyclicity_ratio == pytest.approx(1.0, abs=1e-9)


def test_purely_transitive_ladder_has_zero_cyclicity() -> None:
    # Constructed directly from a scalar rating [1, 0, -1] via
    # matrix[i,j] = r[i] - r[j] -- by construction, a perfectly
    # transitive ladder (agent 0 > agent 1 > agent 2, consistent margins).
    rating = np.array([1.0, 0.0, -1.0])
    matrix = rating[:, None] - rating[None, :]
    decomposition = hodge_decompose(matrix)
    assert decomposition.rating == pytest.approx(rating, abs=1e-9)
    assert decomposition.curl == pytest.approx(np.zeros((3, 3)), abs=1e-9)
    assert decomposition.cyclicity_ratio == pytest.approx(0.0, abs=1e-9)


def test_all_zero_matrix_has_zero_cyclicity_not_full() -> None:
    # A degenerate all-tied population: nothing for a rating to explain,
    # but also nothing cyclic to find -- 0/0 should resolve to 0, not 1.
    matrix = np.zeros((3, 3))
    decomposition = hodge_decompose(matrix)
    assert decomposition.cyclicity_ratio == 0.0
