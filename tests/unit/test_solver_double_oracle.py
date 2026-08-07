"""v3 18c.3 (required tier): `solver.double_oracle`, cross-validated
against `solve_exact`'s already-established exact values on all three
18b fixtures -- the decisive check (does an independent growth-based
mechanism reconstruct an answer this session already trusts from a
different code path), not just "runs without crashing."
"""

from __future__ import annotations

import pytest

from simult_chess.core.types import (
    Bookkeeping,
    CastlingRights,
    Color,
    Square,
    State,
    Token,
)
from simult_chess.rules.ruleset import RuleSet
from simult_chess.solver.double_oracle import solve_by_double_oracle
from simult_chess.solver.exact import solve_exact

RULESET = RuleSet(n_actions=1)
_NO_CASTLING = CastlingRights(
    white_kingside=False,
    white_queenside=False,
    black_kingside=False,
    black_queenside=False,
)


def _sq(name: str) -> Square:
    return Square(ord(name[0]) - ord("a"), int(name[1]) - 1)


# --- matching_pennies_dodge (identical construction used throughout the
# session) -------------------------------------------------------------

_WHITE_KING = Token(id=1, color=Color.WHITE, typ="k")
_BLACK_KING = Token(id=2, color=Color.BLACK, typ="k")
_BLOCKER_A1 = Token(id=10, color=Color.WHITE, typ="p")
_BLOCKER_A3 = Token(id=11, color=Color.WHITE, typ="p")
_BLOCKER_B2 = Token(id=12, color=Color.WHITE, typ="p")
_BLOCKER_B3 = Token(id=13, color=Color.WHITE, typ="p")
_KNIGHT = Token(id=4, color=Color.BLACK, typ="n")


def _dodge_state() -> State:
    return State(
        board={
            _WHITE_KING: Square(0, 1),
            _BLACK_KING: Square(7, 7),
            _BLOCKER_A1: Square(0, 0),
            _BLOCKER_A3: Square(0, 2),
            _BLOCKER_B2: Square(1, 1),
            _BLOCKER_B3: Square(1, 2),
            _KNIGHT: Square(2, 2),
        },
        cooldown=frozenset(),
        reservations_white=(),
        reservations_black=(),
        bookkeeping=Bookkeeping(
            castling_rights=_NO_CASTLING,
            repetition_ledger={},
            no_progress_counter=0,
            phase_index=0,
        ),
    )


def test_matching_pennies_dodge_matches_solve_exact() -> None:
    exact_value = solve_exact(
        _dodge_state(), RULESET, max_nodes=5000, max_depth=1
    ).root.value
    result = solve_by_double_oracle(_dodge_state(), RULESET)
    assert result.converged
    assert result.exploitability_bound == 0.0
    assert result.value == pytest.approx(exact_value, abs=1e-9)


# --- three_way_dodge -----------------------------------------------------

_T_WHITE_KING = Token(id=1, color=Color.WHITE, typ="k")
_T_BLOCKER_A1 = Token(id=10, color=Color.WHITE, typ="p")
_T_BLOCKER_A3 = Token(id=11, color=Color.WHITE, typ="p")
_T_BLOCKER_B2 = Token(id=12, color=Color.WHITE, typ="p")
_T_BLACK_KING = Token(id=2, color=Color.BLACK, typ="k")
_T_KNIGHT_1 = Token(id=4, color=Color.BLACK, typ="n")
_T_KNIGHT_2 = Token(id=5, color=Color.BLACK, typ="n")


def _three_way_state() -> State:
    return State(
        board={
            _T_WHITE_KING: _sq("a2"),
            _T_BLOCKER_A1: _sq("a1"),
            _T_BLOCKER_A3: _sq("a3"),
            _T_BLOCKER_B2: _sq("b2"),
            _T_BLACK_KING: _sq("h8"),
            _T_KNIGHT_1: _sq("c3"),
            _T_KNIGHT_2: _sq("d4"),
        },
        cooldown=frozenset(),
        reservations_white=(),
        reservations_black=(),
        bookkeeping=Bookkeeping(
            castling_rights=_NO_CASTLING,
            repetition_ledger={},
            no_progress_counter=0,
            phase_index=0,
        ),
    )


def test_three_way_dodge_matches_solve_exact() -> None:
    exact_value = solve_exact(
        _three_way_state(), RULESET, max_nodes=20_000, max_depth=1
    ).root.value
    result = solve_by_double_oracle(_three_way_state(), RULESET)
    assert result.converged
    assert result.exploitability_bound == 0.0
    assert result.value == pytest.approx(exact_value, abs=1e-9)


# --- dominant_strategy_contrast -------------------------------------------

_C_WHITE_KING = Token(id=1, color=Color.WHITE, typ="k")
_C_BLOCKER_A1 = Token(id=10, color=Color.WHITE, typ="p")
_C_BLOCKER_A3 = Token(id=11, color=Color.WHITE, typ="p")
_C_BLOCKER_B2 = Token(id=12, color=Color.WHITE, typ="p")
_C_BLOCKER_B3 = Token(id=13, color=Color.WHITE, typ="p")
_C_BLACK_KING = Token(id=2, color=Color.BLACK, typ="k")
_C_KNIGHT = Token(id=4, color=Color.BLACK, typ="n")


def _contrast_state() -> State:
    return State(
        board={
            _C_WHITE_KING: _sq("a2"),
            _C_BLOCKER_A1: _sq("a1"),
            _C_BLOCKER_A3: _sq("a3"),
            _C_BLOCKER_B2: _sq("b2"),
            _C_BLOCKER_B3: _sq("b3"),
            _C_BLACK_KING: _sq("h8"),
            _C_KNIGHT: _sq("d2"),
        },
        cooldown=frozenset(),
        reservations_white=(),
        reservations_black=(),
        bookkeeping=Bookkeeping(
            castling_rights=_NO_CASTLING,
            repetition_ledger={},
            no_progress_counter=0,
            phase_index=0,
        ),
    )


def test_dominant_strategy_contrast_matches_solve_exact() -> None:
    exact_value = solve_exact(
        _contrast_state(), RULESET, max_nodes=20_000, max_depth=1
    ).root.value
    result = solve_by_double_oracle(_contrast_state(), RULESET)
    assert result.converged
    assert result.exploitability_bound == 0.0
    assert result.value == pytest.approx(exact_value, abs=1e-9)


def test_a_deliberately_low_iteration_cap_reports_an_honest_nonzero_bound() -> None:
    # Confirms the bound is a real, reported number on non-convergence,
    # not silently defaulting to 0.0 the way a bug could easily produce.
    result = solve_by_double_oracle(_dodge_state(), RULESET, max_iterations=1)
    assert not result.converged
    assert result.exploitability_bound > 0.0
