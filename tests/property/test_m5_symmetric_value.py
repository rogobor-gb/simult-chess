"""M5 -- Symmetric-position value antisymmetry (INVARIANTS.md M5, spec section
8.1-8.2). Activated in Phase 10 (A8): this is the permanent scope -- chi-
symmetric fixtures with restricted, chi-closed action supports, never
arbitrary sweep states (see INVARIANTS.md's own scope note on M5).
"""

from __future__ import annotations

import random

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")

from simult_chess.core.collision import mirror_program, mirror_state  # noqa: E402
from simult_chess.core.types import (  # noqa: E402
    Bookkeeping,
    CastlingRights,
    Color,
    Square,
    State,
    Token,
)
from simult_chess.rules.ruleset import RuleSet  # noqa: E402
from simult_chess.solver.lp import ATOL, solve_zero_sum  # noqa: E402
from simult_chess.solver.stage_matrix import build_stage_matrix  # noqa: E402
from simult_chess.solver.supports import enumerate_support  # noqa: E402

RULESET = RuleSet()
_BACK_RANK_ORDER = ("r", "n", "b", "q", "k", "b", "n", "r")
_NO_CASTLING = CastlingRights(
    white_kingside=False,
    white_queenside=False,
    black_kingside=False,
    black_queenside=False,
)


def _state(board: dict[Token, Square], castling_rights: CastlingRights) -> State:
    return State(
        board=board,
        cooldown=frozenset(),
        reservations_white=(),
        reservations_black=(),
        bookkeeping=Bookkeeping(
            castling_rights=castling_rights,
            repetition_ledger={},
            no_progress_counter=0,
            phase_index=0,
        ),
    )


def _symmetric_starting_state() -> State:
    """The standard start, with token ids shared across each White/Black
    mirrored pair (file-for-file) so mirror_state(s) == s holds exactly --
    the literal s = chi(s) condition M5 requires. `referee.setup`'s own
    `standard_starting_state` assigns globally sequential ids instead
    (mirror_token preserves id and flips color only, so that scheme does
    not satisfy s = chi(s)), so this fixture is purpose-built here.
    """
    board: dict[Token, Square] = {}
    for file, piece_type in enumerate(_BACK_RANK_ORDER):
        board[Token(id=file, color=Color.WHITE, typ=piece_type)] = Square(file, 0)
        board[Token(id=file, color=Color.BLACK, typ=piece_type)] = Square(file, 7)
    for file in range(8):
        pawn_id = 8 + file
        board[Token(id=pawn_id, color=Color.WHITE, typ="p")] = Square(file, 1)
        board[Token(id=pawn_id, color=Color.BLACK, typ="p")] = Square(file, 6)
    return _state(board, CastlingRights())


def _symmetric_midgame_fixture_knight_pawn() -> State:
    king_id, knight_id, pawn_id = 1, 2, 3
    board = {
        Token(id=king_id, color=Color.WHITE, typ="k"): Square(0, 0),
        Token(id=king_id, color=Color.BLACK, typ="k"): Square(0, 7),
        Token(id=knight_id, color=Color.WHITE, typ="n"): Square(3, 3),
        Token(id=knight_id, color=Color.BLACK, typ="n"): Square(3, 4),
        Token(id=pawn_id, color=Color.WHITE, typ="p"): Square(4, 1),
        Token(id=pawn_id, color=Color.BLACK, typ="p"): Square(4, 6),
    }
    return _state(board, _NO_CASTLING)


def _symmetric_midgame_fixture_rook_pawn() -> State:
    king_id, rook_id, pawn_id = 1, 2, 3
    board = {
        Token(id=king_id, color=Color.WHITE, typ="k"): Square(0, 0),
        Token(id=king_id, color=Color.BLACK, typ="k"): Square(0, 7),
        Token(id=rook_id, color=Color.WHITE, typ="r"): Square(7, 0),
        Token(id=rook_id, color=Color.BLACK, typ="r"): Square(7, 7),
        Token(id=pawn_id, color=Color.WHITE, typ="p"): Square(4, 3),
        Token(id=pawn_id, color=Color.BLACK, typ="p"): Square(4, 4),
    }
    return _state(board, _NO_CASTLING)


_FIXTURES = {
    "standard_start": _symmetric_starting_state,
    "midgame_knight_pawn": _symmetric_midgame_fixture_knight_pawn,
    "midgame_rook_pawn": _symmetric_midgame_fixture_rook_pawn,
}


@pytest.mark.parametrize("fixture_name", sorted(_FIXTURES))
def test_m5_fixtures_are_actually_chi_symmetric(fixture_name: str) -> None:
    """Precondition: s = chi(s), the literal requirement inv M5 states."""
    state = _FIXTURES[fixture_name]()
    assert mirror_state(state) == state


@pytest.mark.parametrize("fixture_name", sorted(_FIXTURES))
def test_m5_stage_matrix_is_antisymmetric_with_value_zero(fixture_name: str) -> None:
    """M5 -- on a chi-symmetric fixture with chi-closed supports (Black's
    support is White's support mirrored, same index order), U = -U^T and
    val(U) = 0 within tolerance (INVARIANTS.md M5, spec section 8.1-8.2)."""
    state = _FIXTURES[fixture_name]()
    rng = random.Random(0)
    support_white = enumerate_support(state, Color.WHITE, RULESET, rng)
    support_black = tuple(mirror_program(program) for program in support_white)

    matrix = build_stage_matrix(state, support_white, support_black, RULESET)

    assert matrix == pytest.approx(-matrix.T, abs=ATOL)
    solution = solve_zero_sum(matrix)
    assert solution.value == pytest.approx(0.0, abs=1e-6)
