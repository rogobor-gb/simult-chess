from __future__ import annotations

from conftest import build_state

from simult_chess.core.types import Color, Square, Token
from simult_chess.referee.setup import standard_starting_state
from simult_chess.ui.board_render import render_board


def test_render_board_starting_position_has_full_material() -> None:
    state = standard_starting_state()
    text = render_board(state)
    assert text.count("P") == 8  # white pawns
    assert text.count("p") == 8  # black pawns
    assert "K" in text
    assert "k" in text


def test_render_board_marks_cooled_tokens() -> None:
    rook = Token(id=1, color=Color.WHITE, typ="r")
    king = Token(id=2, color=Color.WHITE, typ="k")
    black_king = Token(id=3, color=Color.BLACK, typ="k")
    state = build_state(
        {rook: Square(0, 0), king: Square(4, 0), black_king: Square(4, 7)},
        cooldown=frozenset({rook}),
    )
    text = render_board(state)
    assert "R*" in text


def test_render_board_perspective_flips_orientation() -> None:
    state = standard_starting_state()
    white_view = render_board(state, perspective=Color.WHITE)
    black_view = render_board(state, perspective=Color.BLACK)
    assert white_view != black_view
    assert white_view.splitlines()[0].strip().startswith("8")
    assert black_view.splitlines()[0].strip().startswith("1")
