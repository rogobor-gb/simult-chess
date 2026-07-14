from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from conftest import build_state  # noqa: E402

from simult_chess.core.types import Color, Move, Square, Token, Trajectory  # noqa: E402
from simult_chess.rules.ruleset import RuleSet  # noqa: E402
from simult_chess.solver.stage_matrix import (  # noqa: E402
    build_stage_matrix,
    material_difference,
    payoff,
)

RULESET = RuleSet()


def test_material_difference_is_zero_on_a_bare_king_pair() -> None:
    king_w = Token(id=1, color=Color.WHITE, typ="k")
    king_b = Token(id=2, color=Color.BLACK, typ="k")
    state = build_state({king_w: Square(0, 0), king_b: Square(7, 7)})
    assert material_difference(state) == 0.0


def test_material_difference_favors_white_with_extra_material() -> None:
    king_w = Token(id=1, color=Color.WHITE, typ="k")
    king_b = Token(id=2, color=Color.BLACK, typ="k")
    queen_w = Token(id=3, color=Color.WHITE, typ="q")
    state = build_state(
        {king_w: Square(0, 0), king_b: Square(7, 7), queen_w: Square(3, 3)}
    )
    assert material_difference(state) == 9.0


def test_payoff_uses_terminal_values_over_material() -> None:
    king_w = Token(id=1, color=Color.WHITE, typ="k")
    queen_b = Token(id=2, color=Color.BLACK, typ="q")
    state = build_state({king_w: Square(0, 0), queen_b: Square(7, 7)})
    # Material clearly favors Black, but a terminal outcome always wins.
    assert payoff(state, "white_wins") == 1.0
    assert payoff(state, "black_wins") == -1.0
    assert payoff(state, "draw") == 0.0
    assert payoff(state, "ongoing") == material_difference(state)


def test_build_stage_matrix_has_the_right_shape_and_matches_manual_phi() -> None:
    king_w = Token(id=1, color=Color.WHITE, typ="k")
    king_b = Token(id=2, color=Color.BLACK, typ="k")
    pawn_w = Token(id=3, color=Color.WHITE, typ="p")
    state = build_state(
        {king_w: Square(0, 0), king_b: Square(7, 7), pawn_w: Square(4, 1)}
    )
    push_one = Move(
        token=pawn_w, trajectory=Trajectory(path=(Square(4, 1), Square(4, 2)))
    )
    push_two = Move(
        token=pawn_w,
        trajectory=Trajectory(path=(Square(4, 1), Square(4, 2), Square(4, 3))),
    )
    king_step = Move(
        token=king_b, trajectory=Trajectory(path=(Square(7, 7), Square(6, 7)))
    )
    support_white = ((push_one,), (push_two,))
    support_black = ((king_step,),)

    matrix = build_stage_matrix(state, support_white, support_black, RULESET)

    assert matrix.shape == (2, 1)
    # Neither program interacts with the other -- both resolve to ordinary
    # material-difference payoffs (still +1 for the lone extra pawn).
    assert matrix[0, 0] == 1.0
    assert matrix[1, 0] == 1.0
