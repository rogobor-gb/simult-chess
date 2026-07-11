from __future__ import annotations

import itertools

from conftest import build_state

from simult_chess.core.phi import phi
from simult_chess.core.types import Color, Move, Reservation, Square, Token, Trajectory
from simult_chess.rules.ruleset import RuleSet

RULESET = RuleSet()


def test_m4_attacker_declaration_order_does_not_affect_defense_outcome() -> None:
    """M4 — the defensive outcome is invariant to the attacker's intra-program
    declaration order (spec §6.4 worked example, INVARIANTS.md M4).

    Fixed fixture (the d4/e3 worked example) with all k! = 2 orderings of
    the attacker's two captures, per INVARIANTS.md's own check description
    ("permute the attacker's intra-program order over all k! orderings").
    """
    white_king = Token(id=100, color=Color.WHITE, typ="k")
    black_king = Token(id=200, color=Color.BLACK, typ="k")
    d4_pawn = Token(id=1, color=Color.WHITE, typ="p")
    e3_pawn = Token(id=2, color=Color.WHITE, typ="p")
    rook1 = Token(id=3, color=Color.BLACK, typ="r")
    rook2 = Token(id=4, color=Color.BLACK, typ="r")
    state = build_state(
        {
            white_king: Square(0, 0),
            black_king: Square(7, 7),
            d4_pawn: Square(3, 3),
            e3_pawn: Square(4, 2),
            rook1: Square(3, 7),
            rook2: Square(4, 6),
        },
        reservations_white=(
            Reservation(defender=e3_pawn, protege=d4_pawn, age=(0, 0)),
        ),
    )
    rook1_path = (Square(3, 7), Square(3, 6), Square(3, 5), Square(3, 4), Square(3, 3))
    rook2_path = (Square(4, 6), Square(4, 5), Square(4, 4), Square(4, 3), Square(4, 2))
    rook1_move = Move(token=rook1, trajectory=Trajectory(path=rook1_path))
    rook2_move = Move(token=rook2, trajectory=Trajectory(path=rook2_path))
    king_trajectory = Trajectory(path=(Square(0, 0), Square(0, 1)))
    program_white = (Move(token=white_king, trajectory=king_trajectory),)

    results = [
        phi(state, program_white, ordering, RULESET).state
        for ordering in itertools.permutations((rook1_move, rook2_move))
    ]

    assert all(result == results[0] for result in results)
    board_ids = {t.id for t in results[0].board}
    assert d4_pawn.id not in board_ids  # the defense trades, it doesn't prevent
    assert rook1.id not in board_ids
