from __future__ import annotations

from conftest import build_state

from simult_chess.core.moves import extract_declared_moves
from simult_chess.core.types import Castle, Color, Move, Square, Token, Trajectory


def test_extract_declared_moves_indexes_within_each_owner() -> None:
    white_pawn = Token(id=1, color=Color.WHITE, typ="p")
    black_pawn = Token(id=2, color=Color.BLACK, typ="p")
    state = build_state({white_pawn: Square(4, 1), black_pawn: Square(4, 6)})
    white_trajectory = Trajectory(path=(Square(4, 1), Square(4, 2)))
    black_trajectory = Trajectory(path=(Square(4, 6), Square(4, 5)))
    program_white = (Move(token=white_pawn, trajectory=white_trajectory),)
    program_black = (Move(token=black_pawn, trajectory=black_trajectory),)
    declared = extract_declared_moves(state, program_white, program_black)
    assert len(declared) == 2
    white_move = next(m for m in declared if m.color is Color.WHITE)
    black_move = next(m for m in declared if m.color is Color.BLACK)
    assert white_move.index == 1
    assert black_move.index == 1
    assert white_move.kind == "move"


def test_extract_declared_moves_expands_castle_into_two_indexed_entries() -> None:
    king = Token(id=1, color=Color.WHITE, typ="k")
    rook = Token(id=2, color=Color.WHITE, typ="r")
    state = build_state({king: Square(4, 0), rook: Square(7, 0)})
    declared = extract_declared_moves(state, (Castle(side="king"),), ())
    assert len(declared) == 2
    king_move = next(m for m in declared if m.kind == "castle_king")
    rook_move = next(m for m in declared if m.kind == "castle_rook")
    assert king_move.index == 1
    assert rook_move.index == 2
    assert king_move.token == king
    assert rook_move.token == rook
    assert king_move.trajectory.destination == Square(6, 0)
    assert rook_move.trajectory.destination == Square(5, 0)
