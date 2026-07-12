from __future__ import annotations

import pytest
from conftest import build_state

from simult_chess.core.types import (
    Cancel,
    Castle,
    Color,
    Move,
    Reservation,
    Reserve,
    Square,
    Token,
)
from simult_chess.referee.setup import standard_starting_state
from simult_chess.ui import notation

WHITE_KING = Token(id=100, color=Color.WHITE, typ="k")
BLACK_KING = Token(id=200, color=Color.BLACK, typ="k")


def test_str_to_square_and_format_square_round_trip() -> None:
    assert notation.str_to_square("e4") == Square(file=4, rank=3)
    assert notation.format_square(Square(file=0, rank=0)) == "a1"
    assert notation.format_square(Square(file=7, rank=7)) == "h8"


def test_str_to_square_rejects_malformed_text() -> None:
    with pytest.raises(notation.NotationError):
        notation.str_to_square("z9")


def test_parse_program_short_move_infers_origin() -> None:
    state = standard_starting_state()
    program = notation.parse_program("Nf3", state, Color.WHITE)
    assert len(program) == 1
    move = program[0]
    assert isinstance(move, Move)
    assert move.token.typ == "n"
    assert move.trajectory.destination == Square(5, 2)  # f3


def test_parse_program_pawn_short_move_no_piece_letter() -> None:
    state = standard_starting_state()
    program = notation.parse_program("e4", state, Color.WHITE)
    move = program[0]
    assert isinstance(move, Move)
    assert move.token.typ == "p"
    assert move.trajectory.destination == Square(4, 3)


def test_parse_program_full_coordinate_form() -> None:
    state = standard_starting_state()
    program = notation.parse_program("e2e4", state, Color.WHITE)
    move = program[0]
    assert isinstance(move, Move)
    assert move.trajectory.origin == Square(4, 1)
    assert move.trajectory.destination == Square(4, 3)


def test_parse_program_capture_x_marker_is_ignored() -> None:
    white_rook = Token(id=1, color=Color.WHITE, typ="r")
    black_knight = Token(id=2, color=Color.BLACK, typ="n")
    state = build_state(
        {
            WHITE_KING: Square(0, 0),
            BLACK_KING: Square(7, 7),
            white_rook: Square(0, 3),
            black_knight: Square(3, 3),
        }
    )
    program = notation.parse_program("Rxd4", state, Color.WHITE)
    move = program[0]
    assert isinstance(move, Move)
    assert move.token is white_rook
    assert move.trajectory.destination == Square(3, 3)


def test_parse_program_promotion_suffix() -> None:
    pawn = Token(id=1, color=Color.WHITE, typ="p")
    state = build_state(
        {WHITE_KING: Square(0, 0), BLACK_KING: Square(7, 7), pawn: Square(0, 6)}
    )
    program = notation.parse_program("a8=Q", state, Color.WHITE)
    move = program[0]
    assert isinstance(move, Move)
    assert move.promotion == "q"


def test_parse_program_ambiguous_short_move_raises() -> None:
    knight_a = Token(id=1, color=Color.WHITE, typ="n")
    knight_b = Token(id=2, color=Color.WHITE, typ="n")
    state = build_state(
        {
            WHITE_KING: Square(0, 0),
            BLACK_KING: Square(7, 7),
            knight_a: Square(1, 0),  # b1
            knight_b: Square(5, 0),  # f1
        }
    )
    with pytest.raises(notation.NotationError, match="ambiguous"):
        notation.parse_program("Nd2", state, Color.WHITE)


def test_parse_program_unreachable_square_raises() -> None:
    state = standard_starting_state()
    with pytest.raises(notation.NotationError):
        notation.parse_program("Nf6", state, Color.WHITE)


def test_parse_program_castle() -> None:
    state = standard_starting_state()
    # clear the kingside squares for white so castling is geometrically legal
    knight = next(t for t, sq in state.board.items() if sq == Square(6, 0))
    bishop = next(t for t, sq in state.board.items() if sq == Square(5, 0))
    board = dict(state.board)
    del board[knight]
    del board[bishop]
    state = build_state(board, castling_rights=state.bookkeeping.castling_rights)

    program = notation.parse_program("O-O", state, Color.WHITE)
    assert program == (Castle(side="king"),)


def test_parse_program_reserve_current_square() -> None:
    king = Token(id=1, color=Color.WHITE, typ="k")
    rook = Token(id=2, color=Color.WHITE, typ="r")
    state = build_state(
        {king: Square(4, 3), BLACK_KING: Square(7, 7), rook: Square(4, 4)}
    )
    program = notation.parse_program("e5 def Ke4", state, Color.WHITE)
    reserve = program[0]
    assert isinstance(reserve, Reserve)
    assert reserve.defender is king
    assert reserve.protege is rook


def test_parse_program_reserve_aggressive_dual_future_destination() -> None:
    king = Token(id=1, color=Color.WHITE, typ="k")
    pawn = Token(id=2, color=Color.WHITE, typ="p")
    state = build_state(
        {king: Square(4, 3), BLACK_KING: Square(7, 7), pawn: Square(3, 4)}  # d5
    )
    # pawn pushes d5-d6, king (already on e4) reserves the pawn at its NEW square d6
    program = notation.parse_program("d6; d6 def Ke4", state, Color.WHITE)
    move, reserve = program
    assert isinstance(move, Move)
    assert isinstance(reserve, Reserve)
    assert reserve.protege is pawn
    assert reserve.defender is king


def test_parse_program_cancel_by_index() -> None:
    king = Token(id=1, color=Color.WHITE, typ="k")
    rook = Token(id=2, color=Color.WHITE, typ="r")
    reservation = Reservation(defender=king, protege=rook, age=(0, 0))
    state = build_state(
        {king: Square(4, 3), BLACK_KING: Square(7, 7), rook: Square(4, 4)},
        reservations_white=(reservation,),
    )
    program = notation.parse_program("cancel 0", state, Color.WHITE)
    assert program == (Cancel(reservation=reservation),)


def test_parse_program_cancel_out_of_range_raises() -> None:
    state = build_state({WHITE_KING: Square(0, 0), BLACK_KING: Square(7, 7)})
    with pytest.raises(notation.NotationError):
        notation.parse_program("cancel 0", state, Color.WHITE)


def test_format_program_round_trips_a_move() -> None:
    state = standard_starting_state()
    program = notation.parse_program("Nf3", state, Color.WHITE)
    assert notation.format_program(program, state, Color.WHITE) == "Ng1f3"


def test_format_program_round_trips_castle() -> None:
    state = standard_starting_state()
    text = notation.format_program((Castle(side="queen"),), state, Color.WHITE)
    assert text == "O-O-O"
