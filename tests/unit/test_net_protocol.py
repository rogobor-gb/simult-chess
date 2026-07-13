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
    Trajectory,
)
from simult_chess.net import protocol

WHITE_KING = Token(id=100, color=Color.WHITE, typ="k")
BLACK_KING = Token(id=200, color=Color.BLACK, typ="k")


def test_move_round_trips() -> None:
    pawn = Token(id=1, color=Color.WHITE, typ="p")
    state = build_state(
        {WHITE_KING: Square(0, 0), BLACK_KING: Square(7, 7), pawn: Square(4, 1)}
    )
    move = Move(
        token=pawn,
        trajectory=Trajectory(path=(Square(4, 1), Square(4, 2), Square(4, 3))),
        promotion=None,
    )
    data = protocol.serialize_action(move, state, Color.WHITE)
    restored = protocol.deserialize_action(data, state, Color.WHITE)
    assert restored == move


def test_move_with_promotion_round_trips() -> None:
    pawn = Token(id=1, color=Color.WHITE, typ="p")
    state = build_state(
        {WHITE_KING: Square(0, 0), BLACK_KING: Square(7, 7), pawn: Square(0, 6)}
    )
    move = Move(
        token=pawn,
        trajectory=Trajectory(path=(Square(0, 6), Square(0, 7))),
        promotion="q",
    )
    data = protocol.serialize_action(move, state, Color.WHITE)
    restored = protocol.deserialize_action(data, state, Color.WHITE)
    assert restored == move


def test_knight_jump_round_trips() -> None:
    knight = Token(id=1, color=Color.WHITE, typ="n")
    state = build_state(
        {WHITE_KING: Square(0, 0), BLACK_KING: Square(7, 7), knight: Square(1, 0)}
    )
    move = Move(
        token=knight,
        trajectory=Trajectory(path=(Square(1, 0), Square(2, 2)), is_jump=True),
    )
    data = protocol.serialize_action(move, state, Color.WHITE)
    assert data["is_jump"] is True
    restored = protocol.deserialize_action(data, state, Color.WHITE)
    assert restored == move


def test_reserve_round_trips() -> None:
    king = Token(id=1, color=Color.WHITE, typ="k")
    rook = Token(id=2, color=Color.WHITE, typ="r")
    state = build_state(
        {king: Square(4, 3), BLACK_KING: Square(7, 7), rook: Square(4, 4)}
    )
    reserve = Reserve(defender=king, protege=rook)
    data = protocol.serialize_action(reserve, state, Color.WHITE)
    restored = protocol.deserialize_action(data, state, Color.WHITE)
    assert restored == reserve


def test_castle_round_trips() -> None:
    state = build_state({WHITE_KING: Square(4, 0), BLACK_KING: Square(7, 7)})
    castle = Castle(side="queen")
    data = protocol.serialize_action(castle, state, Color.WHITE)
    restored = protocol.deserialize_action(data, state, Color.WHITE)
    assert restored == castle


def test_cancel_round_trips_by_index() -> None:
    king = Token(id=1, color=Color.WHITE, typ="k")
    rook = Token(id=2, color=Color.WHITE, typ="r")
    reservation = Reservation(defender=king, protege=rook, age=(0, 0))
    state = build_state(
        {king: Square(4, 3), BLACK_KING: Square(7, 7), rook: Square(4, 4)},
        reservations_white=(reservation,),
    )
    cancel = Cancel(reservation=reservation)
    data = protocol.serialize_action(cancel, state, Color.WHITE)
    assert data == {"kind": "cancel", "index": 0}
    restored = protocol.deserialize_action(data, state, Color.WHITE)
    assert restored == cancel


def test_program_round_trips_end_to_end() -> None:
    pawn = Token(id=1, color=Color.WHITE, typ="p")
    king = Token(id=2, color=Color.WHITE, typ="k")
    rook = Token(id=3, color=Color.WHITE, typ="r")
    state = build_state(
        {
            pawn: Square(4, 1),
            king: Square(4, 0),
            BLACK_KING: Square(7, 7),
            rook: Square(0, 0),
        }
    )
    program = (
        Move(token=pawn, trajectory=Trajectory(path=(Square(4, 1), Square(4, 2)))),
        Reserve(defender=king, protege=rook),
    )
    data = protocol.serialize_program(program, state, Color.WHITE)
    restored = protocol.deserialize_program(data, state, Color.WHITE)
    assert restored == program


def test_deserialize_unknown_token_id_raises() -> None:
    state = build_state({WHITE_KING: Square(0, 0), BLACK_KING: Square(7, 7)})
    data = [
        {
            "kind": "move",
            "token_id": 999,
            "path": [[0, 0], [0, 1]],
            "is_jump": False,
            "promotion": None,
        }
    ]
    with pytest.raises(protocol.ProtocolError):
        protocol.deserialize_program(data, state, Color.WHITE)
