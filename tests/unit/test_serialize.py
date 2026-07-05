from __future__ import annotations

from conftest import make_bookkeeping

from simult_chess.core.types import Color, Square, State, Token
from simult_chess.referee.serialize import public_position_key, serialize_state


def test_public_position_key_ignores_token_identity_and_reservations() -> None:
    knight_a = Token(id=1, color=Color.WHITE, typ="n")
    knight_b = Token(id=2, color=Color.WHITE, typ="n")
    state_a = State(
        board={knight_a: Square(1, 0)},
        cooldown=frozenset(),
        reservations_white=(),
        reservations_black=(),
        bookkeeping=make_bookkeeping(),
    )
    rook = Token(id=3, color=Color.BLACK, typ="r")
    state_b = State(
        board={knight_b: Square(1, 0)},
        cooldown=frozenset(),
        reservations_white=(),
        reservations_black=(),
        bookkeeping=make_bookkeeping(),
    )
    assert public_position_key(state_a) == public_position_key(state_b)

    state_with_extra_piece = State(
        board={knight_b: Square(1, 0), rook: Square(2, 0)},
        cooldown=frozenset(),
        reservations_white=(),
        reservations_black=(),
        bookkeeping=make_bookkeeping(),
    )
    assert public_position_key(state_a) != public_position_key(state_with_extra_piece)


def test_public_position_key_reflects_cooldown_status() -> None:
    token = Token(id=1, color=Color.WHITE, typ="n")
    state_active = State(
        board={token: Square(1, 0)},
        cooldown=frozenset(),
        reservations_white=(),
        reservations_black=(),
        bookkeeping=make_bookkeeping(),
    )
    state_cooled = State(
        board={token: Square(1, 0)},
        cooldown=frozenset({token}),
        reservations_white=(),
        reservations_black=(),
        bookkeeping=make_bookkeeping(),
    )
    assert public_position_key(state_active) != public_position_key(state_cooled)


def test_serialize_state_structure(minimal_state: State) -> None:
    payload = serialize_state(minimal_state)
    assert set(payload.keys()) == {
        "board",
        "cooldown",
        "reservations_white",
        "reservations_black",
        "bookkeeping",
    }
    assert len(payload["board"]) == 2
    assert payload["bookkeeping"]["phase_index"] == 0
