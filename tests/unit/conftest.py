from __future__ import annotations

from collections.abc import Mapping

import pytest

from simult_chess.core.types import (
    Bookkeeping,
    CastlingRights,
    Color,
    Reservation,
    Square,
    State,
    Token,
)


def make_bookkeeping(
    *, no_progress_counter: int = 0, phase_index: int = 0
) -> Bookkeeping:
    return Bookkeeping(
        castling_rights=CastlingRights(),
        repetition_ledger={},
        no_progress_counter=no_progress_counter,
        phase_index=phase_index,
    )


def build_state(
    board: Mapping[Token, Square],
    *,
    cooldown: frozenset[Token] = frozenset(),
    reservations_white: tuple[Reservation, ...] = (),
    reservations_black: tuple[Reservation, ...] = (),
    castling_rights: CastlingRights | None = None,
    no_progress_counter: int = 0,
    phase_index: int = 0,
) -> State:
    return State(
        board=board,
        cooldown=cooldown,
        reservations_white=reservations_white,
        reservations_black=reservations_black,
        bookkeeping=Bookkeeping(
            castling_rights=castling_rights or CastlingRights(),
            repetition_ledger={},
            no_progress_counter=no_progress_counter,
            phase_index=phase_index,
        ),
    )


@pytest.fixture
def white_king() -> Token:
    return Token(id=1, color=Color.WHITE, typ="k")


@pytest.fixture
def black_king() -> Token:
    return Token(id=2, color=Color.BLACK, typ="k")


@pytest.fixture
def minimal_state(white_king: Token, black_king: Token) -> State:
    return State(
        board={white_king: Square(4, 0), black_king: Square(4, 7)},
        cooldown=frozenset(),
        reservations_white=(),
        reservations_black=(),
        bookkeeping=make_bookkeeping(),
    )
