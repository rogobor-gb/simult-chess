"""The standard 32-piece chess starting position (spec's base setup)."""

from __future__ import annotations

from simult_chess.core.types import (
    Bookkeeping,
    CastlingRights,
    Color,
    PieceType,
    Square,
    State,
    Token,
)

_BACK_RANK_ORDER: tuple[PieceType, ...] = ("r", "n", "b", "q", "k", "b", "n", "r")


def standard_starting_state() -> State:
    """The standard chess starting position, with fresh bookkeeping (phase 0)."""
    board: dict[Token, Square] = {}
    next_id = 1
    for color, back_rank, pawn_rank in ((Color.WHITE, 0, 1), (Color.BLACK, 7, 6)):
        for file, piece_type in enumerate(_BACK_RANK_ORDER):
            token = Token(id=next_id, color=color, typ=piece_type)
            board[token] = Square(file, back_rank)
            next_id += 1
        for file in range(8):
            board[Token(id=next_id, color=color, typ="p")] = Square(file, pawn_rank)
            next_id += 1

    return State(
        board=board,
        cooldown=frozenset(),
        reservations_white=(),
        reservations_black=(),
        bookkeeping=Bookkeeping(
            castling_rights=CastlingRights(),
            repetition_ledger={},
            no_progress_counter=0,
            phase_index=0,
        ),
    )
