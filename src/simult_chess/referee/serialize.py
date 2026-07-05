"""State serialization and the public-position key K(β,C) (spec §10, dev-brief §3.3).

Full serialization is the payload of the repro dump (inv §9); the public
position key is the repetition-ledger coordinate and deliberately excludes
reservations and token identity, since two positions are indistinguishable
to a player if they differ only in which specific token instance sits on a
square, or in dormant (unfired) reservations.
"""

from __future__ import annotations

from typing import Any

from simult_chess.core.types import Reservation, State

PublicPositionKey = tuple[tuple[int, int, str, str, bool], ...]


def public_position_key(state: State) -> PublicPositionKey:
    """Compute :math:`K(s)=\\text{hash}(\\beta,C)`, excluding reservations.

    Parameters
    ----------
    state : State
        The state to key.

    Returns
    -------
    PublicPositionKey
        A hashable, collision-free tuple of ``(file, rank, color, type,
        is_cooled)`` per occupied square, sorted by square — the repetition
        coordinate of inv T3.
    """
    return tuple(
        sorted(
            (
                square.file,
                square.rank,
                token.color.value,
                token.typ,
                token in state.cooldown,
            )
            for token, square in state.board.items()
        )
    )


def _serialize_reservation(reservation: Reservation) -> dict[str, Any]:
    return {
        "defender_id": reservation.defender.id,
        "protege_id": reservation.protege.id,
        "age": list(reservation.age),
    }


def serialize_state(state: State) -> dict[str, Any]:
    """Serialize a full state to a JSON-able, replayable structure (inv §9).

    Parameters
    ----------
    state : State
        The state to serialize.

    Returns
    -------
    dict[str, Any]
        Nested primitives covering board, cooldown, reservations, and
        bookkeeping — sufficient, together with a ``RuleSet`` and programs,
        to replay a :math:`\\Phi` call deterministically.
    """
    return {
        "board": [
            {
                "token_id": token.id,
                "color": token.color.value,
                "type": token.typ,
                "square": [square.file, square.rank],
            }
            for token, square in sorted(state.board.items(), key=lambda kv: kv[0].id)
        ],
        "cooldown": sorted(token.id for token in state.cooldown),
        "reservations_white": [
            _serialize_reservation(r) for r in state.reservations_white
        ],
        "reservations_black": [
            _serialize_reservation(r) for r in state.reservations_black
        ],
        "bookkeeping": {
            "castling_rights": {
                "white_kingside": state.bookkeeping.castling_rights.white_kingside,
                "white_queenside": state.bookkeeping.castling_rights.white_queenside,
                "black_kingside": state.bookkeeping.castling_rights.black_kingside,
                "black_queenside": state.bookkeeping.castling_rights.black_queenside,
            },
            "repetition_ledger": [
                {"key": key, "count": count}
                for key, count in state.bookkeeping.repetition_ledger.items()
            ],
            "no_progress_counter": state.bookkeeping.no_progress_counter,
            "phase_index": state.bookkeeping.phase_index,
        },
    }
