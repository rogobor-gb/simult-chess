"""Fixed factored action grid (Phase 13b, docs/LEARNING_DESIGN.md §3.3, D1/D4).

A pyspiel action integer is a *state-dependent* index into an enumeration, so a
softmax over it is meaningless across states (adapter module docstring). The
learned policy instead lives on a **fixed, state-independent** per-slot grid,
masked to the legal set per state -- the AlphaZero approach, adapted to this
game's four action kinds.

Per-slot layout (``SLOT_SIZE`` = 9026 entries), by contiguous block:

===============  ==================================================  =======
Block            Encoding                                            Size
===============  ==================================================  =======
``Move``         ``from_square (64) x move_type (76)``               4864
``Castle``       2 dedicated entries ``{king, queen}`` (ruling A3)   2
``Reserve``      ``defender_square (64) x protege_square (64)``      4096
``Cancel``       ``protege_square (64)``, oldest-valid tie-break     64
===============  ==================================================  =======

``move_type`` (76) splits as 56 sliding (8 directions x 7 distances) + 8 knight
+ 12 promotion (3 last-rank file-directions x 4 promo types ``{n,b,r,q}``). A
last-rank pawn move is encoded **only** via the promotion sub-block (promotion
is explicit and forced, spec §6.5), so a non-promotion move never lands a pawn
on the last rank. Squares index as ``rank * 8 + file``.

Encoding is a pure function of ``(action, state)`` -- ``state`` only supplies the
current squares of a ``Reserve``/``Cancel``'s tokens. It is injective on the
distinct legal actions of a state, with one deliberate exception matching the
engine's own disambiguation: several standing reservations sharing a protege
square collapse to the single ``Cancel`` entry for that square, decoded to the
**oldest** (R-multi-in's oldest-valid-fires, spec §6.4). The slot-2 head adds
one further entry, ``NO_SECOND_INDEX`` (= ``SLOT_SIZE``), for single-action
programs, so a slot-2 logit vector has ``SLOT_SIZE + 1`` entries.
"""

from __future__ import annotations

from simult_chess.core.types import (
    Action,
    Cancel,
    Castle,
    Move,
    PieceType,
    State,
)

_BOARD = 8
_SQUARES = _BOARD * _BOARD  # 64

# 8 sliding directions, indexed 0..7 (N, NE, E, SE, S, SW, W, NW).
_SLIDE_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1),
)
_SLIDE_DIR_INDEX: dict[tuple[int, int], int] = {
    d: i for i, d in enumerate(_SLIDE_DIRECTIONS)
}
_MAX_DISTANCE = 7
_SLIDE_TYPES = len(_SLIDE_DIRECTIONS) * _MAX_DISTANCE  # 56

# 8 knight deltas, same ordering as core.geometry._KNIGHT_DELTAS.
_KNIGHT_DELTAS: tuple[tuple[int, int], ...] = (
    (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2),
)
_KNIGHT_INDEX: dict[tuple[int, int], int] = {
    d: i for i, d in enumerate(_KNIGHT_DELTAS)
}
_KNIGHT_TYPES = len(_KNIGHT_DELTAS)  # 8

# 12 promotion entries: 3 last-rank file-directions (dfile in {-1,0,+1}) x 4
# promo types. dfile + 1 maps to a direction index in {0,1,2}.
_PROMO_TYPES: tuple[PieceType, ...] = ("n", "b", "r", "q")
_PROMO_INDEX: dict[PieceType, int] = {t: i for i, t in enumerate(_PROMO_TYPES)}
_PROMOTION_TYPES = 3 * len(_PROMO_TYPES)  # 12

MOVE_TYPES = _SLIDE_TYPES + _KNIGHT_TYPES + _PROMOTION_TYPES  # 76
_KNIGHT_BASE = _SLIDE_TYPES  # 56
_PROMO_BASE = _SLIDE_TYPES + _KNIGHT_TYPES  # 64

MOVE_BLOCK = _SQUARES * MOVE_TYPES  # 4864
CASTLE_OFFSET = MOVE_BLOCK  # 4864
CASTLE_SIZE = 2
RESERVE_OFFSET = CASTLE_OFFSET + CASTLE_SIZE  # 4866
RESERVE_SIZE = _SQUARES * _SQUARES  # 4096
CANCEL_OFFSET = RESERVE_OFFSET + RESERVE_SIZE  # 8962
CANCEL_SIZE = _SQUARES  # 64

SLOT_SIZE = CANCEL_OFFSET + CANCEL_SIZE  # 9026
NO_SECOND_INDEX = SLOT_SIZE  # slot-2 only; slot-2 logits have SLOT_SIZE + 1 entries


def square_index(file: int, rank: int) -> int:
    """The flat index ``rank * 8 + file`` of a board square."""
    return rank * _BOARD + file


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _move_type(move: Move) -> int:
    origin = move.trajectory.origin
    destination = move.trajectory.destination
    dfile = destination.file - origin.file
    drank = destination.rank - origin.rank

    if move.promotion is not None:
        # Forced, explicit promotion (spec §6.5): the promo sub-block, keyed on
        # the file-direction (dfile in {-1,0,+1}) and the chosen type.
        promo_direction = dfile + 1
        return (
            _PROMO_BASE
            + promo_direction * len(_PROMO_TYPES)
            + _PROMO_INDEX[move.promotion]
        )
    if move.trajectory.is_jump:
        return _KNIGHT_BASE + _KNIGHT_INDEX[(dfile, drank)]
    distance = max(abs(dfile), abs(drank))
    direction = (_sign(dfile), _sign(drank))
    return _SLIDE_DIR_INDEX[direction] * _MAX_DISTANCE + (distance - 1)


def encode_action(action: Action, state: State) -> int:
    """The fixed-grid index of ``action`` at ``state`` (see module docstring).

    ``state`` supplies only the current squares of a ``Reserve``/``Cancel``'s
    tokens; ``Move``/``Castle`` encode from the action alone.
    """
    if isinstance(action, Move):
        origin = action.trajectory.origin
        return square_index(origin.file, origin.rank) * MOVE_TYPES + _move_type(action)
    if isinstance(action, Castle):
        return CASTLE_OFFSET + (0 if action.side == "king" else 1)
    if isinstance(action, Cancel):
        protege = state.board[action.reservation.protege]
        return CANCEL_OFFSET + square_index(protege.file, protege.rank)
    # Reserve.
    defender = state.board[action.defender]
    protege = state.board[action.protege]
    return (
        RESERVE_OFFSET
        + square_index(defender.file, defender.rank) * _SQUARES
        + square_index(protege.file, protege.rank)
    )
