"""Canonical tensor observation encoding (Phase 12 + ruling D5), pyspiel-free.

The single source of truth for the ``(21, 8, 8)`` planes + ``(7,)`` scalars
encoding whose semantics are documented in ``openspiel_adapter``'s module
docstring. Factored out here (numpy-only, no pyspiel, no torch) so both
``SimultChessObserver`` and the learning system (``simult_chess.learn.net``)
build the *identical* tensor from a native ``State`` without importing each
other's heavy dependency.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from simult_chess.core.types import Color, PieceType, Reservation, State
from simult_chess.rules.ruleset import RuleSet

_PIECE_TYPES: tuple[PieceType, ...] = ("p", "n", "b", "r", "q", "k")
_NUM_PLAYERS = 2

# 12 board + 1 cooldown + 4 reservation-actor + 4 reservation-pairing (D5).
NUM_PLANES = len(_PIECE_TYPES) * _NUM_PLAYERS + 1 + 4 + 4  # 21
NUM_SCALARS = 7
_MAX_OFFSET = 7.0  # normalizes a file/rank displacement in [-7, 7] to [-1, 1]


def fill_planes_scalars(
    planes: npt.NDArray[np.float32],
    scalars: npt.NDArray[np.float32],
    state: State,
    ruleset: RuleSet,
) -> None:
    """Write the encoding of native ``state`` into ``planes`` (shape
    ``(21, 8, 8)``) and ``scalars`` (shape ``(7,)``). ``planes`` is zeroed
    here; ``scalars`` is fully overwritten. See ``openspiel_adapter``'s module
    docstring for the plane/scalar semantics."""
    planes.fill(0.0)

    plane_of: dict[tuple[Color, PieceType], int] = {}
    index = 0
    for color in (Color.WHITE, Color.BLACK):
        for piece_type in _PIECE_TYPES:
            plane_of[(color, piece_type)] = index
            index += 1
    cooldown_plane = index
    index += 1
    white_defender_plane, white_protege_plane = index, index + 1
    black_defender_plane, black_protege_plane = index + 2, index + 3
    index += 4
    white_dq_dfile_plane, white_dq_drank_plane = index, index + 1
    black_dq_dfile_plane, black_dq_drank_plane = index + 2, index + 3

    for token, square in state.board.items():
        plane = plane_of[(token.color, token.typ)]
        planes[plane, square.rank, square.file] = 1.0
        if token in state.cooldown:
            planes[cooldown_plane, square.rank, square.file] = 1.0

    # oldest-per-defender-square reservation, for the pairing planes (D5): if a
    # square defends several proteges (spec R-multi-out), encode the offset to
    # the OLDEST active one, matching R-multi-in's firing order.
    pairing_src: dict[tuple[bool, tuple[int, int]], Reservation] = {}
    for reservation in (*state.reservations_white, *state.reservations_black):
        defender_square = state.board.get(reservation.defender)
        protege_square = state.board.get(reservation.protege)
        is_white = reservation.defender.color is Color.WHITE
        defender_plane = white_defender_plane if is_white else black_defender_plane
        protege_plane = white_protege_plane if is_white else black_protege_plane
        if defender_square is not None:
            planes[defender_plane, defender_square.rank, defender_square.file] = 1.0
        if protege_square is not None:
            planes[protege_plane, protege_square.rank, protege_square.file] = 1.0
        if defender_square is not None and protege_square is not None:
            key = (is_white, (defender_square.file, defender_square.rank))
            current = pairing_src.get(key)
            if current is None or reservation.age < current.age:
                pairing_src[key] = reservation

    for (is_white, _), reservation in pairing_src.items():
        defender_square = state.board[reservation.defender]
        protege_square = state.board[reservation.protege]
        dfile = (protege_square.file - defender_square.file) / _MAX_OFFSET
        drank = (protege_square.rank - defender_square.rank) / _MAX_OFFSET
        dfile_plane = white_dq_dfile_plane if is_white else black_dq_dfile_plane
        drank_plane = white_dq_drank_plane if is_white else black_dq_drank_plane
        planes[dfile_plane, defender_square.rank, defender_square.file] = dfile
        planes[drank_plane, defender_square.rank, defender_square.file] = drank

    rights = state.bookkeeping.castling_rights
    scalars[0] = float(rights.white_kingside)
    scalars[1] = float(rights.white_queenside)
    scalars[2] = float(rights.black_kingside)
    scalars[3] = float(rights.black_queenside)
    scalars[4] = float(state.bookkeeping.no_progress_counter)
    scalars[5] = float(state.bookkeeping.phase_index % 2)
    scalars[6] = float(ruleset.horizon)


def encode_state(
    state: State, ruleset: RuleSet
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """A fresh ``(planes, scalars)`` pair for native ``state`` -- the
    convenience form the learning system uses (the observer fills its own
    persistent pyspiel buffer via ``fill_planes_scalars``)."""
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    scalars = np.zeros(NUM_SCALARS, dtype=np.float32)
    fill_planes_scalars(planes, scalars, state, ruleset)
    return planes, scalars
