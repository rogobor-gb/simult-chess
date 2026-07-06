"""Geometry oracle: pseudo-legal trajectories on β only, spec §4.2.

No look-ahead: every function here reasons about a single, fixed board
snapshot. Simultaneity is handled entirely by Φ (Phase 3), never here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from simult_chess.core.types import CastleSide, Color, Square, State, Token, Trajectory

_KNIGHT_DELTAS: tuple[tuple[int, int], ...] = (
    (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2),
)
_KING_DELTAS: tuple[tuple[int, int], ...] = (
    (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1),
)
_BISHOP_DIRS: tuple[tuple[int, int], ...] = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_ROOK_DIRS: tuple[tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
_QUEEN_DIRS: tuple[tuple[int, int], ...] = _BISHOP_DIRS + _ROOK_DIRS

_PAWN_START_RANK: dict[Color, int] = {Color.WHITE: 1, Color.BLACK: 6}
_PAWN_DIRECTION: dict[Color, int] = {Color.WHITE: 1, Color.BLACK: -1}

OccupantLookup = Callable[[Square], "Token | None"]


def occupant_lookup(board: Mapping[Token, Square]) -> OccupantLookup:
    """Build :math:`\\beta^{-1}`: an O(1) square-to-occupant lookup for `board`."""
    inverse = {square: token for token, square in board.items()}
    return inverse.get


def _in_bounds(file: int, rank: int) -> bool:
    return 0 <= file <= 7 and 0 <= rank <= 7


def _slide(
    occupant: OccupantLookup,
    mover_color: Color,
    origin: Square,
    directions: tuple[tuple[int, int], ...],
) -> list[Trajectory]:
    trajectories: list[Trajectory] = []
    for step_file, step_rank in directions:
        path: list[Square] = [origin]
        file, rank = origin.file, origin.rank
        while True:
            file, rank = file + step_file, rank + step_rank
            if not _in_bounds(file, rank):
                break
            square = Square(file, rank)
            occ = occupant(square)
            if occ is None:
                path.append(square)
                trajectories.append(Trajectory(path=tuple(path)))
                continue
            if occ.color is not mover_color:
                path.append(square)
                trajectories.append(Trajectory(path=tuple(path)))
            break
    return trajectories


def _knight_trajectories(
    occupant: OccupantLookup, mover_color: Color, origin: Square
) -> list[Trajectory]:
    trajectories: list[Trajectory] = []
    for delta_file, delta_rank in _KNIGHT_DELTAS:
        file, rank = origin.file + delta_file, origin.rank + delta_rank
        if not _in_bounds(file, rank):
            continue
        dest = Square(file, rank)
        occ = occupant(dest)
        if occ is None or occ.color is not mover_color:
            trajectories.append(Trajectory(path=(origin, dest), is_jump=True))
    return trajectories


def _king_step_trajectories(
    occupant: OccupantLookup, mover_color: Color, origin: Square
) -> list[Trajectory]:
    trajectories: list[Trajectory] = []
    for delta_file, delta_rank in _KING_DELTAS:
        file, rank = origin.file + delta_file, origin.rank + delta_rank
        if not _in_bounds(file, rank):
            continue
        dest = Square(file, rank)
        occ = occupant(dest)
        if occ is None or occ.color is not mover_color:
            trajectories.append(Trajectory(path=(origin, dest)))
    return trajectories


def _pawn_trajectories(
    occupant: OccupantLookup, mover_color: Color, origin: Square
) -> list[Trajectory]:
    trajectories: list[Trajectory] = []
    direction = _PAWN_DIRECTION[mover_color]
    one_step_rank = origin.rank + direction
    if not _in_bounds(origin.file, one_step_rank):
        return trajectories
    one_step = Square(origin.file, one_step_rank)
    if occupant(one_step) is None:
        trajectories.append(Trajectory(path=(origin, one_step)))
        if origin.rank == _PAWN_START_RANK[mover_color]:
            two_step = Square(origin.file, origin.rank + 2 * direction)
            if occupant(two_step) is None:
                trajectories.append(Trajectory(path=(origin, one_step, two_step)))
    for delta_file in (-1, 1):
        capture_file = origin.file + delta_file
        if not _in_bounds(capture_file, one_step_rank):
            continue
        capture_square = Square(capture_file, one_step_rank)
        occ = occupant(capture_square)
        if occ is not None and occ.color is not mover_color:
            trajectories.append(Trajectory(path=(origin, capture_square)))
    return trajectories


def pseudo_legal_trajectories(state: State, token: Token) -> list[Trajectory]:
    """All geometrically legal trajectories for `token` on `state.board` (spec §4.2)."""
    origin = state.board[token]
    occupant = occupant_lookup(state.board)
    color = token.color
    if token.typ == "n":
        return _knight_trajectories(occupant, color, origin)
    if token.typ == "k":
        return _king_step_trajectories(occupant, color, origin)
    if token.typ == "p":
        return _pawn_trajectories(occupant, color, origin)
    if token.typ == "b":
        return _slide(occupant, color, origin, _BISHOP_DIRS)
    if token.typ == "r":
        return _slide(occupant, color, origin, _ROOK_DIRS)
    if token.typ == "q":
        return _slide(occupant, color, origin, _QUEEN_DIRS)
    raise ValueError(f"unknown piece type {token.typ!r}")


def _ray_trajectory_to(
    occupant: OccupantLookup,
    origin: Square,
    target: Square,
    directions: tuple[tuple[int, int], ...],
) -> Trajectory | None:
    delta_file = target.file - origin.file
    delta_rank = target.rank - origin.rank
    if delta_file == 0 and delta_rank == 0:
        return None
    if not (delta_file == 0 or delta_rank == 0 or abs(delta_file) == abs(delta_rank)):
        return None
    step_file = (delta_file > 0) - (delta_file < 0)
    step_rank = (delta_rank > 0) - (delta_rank < 0)
    if (step_file, step_rank) not in directions:
        return None
    path = [origin]
    file, rank = origin.file, origin.rank
    while (file, rank) != (target.file, target.rank):
        file, rank = file + step_file, rank + step_rank
        square = Square(file, rank)
        if square != target and occupant(square) is not None:
            return None
        path.append(square)
    return Trajectory(path=tuple(path))


def capturing_pattern_trajectory(
    state: State, defender: Token, target: Square
) -> Trajectory | None:
    """The trajectory `defender` needs to reach `target`, ignoring its occupant.

    Spec §4.3.

    Used for reservation admissibility: the protégé's square holds a friendly
    piece by construction, so the usual "destination holds an enemy" rule is
    deliberately not applied here — only the geometric pattern and a clear
    interior matter.
    """
    origin = state.board[defender]
    occupant = occupant_lookup(state.board)
    if defender.typ == "n":
        delta = (target.file - origin.file, target.rank - origin.rank)
        if delta in _KNIGHT_DELTAS:
            return Trajectory(path=(origin, target), is_jump=True)
        return None
    if defender.typ == "k":
        delta = (target.file - origin.file, target.rank - origin.rank)
        if delta in _KING_DELTAS:
            return Trajectory(path=(origin, target))
        return None
    if defender.typ == "p":
        direction = _PAWN_DIRECTION[defender.color]
        delta_rank = target.rank - origin.rank
        delta_file = abs(target.file - origin.file)
        if delta_rank == direction and delta_file == 1:
            return Trajectory(path=(origin, target))
        return None
    if defender.typ == "b":
        return _ray_trajectory_to(occupant, origin, target, _BISHOP_DIRS)
    if defender.typ == "r":
        return _ray_trajectory_to(occupant, origin, target, _ROOK_DIRS)
    if defender.typ == "q":
        return _ray_trajectory_to(occupant, origin, target, _QUEEN_DIRS)
    raise ValueError(f"unknown piece type {defender.typ!r}")


@dataclass(frozen=True, slots=True)
class CastleMove:
    """A geometrically/history-legal castling move, spec §6.6.

    Parameters
    ----------
    king_token, rook_token : Token
        The synchronized co-movers.
    king_trajectory, rook_trajectory : Trajectory
        Their respective sub-trajectories, each assessable under (V)/(E).
    """

    king_token: Token
    king_trajectory: Trajectory
    rook_token: Token
    rook_trajectory: Trajectory


@dataclass(frozen=True, slots=True)
class _CastleLayout:
    king_from: Square
    king_to: Square
    rook_from: Square
    rook_to: Square
    empty: tuple[Square, ...]
    right: str


_CASTLE_LAYOUT: dict[tuple[Color, CastleSide], _CastleLayout] = {
    (Color.WHITE, "king"): _CastleLayout(
        Square(4, 0), Square(6, 0), Square(7, 0), Square(5, 0),
        (Square(5, 0), Square(6, 0)), "white_kingside",
    ),
    (Color.WHITE, "queen"): _CastleLayout(
        Square(4, 0), Square(2, 0), Square(0, 0), Square(3, 0),
        (Square(1, 0), Square(2, 0), Square(3, 0)), "white_queenside",
    ),
    (Color.BLACK, "king"): _CastleLayout(
        Square(4, 7), Square(6, 7), Square(7, 7), Square(5, 7),
        (Square(5, 7), Square(6, 7)), "black_kingside",
    ),
    (Color.BLACK, "queen"): _CastleLayout(
        Square(4, 7), Square(2, 7), Square(0, 7), Square(3, 7),
        (Square(1, 7), Square(2, 7), Square(3, 7)), "black_queenside",
    ),
}


def castle_move(state: State, color: Color, side: CastleSide) -> CastleMove | None:
    """Return the `CastleMove` for `color`/`side` if legal on `state`, else `None`.

    Legality reduces to the geometric/history conditions of spec §6.6:
    the relevant castling right holds, king and rook are on their home
    squares, and the squares between them are empty on β. "Through check"
    does not apply (no check exists, spec §10).
    """
    layout = _CASTLE_LAYOUT[(color, side)]
    if not getattr(state.bookkeeping.castling_rights, layout.right):
        return None
    occupant = occupant_lookup(state.board)
    king = occupant(layout.king_from)
    rook = occupant(layout.rook_from)
    if king is None or king.typ != "k" or king.color is not color:
        return None
    if rook is None or rook.typ != "r" or rook.color is not color:
        return None
    if any(occupant(square) is not None for square in layout.empty):
        return None
    return CastleMove(
        king_token=king,
        king_trajectory=Trajectory(path=(layout.king_from, layout.king_to)),
        rook_token=rook,
        rook_trajectory=Trajectory(path=(layout.rook_from, layout.rook_to)),
    )
