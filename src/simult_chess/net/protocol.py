"""JSON (de)serialization of a `Program` for the wire (spec §11.5, Phase 8).

Actions reference live `Token`s by identity; both peers derive `State` the
same way (identical previous state + identical revealed programs through
the same pure `phi`), so a token id plus the *receiving* peer's own
`state.board` is enough to reconstruct the exact `Token` instance -- no
need to serialize color/type redundantly. A `Cancel`'s reservation is
likewise reduced to its index in `state.reservations(color)`, which is
guaranteed identical on both peers for the same reason.
"""

from __future__ import annotations

from typing import Any

from simult_chess.core.types import (
    Action,
    Cancel,
    Castle,
    CastleSide,
    Color,
    Move,
    PieceType,
    Program,
    Reserve,
    Square,
    State,
    Token,
    Trajectory,
)


class ProtocolError(Exception):
    """A peer sent a message that violates the commit-reveal handshake."""


def _token_by_id(state: State, token_id: int) -> Token:
    for token in state.board:
        if token.id == token_id:
            return token
    raise ProtocolError(f"no live token with id {token_id}")


def serialize_action(action: Action, state: State, color: Color) -> dict[str, Any]:
    """Render one `Action` to JSON-safe primitives, relative to `state`."""
    if isinstance(action, Move):
        return {
            "kind": "move",
            "token_id": action.token.id,
            "path": [[sq.file, sq.rank] for sq in action.trajectory.path],
            "is_jump": action.trajectory.is_jump,
            "promotion": action.promotion,
        }
    if isinstance(action, Reserve):
        return {
            "kind": "reserve",
            "defender_id": action.defender.id,
            "protege_id": action.protege.id,
        }
    if isinstance(action, Castle):
        return {"kind": "castle", "side": action.side}
    if isinstance(action, Cancel):
        index = state.reservations(color).index(action.reservation)
        return {"kind": "cancel", "index": index}
    raise TypeError(f"unknown action {action!r}")


def deserialize_action(data: dict[str, Any], state: State, color: Color) -> Action:
    """Reconstruct one `Action` from wire data, resolving tokens against `state`."""
    kind = data["kind"]
    if kind == "move":
        token = _token_by_id(state, data["token_id"])
        path = tuple(Square(file=f, rank=r) for f, r in data["path"])
        promotion: PieceType | None = data["promotion"]
        trajectory = Trajectory(path=path, is_jump=data["is_jump"])
        return Move(token=token, trajectory=trajectory, promotion=promotion)
    if kind == "reserve":
        defender = _token_by_id(state, data["defender_id"])
        protege = _token_by_id(state, data["protege_id"])
        return Reserve(defender=defender, protege=protege)
    if kind == "castle":
        side: CastleSide = data["side"]
        return Castle(side=side)
    if kind == "cancel":
        reservations = state.reservations(color)
        index = data["index"]
        if not 0 <= index < len(reservations):
            raise ProtocolError(f"cancel index {index} out of range")
        return Cancel(reservation=reservations[index])
    raise ProtocolError(f"unknown action kind {kind!r}")


def serialize_program(
    program: Program, state: State, color: Color
) -> list[dict[str, Any]]:
    """Render a whole `Program` to a JSON-safe list, relative to `state`."""
    return [serialize_action(action, state, color) for action in program]


def deserialize_program(
    data: list[dict[str, Any]], state: State, color: Color
) -> Program:
    """Reconstruct a whole `Program` from wire data, relative to `state`."""
    return tuple(deserialize_action(item, state, color) for item in data)
