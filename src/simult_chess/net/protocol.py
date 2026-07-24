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

from typing import Any, Final

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

#: Wire-protocol version, bumped on any breaking change to the message schema
#: or ordering contract. Exchanged in the handshake (Phase 15a); a mismatch
#: aborts before phase 0. Independent of ``SPEC_VERSION`` (the rules) and of a
#: ``RuleSet`` fingerprint (the rule *values*).
PROTOCOL_VERSION: Final = 1

#: Version of the rules specification both peers must share. The engine
#: implements ``docs/simultaneous_chess_spec_v1.md`` v1.1; two peers on
#: different spec versions could compute Φ differently, so the handshake
#: rejects a mismatch by name.
SPEC_VERSION: Final = "1.1"

#: Message ``type`` values. The per-phase *action slot* is exactly one of
#: ``commit`` / ``resign`` / ``abort`` / ``accept_draw`` from each peer; a
#: ``commit`` may carry ``offer_draw: true`` (a standing offer the peer may
#: accept on a later slot, or decline by simply playing on). ``ping``/``pong``
#: are the keepalive; ``reveal``/``ack`` follow a mutual commit; ``handshake``
#: opens the match and ``rematch`` may follow its end. Documented in full by
#: ``docs/PROTOCOL.md`` (Phase 15c).
MSG_HANDSHAKE: Final = "handshake"
MSG_COMMIT: Final = "commit"
MSG_REVEAL: Final = "reveal"
MSG_ACK: Final = "ack"
MSG_PING: Final = "ping"
MSG_PONG: Final = "pong"
MSG_RESIGN: Final = "resign"
MSG_ABORT: Final = "abort"
MSG_OFFER_DRAW: Final = "offer_draw"
MSG_ACCEPT_DRAW: Final = "accept_draw"
MSG_DECLINE_DRAW: Final = "decline_draw"
MSG_REMATCH: Final = "rematch"

#: The action-slot message types (one per peer per phase). ``offer_draw`` is
#: not here: it rides on a ``commit`` as a flag rather than replacing it, so a
#: draw offer never costs a phase.
ACTION_SLOT_TYPES: Final = frozenset(
    {MSG_COMMIT, MSG_RESIGN, MSG_ABORT, MSG_ACCEPT_DRAW}
)


class ProtocolError(Exception):
    """A peer sent a message that violates the commit-reveal handshake."""


class TransportTimeout(ProtocolError):
    """A timed ``recv`` elapsed with no message.

    A subclass of :class:`ProtocolError` so existing ``except ProtocolError``
    sites still catch a dead peer, but distinct so the keepalive loop can tell
    "quiet for one interval, ping again" from "connection lost" — the two must
    not be conflated once decision time is unbounded (Phase 15a, B1/B5).
    """


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
