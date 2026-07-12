"""Textual action DSL: parse/format a program for the TUI (dev brief §3.5, Phase 7).

Grammar (one action per ``;``-separated clause; up to `RuleSet.n_actions` clauses)::

    move    := [PIECE] SQUARE SQUARE          # full coordinate form, e.g. "Ng1f3"
             | [PIECE] SQUARE                 # short form, e.g. "Nf3", "e4"
               ["=" PROMO]                    # promotion suffix, e.g. "e8=Q"
    castle  := "O-O" | "O-O-O"                # (also accepts "0-0"/"0-0-0")
    reserve := SQUARE "def" [PIECE] SQUARE    # protege-square "def" defender
    cancel  := "cancel" INDEX                 # index into state.reservations(color)

    PIECE   := "N" | "B" | "R" | "Q" | "K"    # uppercase; pawn has no letter
    PROMO   := "N" | "B" | "R" | "Q"
    SQUARE  := [a-h][1-8]                     # lowercase file, e.g. "e4"

A lone "x"/"X" (capture marker, e.g. "Nxf3") is accepted and ignored -- this
variant's simultaneity means declaration-time text can never promise a
capture actually lands (spec §5's fizzle/annihilation stages decide that),
so "x" is cosmetic only, never validated against the board.

The short move form's origin is inferred by scanning `color`'s own pieces
of the matching type for exactly one whose declaration-time trajectories
reach the destination (spec §4.2); an ambiguous or empty match raises
`NotationError` asking for the full coordinate form. Partial (file-only)
disambiguation (e.g. "exd5") is not supported -- write "e4d5" instead.

A `reserve` protege-square is resolved against *this program's* declared
move destinations first, then against the current board -- the "aggressive
dual" pattern (spec §4.3) names the square a piece is moving *into* this
same phase.
"""

from __future__ import annotations

import re

from simult_chess.core import geometry
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
)

_PIECE_TO_TYP: dict[str, PieceType] = {"N": "n", "B": "b", "R": "r", "Q": "q", "K": "k"}
_TYP_TO_PIECE: dict[PieceType, str] = {v: k for k, v in _PIECE_TO_TYP.items()}
_FILES = "abcdefgh"
_DEF_RE = re.compile(r"\bdef\b", re.IGNORECASE)


class NotationError(ValueError):
    """The DSL text does not parse, or names a piece/square that isn't there."""


def format_square(square: Square) -> str:
    """Render `square` in algebraic form, e.g. ``Square(4, 3)`` -> ``"e4"``."""
    return str(square)


def str_to_square(text: str) -> Square:
    """Parse an algebraic square like ``"e4"``; raises `NotationError` if malformed."""
    if len(text) != 2 or text[0] not in _FILES or text[1] not in "12345678":
        raise NotationError(f"not a square: {text!r}")
    return Square(file=_FILES.index(text[0]), rank=int(text[1]) - 1)


def _occupant_at(state: State, square: Square) -> Token | None:
    return geometry.occupant_lookup(state.board)(square)


def _classify(text: str) -> str:
    normalized = text.strip().upper().replace("0", "O")
    if normalized in ("O-O", "O-O-O"):
        return "castle"
    if _DEF_RE.search(text):
        return "reserve"
    if text.strip().lower().startswith("cancel"):
        return "cancel"
    return "move"


def _parse_move(text: str, state: State, color: Color) -> Move:
    body = text.strip()
    promotion: PieceType | None = None
    if "=" in body:
        body, promo_letter = body.rsplit("=", 1)
        promo_letter = promo_letter.strip().upper()
        if promo_letter not in ("N", "B", "R", "Q"):
            raise NotationError(f"bad promotion suffix: ={promo_letter!r}")
        promotion = _PIECE_TO_TYP[promo_letter]
    body = body.replace("x", "").replace("X", "").strip()

    piece_letter: str | None = None
    if body and body[0] in _PIECE_TO_TYP:
        piece_letter = body[0]
        body = body[1:]

    if len(body) == 4:
        origin = str_to_square(body[:2])
        dest = str_to_square(body[2:])
        token = _occupant_at(state, origin)
        if token is None or token.color is not color:
            raise NotationError(f"no {color.value} piece on {body[:2]}")
    elif len(body) == 2:
        dest = str_to_square(body)
        expected_typ = _PIECE_TO_TYP.get(piece_letter, "p") if piece_letter else "p"
        candidates = [
            t
            for t in state.board
            if t.color is color
            and t.typ == expected_typ
            and any(
                traj.destination == dest
                for traj in geometry.pseudo_legal_trajectories(state, t)
            )
        ]
        if not candidates:
            raise NotationError(f"no {expected_typ} can reach {format_square(dest)}")
        if len(candidates) > 1:
            raise NotationError(
                f"ambiguous: multiple pieces can reach {format_square(dest)}; "
                f"use the full origin+dest form"
            )
        token = candidates[0]
    else:
        raise NotationError(f"malformed move: {text!r}")

    if piece_letter is not None and _PIECE_TO_TYP[piece_letter] != token.typ:
        origin_sq = format_square(state.board[token])
        raise NotationError(f"{origin_sq} is not a {piece_letter}")

    trajectories = [
        t
        for t in geometry.pseudo_legal_trajectories(state, token)
        if t.destination == dest
    ]
    if not trajectories:
        origin_sq = format_square(state.board[token])
        dest_sq = format_square(dest)
        raise NotationError(f"{token.typ} on {origin_sq} cannot reach {dest_sq}")
    return Move(token=token, trajectory=trajectories[0], promotion=promotion)


def _parse_castle(text: str) -> Castle:
    normalized = text.strip().upper().replace("0", "O")
    side: CastleSide = "king" if normalized == "O-O" else "queen"
    return Castle(side=side)


def _parse_reserve(
    text: str, state: State, color: Color, future_destinations: dict[Square, Token]
) -> Reserve:
    parts = _DEF_RE.split(text, maxsplit=1)
    if len(parts) != 2:
        detail = f"malformed reserve (expected '<square> def <piece>'): {text!r}"
        raise NotationError(detail)
    protege_text, defender_text = (p.strip() for p in parts)

    protege_square = str_to_square(protege_text)
    protege = future_destinations.get(protege_square) or _occupant_at(
        state, protege_square
    )
    if protege is None or protege.color is not color:
        raise NotationError(f"no {color.value} piece on/moving to {protege_text}")

    defender_piece_letter: str | None = None
    if defender_text and defender_text[0] in _PIECE_TO_TYP:
        defender_piece_letter = defender_text[0]
        defender_text = defender_text[1:]
    defender_square = str_to_square(defender_text)
    defender = _occupant_at(state, defender_square)
    if defender is None or defender.color is not color:
        raise NotationError(f"no {color.value} piece on {defender_text}")
    if (
        defender_piece_letter is not None
        and _PIECE_TO_TYP[defender_piece_letter] != defender.typ
    ):
        raise NotationError(f"{defender_text} is not a {defender_piece_letter}")

    return Reserve(defender=defender, protege=protege)


def _parse_cancel(text: str, state: State, color: Color) -> Cancel:
    rest = text.strip()[len("cancel") :].strip()
    try:
        index = int(rest)
    except ValueError as exc:
        raise NotationError(f"malformed cancel (expected an index): {text!r}") from exc
    reservations = state.reservations(color)
    if not 0 <= index < len(reservations):
        detail = f"reservation index {index} out of range (have {len(reservations)})"
        raise NotationError(detail)
    return Cancel(reservation=reservations[index])


def parse_program(text: str, state: State, color: Color) -> Program:
    """Parse a ``;``-separated program string into a `Program` tuple.

    Raises `NotationError` on any malformed or unresolvable clause. Does
    *not* check `L(s,\\pi)` legality -- that is `legality.check_legal_program`'s
    job, run separately so its per-clause `Violation`s can be reported.
    """
    raw_parts = [p.strip() for p in text.split(";") if p.strip()]
    kinds = [(_classify(p), p) for p in raw_parts]

    move_castle: dict[int, Action] = {}
    future_destinations: dict[Square, Token] = {}
    for index, (kind, raw) in enumerate(kinds):
        if kind == "move":
            move_action = _parse_move(raw, state, color)
            future_destinations[move_action.trajectory.destination] = move_action.token
            move_castle[index] = move_action
        elif kind == "castle":
            castle_action = _parse_castle(raw)
            move_castle[index] = castle_action
            castle_move = geometry.castle_move(state, color, castle_action.side)
            if castle_move is not None:
                future_destinations[castle_move.king_trajectory.destination] = (
                    castle_move.king_token
                )
                future_destinations[castle_move.rook_trajectory.destination] = (
                    castle_move.rook_token
                )

    actions: list[Action] = []
    for index, (kind, raw) in enumerate(kinds):
        if kind in ("move", "castle"):
            actions.append(move_castle[index])
        elif kind == "reserve":
            actions.append(_parse_reserve(raw, state, color, future_destinations))
        elif kind == "cancel":
            actions.append(_parse_cancel(raw, state, color))
        else:
            raise NotationError(f"unrecognized action: {raw!r}")
    return tuple(actions)


def format_action(action: Action, state: State, color: Color) -> str:
    """Render a single `Action` back to DSL text (for echoing/logging)."""
    if isinstance(action, Move):
        letter = "" if action.token.typ == "p" else _TYP_TO_PIECE[action.token.typ]
        origin = format_square(action.trajectory.origin)
        dest = format_square(action.trajectory.destination)
        text = f"{letter}{origin}{dest}"
        if action.promotion is not None:
            text += f"={_TYP_TO_PIECE[action.promotion]}"
        return text
    if isinstance(action, Castle):
        return "O-O" if action.side == "king" else "O-O-O"
    if isinstance(action, Reserve):
        defender_letter = _TYP_TO_PIECE.get(action.defender.typ, "")
        protege_square = format_square(state.board[action.protege])
        defender_square = format_square(state.board[action.defender])
        return f"{protege_square} def {defender_letter}{defender_square}"
    if isinstance(action, Cancel):
        index = state.reservations(color).index(action.reservation)
        return f"cancel {index}"
    raise TypeError(f"unknown action {action!r}")


def format_program(program: Program, state: State, color: Color) -> str:
    """Render a whole `Program` back to DSL text, ``;``-joined."""
    return "; ".join(format_action(action, state, color) for action in program)
