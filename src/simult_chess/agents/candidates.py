"""Shared candidate-action enumeration for agents (dev brief Phase 6)."""

from __future__ import annotations

import random

from simult_chess.core import geometry
from simult_chess.core.types import (
    Action,
    Cancel,
    Castle,
    CastleSide,
    Color,
    Move,
    PieceType,
    Reserve,
    State,
)

_PROMOTABLE: tuple[PieceType, ...] = ("n", "b", "r", "q")
_LAST_RANK = {Color.WHITE: 7, Color.BLACK: 0}
_CASTLE_SIDES: tuple[CastleSide, CastleSide] = ("king", "queen")


def move_candidates(state: State, color: Color, rng: random.Random) -> list[Action]:
    """Every individually-legal Move/Castle action for `color` (spec §4.2, §6.6).

    A pawn move reaching the last rank gets a random promotion choice
    (`rng`-driven, so callers stay pure functions of their own seed).
    """
    candidates: list[Action] = []
    for token in state.board:
        if token.color is not color or token in state.cooldown:
            continue
        for trajectory in geometry.pseudo_legal_trajectories(state, token):
            promotion = None
            if token.typ == "p" and trajectory.destination.rank == _LAST_RANK[color]:
                promotion = rng.choice(_PROMOTABLE)
            candidates.append(
                Move(token=token, trajectory=trajectory, promotion=promotion)
            )
    for side in _CASTLE_SIDES:
        if geometry.castle_move(state, color, side) is not None:
            candidates.append(Castle(side=side))
    return candidates


def reserve_candidates(state: State, color: Color) -> list[Action]:
    """Every individually-admissible Reserve action for `color` (spec §4.3)."""
    candidates: list[Action] = []
    for defender in state.board:
        if defender.color is not color or defender in state.cooldown:
            continue
        for protege in state.board:
            if protege.color is not color or protege is defender:
                continue
            target = state.board[protege]
            pattern = geometry.capturing_pattern_trajectory(state, defender, target)
            if pattern is not None:
                candidates.append(Reserve(defender=defender, protege=protege))
    return candidates


def cancel_candidates(state: State, color: Color) -> list[Action]:
    """Every Cancel action for `color` (spec §4.1, §9): one per standing
    reservation in R_color.

    Before Phase 13b (ruling D3, docs/LEARNING_DESIGN.md) the shared
    candidate generation never emitted Cancel, so no stdlib agent could
    construct one -- the root of the Phase 11b campaign's *structural*
    cancellation rate of 0.000. A Cancel-only program is L2-illegal while a
    legal displacement exists, so a cancel is useful only paired with a
    Move/Castle; callers that assemble multi-action programs (e.g.
    `random_legal_program`) legalize the pairing through `L(s,π)`.
    """
    return [Cancel(reservation=r) for r in state.reservations(color)]
