"""A material-heuristic agent over single-action programs (dev brief Phase 6)."""

from __future__ import annotations

import random

from simult_chess.agents.candidates import move_candidates
from simult_chess.core import geometry
from simult_chess.core.types import Action, Color, Move, PieceType, Program, State
from simult_chess.rules.ruleset import RuleSet

_PIECE_VALUES: dict[PieceType, int] = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}


def _capture_value(state: State, color: Color, action: Action) -> int:
    if not isinstance(action, Move):
        return -1
    occupant = geometry.occupant_lookup(state.board)
    target = occupant(action.trajectory.destination)
    if target is None or target.color is color:
        return -1
    return _PIECE_VALUES[target.typ]


def greedy_program(
    state: State, color: Color, ruleset: RuleSet, rng: random.Random
) -> Program:
    """Take the highest-value capture available, else a random legal move.

    Single-action programs only (dev brief: "material heuristic over
    single-action programs") — never declares a reservation.
    """
    candidates = move_candidates(state, color, rng)
    if not candidates:
        return ()

    best_value = max(_capture_value(state, color, action) for action in candidates)
    if best_value > 0:
        best = [a for a in candidates if _capture_value(state, color, a) == best_value]
        return (rng.choice(best),)
    return (rng.choice(candidates),)
