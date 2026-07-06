"""Declaration-legality predicate L(s, π_ω), spec §4.4; INVARIANTS.md L1-L6.

Every check here runs against the declaration-time board only — no
look-ahead, matching spec §4.2's "simultaneity is handled entirely by Φ,
never by look-ahead in declaration."
"""

from __future__ import annotations

from simult_chess.core import geometry
from simult_chess.core.collision import conflicts
from simult_chess.core.types import (
    Action,
    Cancel,
    Castle,
    CastleSide,
    Color,
    Move,
    Program,
    Reserve,
    State,
    Token,
    Trajectory,
)
from simult_chess.core.violation import Violation
from simult_chess.rules.ruleset import RuleSet


def king_token(state: State, color: Color) -> Token | None:
    """The live king token of `color`, if any."""
    for token in state.board:
        if token.color is color and token.typ == "k":
            return token
    return None


def actor_of(action: Action, color: Color, state: State) -> Token | None:
    """The action's actor (INVARIANTS.md §1): mover for Move, king for Castle,
    defender for Reserve, `None` (no board actor) for Cancel."""
    if isinstance(action, Move):
        return action.token
    if isinstance(action, Reserve):
        return action.defender
    if isinstance(action, Castle):
        return king_token(state, color)
    if isinstance(action, Cancel):
        return None
    raise TypeError(f"unknown action {action!r}")


def has_any_legal_displacement(state: State, color: Color) -> bool:
    """Whether `color` has any legal Move or Castle at all (L2 exception)."""
    for token in state.board:
        if token.color is color and geometry.pseudo_legal_trajectories(state, token):
            return True
    sides: tuple[CastleSide, CastleSide] = ("king", "queen")
    for side in sides:
        if geometry.castle_move(state, color, side) is not None:
            return True
    return False


def check_l1_budget(program: Program, ruleset: RuleSet) -> list[Violation]:
    """L1 — budget: :math:`1\\le|\\pi_\\omega|\\le N`."""
    if 1 <= len(program) <= ruleset.n_actions:
        return []
    detail = f"program has {len(program)} actions, budget is [1,{ruleset.n_actions}]"
    return [Violation("L1", detail)]


def check_l2_mandatory_displacement(
    state: State, program: Program, color: Color
) -> list[Violation]:
    """L2 — at least one Move/Castle, unless no legal displacement exists at all."""
    has_displacement = any(isinstance(a, Move | Castle) for a in program)
    if has_displacement or not has_any_legal_displacement(state, color):
        return []
    detail = "no Move/Castle declared though a legal displacement exists"
    return [Violation("L2", detail)]


def check_l3_distinct_actors(
    state: State, program: Program, color: Color
) -> list[Violation]:
    """L3 — each token is the actor of at most one action."""
    actor_ids = [
        actor.id
        for action in program
        if (actor := actor_of(action, color, state)) is not None
    ]
    if len(set(actor_ids)) == len(actor_ids):
        return []
    return [Violation("L3", f"token(s) act more than once: {actor_ids}")]


def check_l4_cooldown_respected(
    state: State, program: Program, color: Color
) -> list[Violation]:
    """L4 — no actor lies in the cooldown set C."""
    violations: list[Violation] = []
    for action in program:
        actor = actor_of(action, color, state)
        if actor is not None and actor in state.cooldown:
            violations.append(Violation("L4", f"actor {actor.id} is cooled"))
    return violations


def _own_move_trajectories(
    state: State, program: Program, color: Color
) -> list[Trajectory]:
    trajectories: list[Trajectory] = []
    for action in program:
        if isinstance(action, Move):
            trajectories.append(action.trajectory)
        elif isinstance(action, Castle):
            castle = geometry.castle_move(state, color, action.side)
            if castle is not None:
                trajectories.append(castle.king_trajectory)
                trajectories.append(castle.rook_trajectory)
    return trajectories


def _own_moving_token_ids(state: State, program: Program, color: Color) -> set[int]:
    moving_ids: set[int] = set()
    for action in program:
        if isinstance(action, Move):
            moving_ids.add(action.token.id)
        elif isinstance(action, Castle):
            castle = geometry.castle_move(state, color, action.side)
            if castle is not None:
                moving_ids.add(castle.king_token.id)
                moving_ids.add(castle.rook_token.id)
    return moving_ids


def check_l5_own_consistency(
    state: State, program: Program, color: Color
) -> list[Violation]:
    """L5 — own executing moves are non-conflicting; none targets a friendly piece."""
    violations: list[Violation] = []
    trajectories = _own_move_trajectories(state, program, color)
    for i in range(len(trajectories)):
        for j in range(i + 1, len(trajectories)):
            if conflicts(trajectories[i], trajectories[j]):
                violations.append(Violation("L5", f"own moves {i} and {j} conflict"))

    moving_ids = _own_moving_token_ids(state, program, color)
    occupant = geometry.occupant_lookup(state.board)
    for trajectory in trajectories:
        occ = occupant(trajectory.destination)
        if occ is not None and occ.color is color and occ.id not in moving_ids:
            dest = trajectory.destination
            violations.append(Violation("L5", f"own move lands on own piece at {dest}"))
    return violations


def check_l6_geometric_legality(
    state: State, program: Program, color: Color
) -> list[Violation]:
    """L6 — each Move/Reserve/Castle/Cancel satisfies its own geometric predicate."""
    violations: list[Violation] = []
    for index, action in enumerate(program):
        if isinstance(action, Move):
            if action.trajectory not in geometry.pseudo_legal_trajectories(
                state, action.token
            ):
                detail = f"action {index}: illegal move for token {action.token.id}"
                violations.append(Violation("L6", detail))
        elif isinstance(action, Reserve):
            if action.protege not in state.board or action.defender not in state.board:
                detail = f"action {index}: reservation references a non-live token"
                violations.append(Violation("L6", detail))
            else:
                target = state.board[action.protege]
                pattern = geometry.capturing_pattern_trajectory(
                    state, action.defender, target
                )
                if pattern is None:
                    detail = (
                        f"action {index}: reservation "
                        f"({action.defender.id},{action.protege.id}) not admissible"
                    )
                    violations.append(Violation("L6", detail))
        elif isinstance(action, Castle):
            if geometry.castle_move(state, color, action.side) is None:
                detail = f"action {index}: castle {action.side} illegal"
                violations.append(Violation("L6", detail))
        elif isinstance(action, Cancel):
            if action.reservation not in state.reservations(color):
                detail = (
                    f"action {index}: cancel names a reservation "
                    f"not in R_{color.value}"
                )
                violations.append(Violation("L6", detail))
        else:
            raise TypeError(f"unknown action {action!r}")
    return violations


def check_legal_program(
    state: State, program: Program, color: Color, ruleset: RuleSet
) -> list[Violation]:
    """Run L1-L6 in sequence, returning every violation found (empty if legal)."""
    return [
        *check_l1_budget(program, ruleset),
        *check_l2_mandatory_displacement(state, program, color),
        *check_l3_distinct_actors(state, program, color),
        *check_l4_cooldown_respected(state, program, color),
        *check_l5_own_consistency(state, program, color),
        *check_l6_geometric_legality(state, program, color),
    ]


def is_legal_program(
    state: State, program: Program, color: Color, ruleset: RuleSet
) -> bool:
    """The boolean predicate :math:`L(s,\\pi_\\omega)` (spec §4.4)."""
    return check_legal_program(state, program, color, ruleset) == []
