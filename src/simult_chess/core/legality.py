"""Declaration-legality predicate L(s, π_ω), spec §4.4; INVARIANTS.md L1-L6.

Every check here runs against the declaration-time board only — no
look-ahead, matching spec §4.2's "simultaneity is handled entirely by Φ,
never by look-ahead in declaration."

L3 (`check_l3_distinct_actors`) implements the v1.1 ruling A3: a `Castle`
action's actor set is `{king, flank_rook}` (spec §4.1/§6.6), not the king
alone, so it now catches a program that castles and also separately
declares an action for that same rook.
"""

from __future__ import annotations

from simult_chess.core import geometry
from simult_chess.core.collision import conflicts
from simult_chess.core.geometry import OccupantLookup
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

_LAST_RANK = {Color.WHITE: 7, Color.BLACK: 0}
_PROMOTABLE_TYPES = frozenset({"n", "b", "r", "q"})


def king_token(state: State, color: Color) -> Token | None:
    """The live king token of `color`, if any."""
    for token in state.board:
        if token.color is color and token.typ == "k":
            return token
    return None


def actors_of(action: Action, color: Color, state: State) -> tuple[Token, ...]:
    """The action's actor set (INVARIANTS.md §1, spec §4.1/§4.4.3/§6.6): the mover
    for Move; defender for Reserve; ``{king, flank_rook}`` for Castle (both
    synchronized sub-movers are actors, per the v1.1 ruling A3 — not the king
    alone); empty (no board actor) for Cancel."""
    if isinstance(action, Move):
        return (action.token,)
    if isinstance(action, Reserve):
        return (action.defender,)
    if isinstance(action, Castle):
        castle = geometry.castle_move(state, color, action.side)
        if castle is None:
            king = king_token(state, color)
            return (king,) if king is not None else ()
        return (castle.king_token, castle.rook_token)
    if isinstance(action, Cancel):
        return ()
    raise TypeError(f"unknown action {action!r}")


def has_any_legal_displacement(state: State, color: Color) -> bool:
    """Whether `color` has any legal Move or Castle at all (L2 exception).

    Cooldown-aware (ruling 17b, 2026-07-28 fix): a cooled token contributes
    no legal Move regardless of its raw geometric mobility, since L4 forbids
    declaring it as an actor. Before king cooldown existed, this was
    geometry-only and got away with it — pawns and kings, the two types most
    often the *sole* geometrically-mobile piece, were both cooldown-exempt,
    so "geometrically possible" and "actually declarable" always agreed for
    them. Once a king can be cooled (ruling 17b), that agreement breaks: a
    self-play sweep hit a state where the king was the only piece with a
    pseudo-legal trajectory, cooled, and every other piece geometrically
    stuck — this function said "a displacement exists" (true, geometrically),
    L2 then required a Move/Castle in the program, and L4 rejected every
    program containing one, leaving *no* legal program at all. Filtering
    cooldown here closes that gap. See :func:`has_any_legal_program` for the
    closed existence question this function deliberately does not answer
    (Reserve/Cancel are out of scope for L2).
    """
    for token in state.board:
        if (
            token.color is color
            and token not in state.cooldown
            and geometry.pseudo_legal_trajectories(state, token)
        ):
            return True
    sides: tuple[CastleSide, CastleSide] = ("king", "queen")
    for side in sides:
        if geometry.castle_move(state, color, side) is not None:
            return True
    return False


def has_any_legal_program(state: State, color: Color, ruleset: RuleSet) -> bool:
    """Whether `color` has at least one legal 1..N-action program this phase.

    Cooldown- and ruleset-aware, and — unlike
    :func:`has_any_legal_displacement`, which only ever answers the narrower
    Move/Castle-only question L2 needs — also covers Reserve and Cancel: this
    is the *closed* question compute_cooldown's king safety valve (ruling
    17b, 2026-07-28) needs — could this player construct any legal program at
    all, given the proposed cooldown set? A castling king/rook is never
    cooled if
    `geometry.castle_move` finds it (rights intact ⇒ neither has ever
    displaced ⇒ neither can be in cooldown), so no extra check is needed
    there. A `Cancel`'s actor set is empty (`actors_of`), so it is never
    blocked by cooldown — only by `ruleset.cancellation_enabled` (L6).
    """
    for token in state.board:
        if (
            token.color is color
            and token not in state.cooldown
            and geometry.pseudo_legal_trajectories(state, token)
        ):
            return True
    sides: tuple[CastleSide, CastleSide] = ("king", "queen")
    for side in sides:
        if geometry.castle_move(state, color, side) is not None:
            return True
    if ruleset.cancellation_enabled and state.reservations(color):
        return True
    occupant = geometry.occupant_lookup(state.board)
    for defender in state.board:
        if defender.color is not color or defender in state.cooldown:
            continue
        origin = state.board[defender]
        for protege in state.board:
            if protege.color is not color or protege is defender:
                continue
            target = state.board[protege]
            pattern = geometry.capturing_pattern_trajectory_at(
                defender.typ, defender.color, origin, target, occupant
            )
            if pattern is not None:
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
    """L3 — each token is the actor of at most one action. A `Castle`
    contributes two actors, `{king, flank_rook}` (spec v1.1, A3), so a program
    that castles and also separately declares an action for that same rook
    fails here."""
    actor_ids = [
        actor.id
        for action in program
        for actor in actors_of(action, color, state)
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
        for actor in actors_of(action, color, state):
            if actor in state.cooldown:
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
    state: State,
    program: Program,
    color: Color,
    *,
    occupant: OccupantLookup | None = None,
) -> list[Violation]:
    """L6 — each Move/Reserve/Castle/Cancel satisfies its own geometric predicate.

    v3 18a.3 (audit B9): `occupant`, if given, is reused for every `Reserve`
    action's admissibility check instead of each one rebuilding its own
    lookup from `state.board` (`geometry.capturing_pattern_trajectory`'s own
    convenience behaviour) -- the same fix already applied to
    `agents.candidates.reserve_candidates` (e331623). Optional and `None`
    by default so every existing caller's behaviour is unchanged; hot paths
    that call this many times against the same `state` (the legality masks
    in `learn.action_grid`) build one `occupant` up front and pass it
    through `check_partial_program`/`check_legal_program`/`is_legal_program`.
    """
    resolved_occupant = occupant if occupant is not None else geometry.occupant_lookup(
        state.board
    )
    violations: list[Violation] = []
    for index, action in enumerate(program):
        if isinstance(action, Move):
            if action.trajectory not in geometry.pseudo_legal_trajectories(
                state, action.token
            ):
                detail = f"action {index}: illegal move for token {action.token.id}"
                violations.append(Violation("L6", detail))
            reaches_last_rank = (
                action.token.typ == "p"
                and action.trajectory.destination.rank == _LAST_RANK[action.token.color]
            )
            if reaches_last_rank and action.promotion not in _PROMOTABLE_TYPES:
                detail = f"action {index}: pawn reaching last rank must promote"
                violations.append(Violation("L6", detail))
            elif not reaches_last_rank and action.promotion is not None:
                detail = f"action {index}: promotion declared without reaching last"
                violations.append(Violation("L6", detail))
        elif isinstance(action, Reserve):
            if action.protege not in state.board or action.defender not in state.board:
                detail = f"action {index}: reservation references a non-live token"
                violations.append(Violation("L6", detail))
            else:
                # The "aggressive dual" pattern (spec §4.3): if the protégé
                # is itself moving this program, admissibility is judged
                # against its declared *destination*, not its current square.
                protege_move = next(
                    (
                        a
                        for a in program
                        if isinstance(a, Move) and a.token == action.protege
                    ),
                    None,
                )
                target = (
                    protege_move.trajectory.destination
                    if protege_move is not None
                    else state.board[action.protege]
                )
                pattern = geometry.capturing_pattern_trajectory_at(
                    action.defender.typ,
                    action.defender.color,
                    state.board[action.defender],
                    target,
                    resolved_occupant,
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


def check_partial_program(
    state: State,
    program: Program,
    color: Color,
    ruleset: RuleSet,
    *,
    occupant: OccupantLookup | None = None,
) -> list[Violation]:
    """Run only the clauses valid for a *program under construction* (Phase 16.1).

    L3–L6 (distinct actors, cooldown, own-consistency, geometric legality) all
    hold clause-by-clause as a player builds a program, so they can validate
    each partial edit live in a UI. The two *whole-program* clauses are
    skipped: L1 (the 1..N budget) rejects the empty in-progress program, and L2
    (mandatory displacement) can only be judged once the program is complete.
    Full `L` still runs on submit — this shares the exact clause functions with
    :func:`check_legal_program`, never a copy.

    ``ruleset`` is accepted for signature parity with :func:`check_legal_program`
    (L3–L6 do not currently read it). ``occupant`` (v3 18a.3, audit B9) is
    forwarded to :func:`check_l6_geometric_legality` -- see its docstring.
    """
    del ruleset
    return [
        *check_l3_distinct_actors(state, program, color),
        *check_l4_cooldown_respected(state, program, color),
        *check_l5_own_consistency(state, program, color),
        *check_l6_geometric_legality(state, program, color, occupant=occupant),
    ]


def check_legal_program(
    state: State,
    program: Program,
    color: Color,
    ruleset: RuleSet,
    *,
    occupant: OccupantLookup | None = None,
) -> list[Violation]:
    """Run L1-L6 in sequence, returning every violation found (empty if legal).

    ``occupant`` (v3 18a.3, audit B9): see :func:`check_l6_geometric_legality`.
    """
    return [
        *check_l1_budget(program, ruleset),
        *check_l2_mandatory_displacement(state, program, color),
        *check_partial_program(state, program, color, ruleset, occupant=occupant),
    ]


def is_legal_program(
    state: State,
    program: Program,
    color: Color,
    ruleset: RuleSet,
    *,
    occupant: OccupantLookup | None = None,
) -> bool:
    """The boolean predicate :math:`L(s,\\pi_\\omega)` (spec §4.4).

    ``occupant`` (v3 18a.3, audit B9): see :func:`check_l6_geometric_legality`.
    """
    return check_legal_program(state, program, color, ruleset, occupant=occupant) == []
