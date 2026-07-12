"""Resolution-trace checks R1-R18, T1, T3, T4.

INVARIANTS.md §5-6, check-point TRACE/STATE. Each `check_rN`/`check_tN`
function inspects a `(state_pre, state_post, trace)` triple from one
`phi()` call and returns the `Violation`s found. These are
redundant-by-design: the stage implementations already enforce these
properties algorithmically, but re-deriving/re-checking them here catches
regressions (a "corrupted Φ") that unit tests on individual stages might
not exercise in combination.

T2 ("no check/checkmate") is a static absence-of-a-rule property, not
something observable in a single trace — it has no runtime check here.
"""

from __future__ import annotations

from simult_chess.core import geometry
from simult_chess.core.collision import edge_conflict, vertex_conflict
from simult_chess.core.moves import DeclaredMove
from simult_chess.core.phi import PhiTrace
from simult_chess.core.stages.annihilate import resolve_annihilation
from simult_chess.core.stages.closure import detect_terminal
from simult_chess.core.types import State
from simult_chess.core.violation import Violation
from simult_chess.referee.serialize import public_position_key
from simult_chess.rules.ruleset import RuleSet

_COOLDOWN_EXEMPT_TYPES = frozenset({"p", "k"})


def check_r1_fizzle_f1(state_pre: State, trace: PhiTrace) -> list[Violation]:
    """R1 — a pawn diagonal capture fizzles iff its declaration-time target executed."""
    violations: list[Violation] = []
    occupant = geometry.occupant_lookup(state_pre.board)
    executing_ids = {m.token.id for m in trace.executing}
    for outcome in trace.fizzled:
        if outcome.cause != "F1":
            continue
        target = occupant(outcome.move.trajectory.destination)
        if target is None:
            detail = f"F1 fizzle of {outcome.move.token.id} has no target"
            violations.append(Violation("R1", detail))
            continue
        if target.id not in executing_ids:
            detail = (
                f"F1 fizzle of {outcome.move.token.id}: "
                f"target {target.id} did not execute"
            )
            violations.append(Violation("R1", detail))
    return violations


def check_r2_fizzle_f2(trace: PhiTrace, ruleset: RuleSet) -> list[Violation]:
    """R2 — F2 fizzles come in cross-color pairs; pawn-only if scoped that way."""
    violations: list[Violation] = []
    f2_outcomes = [o for o in trace.fizzled if o.cause == "F2"]
    by_dest: dict[object, list[DeclaredMove]] = {}
    for outcome in f2_outcomes:
        by_dest.setdefault(outcome.move.trajectory.destination, []).append(outcome.move)
    for destination, movers in by_dest.items():
        colors = {m.color for m in movers}
        if len(colors) < 2:
            detail = f"F2 fizzle at {destination} is not cross-color"
            violations.append(Violation("R2", detail))
        if ruleset.pawn_same_square_fizzle_scope == "both_pawns":
            if not all(m.token.typ == "p" for m in movers):
                detail = f"F2 fizzle at {destination} includes a non-pawn"
                violations.append(Violation("R2", detail))
    return violations


def check_r3_edge_conflict(trace: PhiTrace) -> list[Violation]:
    """R3 — an (E)-only annihilated pair has disjoint swept sets."""
    violations: list[Violation] = []
    for event in trace.annihilated:
        white_traj = event.white_move.trajectory
        black_traj = event.black_move.trajectory
        has_v = vertex_conflict(white_traj, black_traj)
        has_e = edge_conflict(white_traj, black_traj)
        if not (has_v or has_e):
            detail = f"annihilated pair at rank {event.rank} has no (V)/(E) conflict"
            violations.append(Violation("R3", detail))
        if has_e and not has_v and (white_traj.swept & black_traj.swept):
            detail = "(E)-only pair unexpectedly shares swept squares"
            violations.append(Violation("R3", detail))
    return violations


def check_r4_annihilation_matching(
    trace: PhiTrace, ruleset: RuleSet
) -> list[Violation]:
    """R4 — greedy rank-order matching reproduces the canonical surviving set."""
    recomputed = resolve_annihilation(trace.executing, ruleset)
    traced = frozenset(
        m for event in trace.annihilated for m in (event.white_move, event.black_move)
    )
    if recomputed.annihilated != traced:
        return [Violation("R4", "recomputed annihilation set differs from trace")]
    return []


def check_r5_one_survivor_per_destination(trace: PhiTrace) -> list[Violation]:
    """R5 — at most one surviving mover targets any given square."""
    counts: dict[object, int] = {}
    for move in trace.survivors:
        dest = move.trajectory.destination
        counts[dest] = counts.get(dest, 0) + 1
    return [
        Violation("R5", f"{count} survivors target {dest}")
        for dest, count in counts.items()
        if count > 1
    ]


def check_r6_vacated_square(state_pre: State, trace: PhiTrace) -> list[Violation]:
    """R6 — no token is captured on a square it vacated."""
    violations: list[Violation] = []
    origins = {token.id: square for token, square in state_pre.board.items()}
    executing_ids = {m.token.id for m in trace.executing}
    for token, square in trace.captured:
        if token.id in executing_ids and origins.get(token.id) == square:
            detail = f"token {token.id} captured at its own vacated origin {square}"
            violations.append(Violation("R6", detail))
    return violations


def check_r7_intermezzo_precedence(
    state_pre: State, trace: PhiTrace
) -> list[Violation]:
    """R7 — a fired defender is never captured at the origin it vacated to fire.

    A fired defender *can* legitimately be captured later in the same
    battery chain (at the contested square it moved to) — R7 is specifically
    that any attacker whose declared destination was the defender's own
    origin square misses, because the defender left first.
    """
    origins = {token.id: square for token, square in state_pre.board.items()}
    fired_defender_ids = {fired.defender.id for fired in trace.fired}
    captured_pairs = {(token.id, square) for token, square in trace.captured}
    violations: list[Violation] = []
    for defender_id in fired_defender_ids:
        origin = origins.get(defender_id)
        if origin is not None and (defender_id, origin) in captured_pairs:
            detail = f"fired defender {defender_id} captured at its vacated origin"
            violations.append(Violation("R7", detail))
    return violations


def check_r9_defender_fires_once(trace: PhiTrace) -> list[Violation]:
    """R9 — a defender fires at most once per phase."""
    defenders = [fired.defender.id for fired in trace.fired]
    duplicates = {d for d in defenders if defenders.count(d) > 1}
    return [Violation("R9", f"defender {d} fired more than once") for d in duplicates]


def check_r10_mover_as_defender_forbidden(trace: PhiTrace) -> list[Violation]:
    """R10 — no fired defender also executed a Move/Castle this phase."""
    mover_ids = {m.token.id for m in trace.executing}
    violations: list[Violation] = []
    for fired in trace.fired:
        if fired.defender.id in mover_ids:
            detail = f"fired defender {fired.defender.id} also moved this phase"
            violations.append(Violation("R10", detail))
    return violations


def check_r12_cascade_termination(state_pre: State, trace: PhiTrace) -> list[Violation]:
    """R12 — the cascade removes strictly fewer tokens than the live count."""
    live_count = len(state_pre.board)
    if len(trace.captured) >= live_count:
        detail = f"{len(trace.captured)} captures >= {live_count} live tokens"
        return [Violation("R12", detail)]
    return []


def check_r13_cooldown(state_pre: State, state_post: State) -> list[Violation]:
    """R13 — every cooled token displaced this phase and is neither pawn nor king."""
    violations: list[Violation] = []
    pre_by_id = {token.id: square for token, square in state_pre.board.items()}
    for token in state_post.cooldown:
        if token.typ in _COOLDOWN_EXEMPT_TYPES:
            detail = f"cooled token {token.id} is pawn/king"
            violations.append(Violation("R13", detail))
            continue
        post_square = state_post.board.get(token)
        pre_square = pre_by_id.get(token.id)
        if pre_square is not None and pre_square == post_square:
            detail = f"cooled token {token.id} did not displace this phase"
            violations.append(Violation("R13", detail))
    return violations


def check_r14_promotion(
    state_pre: State, state_post: State, trace: PhiTrace
) -> list[Violation]:
    """R14 — every promoted token enters cooldown; its type actually changed."""
    violations: list[Violation] = []
    pre_by_id = {token.id: token for token in state_pre.board}
    post_by_id = {token.id: token for token in state_post.board}
    cooled_ids = {token.id for token in state_post.cooldown}
    for token_id in trace.promoted:
        before = pre_by_id.get(token_id)
        after = post_by_id.get(token_id)
        if before is None or after is None:
            continue
        if before.typ == after.typ:
            detail = f"promoted token {token_id} type unchanged"
            violations.append(Violation("R14", detail))
        if token_id not in cooled_ids:
            violations.append(Violation("R14", f"promoted token {token_id} not cooled"))
    return violations


def check_r16_fizzled_inertness(
    state_pre: State, state_post: State, trace: PhiTrace
) -> list[Violation]:
    """R16 — a fizzled move itself leaves its token on the origin, uncooled.

    This is a claim about *that move*, not a lifetime guarantee for the
    token: a token whose own move fizzled is stationary, and a stationary,
    uncooled token remains eligible to *separately* fire as a recapturing
    defender this same phase (R10) — which legitimately displaces and
    cools it. Only flag a fizzled token that neither moved on its own
    (impossible, by definition) nor fired as a defender.
    """
    violations: list[Violation] = []
    pre_by_id = {token.id: square for token, square in state_pre.board.items()}
    post_by_id = {
        token.id: (token, square) for token, square in state_post.board.items()
    }
    cooled_ids = {token.id for token in state_post.cooldown}
    fired_defender_ids = {fired.defender.id for fired in trace.fired}
    for outcome in trace.fizzled:
        token_id = outcome.move.token.id
        if token_id in fired_defender_ids:
            continue  # legitimately displaced by firing, not by its own move
        pre_square = pre_by_id.get(token_id)
        post_entry = post_by_id.get(token_id)
        if pre_square is not None and post_entry is not None:
            if post_entry[1] != pre_square:
                detail = f"fizzled token {token_id} moved anyway"
                violations.append(Violation("R16", detail))
        if token_id in cooled_ids:
            violations.append(Violation("R16", f"fizzled token {token_id} was cooled"))
    return violations


def check_r17_cancellation(state_post: State, trace: PhiTrace) -> list[Violation]:
    """R17 — cancelled reservations are absent from the post-state."""
    all_kept = state_post.reservations_white + state_post.reservations_black
    violations: list[Violation] = []
    for reservation in trace.cancelled:
        if reservation in all_kept:
            detail = (
                f"cancelled reservation ({reservation.defender.id},"
                f"{reservation.protege.id}) still present"
            )
            violations.append(Violation("R17", detail))
    return violations


def check_r18_token_conservation(
    state_pre: State, state_post: State
) -> list[Violation]:
    """R18 — live tokens after are a subset (by id) of live tokens before."""
    pre_ids = {token.id for token in state_pre.board}
    post_ids = {token.id for token in state_post.board}
    extra = post_ids - pre_ids
    if extra:
        return [Violation("R18", f"token ids {extra} appeared without precedent")]
    return []


def check_t1_terminal(state_post: State, outcome: str) -> list[Violation]:
    """T1 — recorded outcome matches direct king-count inspection of the post-state.

    A "draw" may also be forced by T3/T4 on top of a board that still has
    both kings (`detect_terminal` returns "ongoing") — that's not a T1
    concern. This only fires when the board *itself* unambiguously implies
    a king-based terminal and the recorded outcome disagrees.
    """
    recomputed = detect_terminal(state_post.board)
    if recomputed != "ongoing" and outcome != recomputed:
        detail = f"board implies {recomputed!r} but outcome is {outcome!r}"
        return [Violation("T1", detail)]
    return []


def check_t3_repetition(state_post: State, outcome: str) -> list[Violation]:
    """T3 — a public position occurring 3+ times forces a draw."""
    key = public_position_key(state_post)
    count = state_post.bookkeeping.repetition_ledger.get(key, 0)
    still_ongoing = detect_terminal(state_post.board) == "ongoing"
    if count >= 3 and outcome != "draw" and still_ongoing:
        detail = f"position repeated {count} times but outcome is {outcome!r}"
        return [Violation("T3", detail)]
    return []


def check_t4_no_progress(
    state_post: State, outcome: str, ruleset: RuleSet
) -> list[Violation]:
    """T4 — the no-progress counter reaching the horizon forces a draw."""
    counter = state_post.bookkeeping.no_progress_counter
    still_ongoing = detect_terminal(state_post.board) == "ongoing"
    if counter >= ruleset.horizon and outcome != "draw" and still_ongoing:
        detail = f"no-progress counter {counter} but outcome is {outcome!r}"
        return [Violation("T4", detail)]
    return []


def check_all_trace(
    state_pre: State,
    state_post: State,
    trace: PhiTrace,
    outcome: str,
    ruleset: RuleSet,
) -> list[Violation]:
    """Run every R-*/T-* trace/state check for one Φ call."""
    return [
        *check_r1_fizzle_f1(state_pre, trace),
        *check_r2_fizzle_f2(trace, ruleset),
        *check_r3_edge_conflict(trace),
        *check_r4_annihilation_matching(trace, ruleset),
        *check_r5_one_survivor_per_destination(trace),
        *check_r6_vacated_square(state_pre, trace),
        *check_r7_intermezzo_precedence(state_pre, trace),
        *check_r9_defender_fires_once(trace),
        *check_r10_mover_as_defender_forbidden(trace),
        *check_r12_cascade_termination(state_pre, trace),
        *check_r13_cooldown(state_pre, state_post),
        *check_r14_promotion(state_pre, state_post, trace),
        *check_r16_fizzled_inertness(state_pre, state_post, trace),
        *check_r17_cancellation(state_post, trace),
        *check_r18_token_conservation(state_pre, state_post),
        *check_t1_terminal(state_post, outcome),
        *check_t3_repetition(state_post, outcome),
        *check_t4_no_progress(state_post, outcome, ruleset),
    ]
