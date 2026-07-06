"""State well-formedness checks WF1-WF7 (INVARIANTS.md §3, check-point STATE).

Each ``check_wf*`` function returns the list of ``Violation``s found (empty
if none), rather than raising, so a harness (Phase 4) can run in either
strict mode (raise on first non-empty result) or lenient mode (aggregate
across a self-play sweep). Single-state checks take one ``State``;
transition checks compare a ``before``/``after`` pair.
"""

from __future__ import annotations

from collections.abc import Set as AbstractSet

from simult_chess.core.types import Color, State
from simult_chess.core.violation import Violation
from simult_chess.rules.ruleset import RuleSet

_VALID_TYPES = frozenset({"p", "n", "b", "r", "q", "k"})
_COOLDOWN_EXEMPT_TYPES = frozenset({"p", "k"})
_CASTLING_FLANK_FIELDS = (
    "white_kingside",
    "white_queenside",
    "black_kingside",
    "black_queenside",
)


def check_wf1_occupancy_injectivity(state: State) -> list[Violation]:
    """WF1 — no square holds two live tokens."""
    squares = list(state.board.values())
    if len(set(squares)) == len(squares):
        return []
    offenders: dict[object, int] = {}
    for square in squares:
        offenders[square] = offenders.get(square, 0) + 1
    shared = {square: count for square, count in offenders.items() if count > 1}
    return [Violation("WF1", f"square(s) occupied by multiple tokens: {shared}")]


def check_wf2_domain(state: State) -> list[Violation]:
    """WF2 (domain half) — every live token has a valid color and type."""
    return [
        Violation("WF2", f"token {token.id} has invalid type {token.typ!r}")
        for token in state.board
        if token.typ not in _VALID_TYPES
    ]


def check_wf2_type_constancy(
    before: State, after: State, *, promoted: AbstractSet[int] = frozenset()
) -> list[Violation]:
    """WF2 (constancy half) — typ changes only via a recorded promotion.

    Parameters
    ----------
    before, after : State
        States either side of one phase transition.
    promoted : AbstractSet[int]
        Token IDs that legitimately promoted this phase (spec §6.5).
    """
    violations: list[Violation] = []
    before_by_id = {token.id: token for token in before.board}
    after_by_id = {token.id: token for token in after.board}
    for token_id, before_token in before_by_id.items():
        after_token = after_by_id.get(token_id)
        if after_token is None:
            continue
        if after_token.color is not before_token.color:
            violations.append(Violation("WF2", f"token {token_id} changed color"))
        if after_token.typ != before_token.typ and token_id not in promoted:
            violations.append(
                Violation(
                    "WF2",
                    f"token {token_id} changed type "
                    f"{before_token.typ!r}->{after_token.typ!r} without promotion",
                )
            )
    return violations


def check_wf3_cooldown_membership(state: State) -> list[Violation]:
    """WF3 — cooldown tokens are live and neither pawn nor king."""
    violations: list[Violation] = []
    live = set(state.board.keys())
    for token in state.cooldown:
        if token not in live:
            violations.append(Violation("WF3", f"cooled token {token.id} is not live"))
        elif token.typ in _COOLDOWN_EXEMPT_TYPES:
            detail = f"cooled token {token.id} has exempt type {token.typ!r}"
            violations.append(Violation("WF3", detail))
    return violations


def check_wf4_king_count(
    state: State, *, allow_terminal: bool = False
) -> list[Violation]:
    """WF4 — each color has exactly one king (at most one; zero only if terminal)."""
    violations: list[Violation] = []
    counts = {Color.WHITE: 0, Color.BLACK: 0}
    for token in state.board:
        if token.typ == "k":
            counts[token.color] += 1
    for color, count in counts.items():
        if count > 1:
            violations.append(Violation("WF4", f"{color.value} has {count} kings"))
        elif count == 0 and not allow_terminal:
            violations.append(
                Violation("WF4", f"{color.value} has no king in a non-terminal state")
            )
    return violations


def check_wf5_reservation_order(state: State) -> list[Violation]:
    """WF5 — each R_omega is strictly age-ordered; age stamps are globally unique."""
    violations: list[Violation] = []
    all_ages: list[tuple[int, int]] = []
    for owner, reservations in (
        ("W", state.reservations_white),
        ("B", state.reservations_black),
    ):
        ages = [r.age for r in reservations]
        all_ages.extend(ages)
        if any(a >= b for a, b in zip(ages, ages[1:], strict=False)):
            violations.append(
                Violation("WF5", f"R_{owner} is not strictly age-ordered: {ages}")
            )
    if len(set(all_ages)) != len(all_ages):
        violations.append(
            Violation("WF5", f"age stamps not unique across R_W ∪ R_B: {all_ages}")
        )
    return violations


def check_wf6_reservation_referential_integrity(state: State) -> list[Violation]:
    """WF6 — reservations reference live, same-color, distinct tokens.

    Square-stability ("neither has displaced since registration") is not
    checkable from a single state; it is enforced procedurally at closure
    (spec §6.7) when Phase 3 lands.
    """
    violations: list[Violation] = []
    live = set(state.board.keys())
    for owner, reservations in (
        (Color.WHITE, state.reservations_white),
        (Color.BLACK, state.reservations_black),
    ):
        for r in reservations:
            if r.defender not in live:
                violations.append(
                    Violation("WF6", f"defender {r.defender.id} is not live")
                )
                continue
            if r.protege not in live:
                violations.append(
                    Violation("WF6", f"protege {r.protege.id} is not live")
                )
                continue
            if r.defender.color is not owner or r.protege.color is not owner:
                violations.append(
                    Violation(
                        "WF6",
                        f"reservation ({r.defender.id},{r.protege.id}) color "
                        f"mismatch with owner {owner.value}",
                    )
                )
            if r.defender == r.protege:
                detail = f"defender and protege are the same token {r.defender.id}"
                violations.append(Violation("WF6", detail))
    return violations


def check_wf7_bookkeeping_ranges(state: State, ruleset: RuleSet) -> list[Violation]:
    """WF7 (range half) — no-progress counter and phase index in valid ranges."""
    violations: list[Violation] = []
    nu = state.bookkeeping.no_progress_counter
    if not (0 <= nu <= ruleset.horizon):
        violations.append(
            Violation("WF7", f"no-progress counter {nu} outside [0,{ruleset.horizon}]")
        )
    if state.bookkeeping.phase_index < 0:
        violations.append(
            Violation("WF7", f"phase index {state.bookkeeping.phase_index} is negative")
        )
    return violations


def check_wf7_bookkeeping_monotone(before: State, after: State) -> list[Violation]:
    """WF7 (monotonicity half) — castling rights never regained; phase index +1."""
    violations: list[Violation] = []
    b_rights = before.bookkeeping.castling_rights
    a_rights = after.bookkeeping.castling_rights
    for flank in _CASTLING_FLANK_FIELDS:
        if getattr(a_rights, flank) and not getattr(b_rights, flank):
            violations.append(Violation("WF7", f"castling right {flank} was regained"))
    if after.bookkeeping.phase_index != before.bookkeeping.phase_index + 1:
        violations.append(
            Violation(
                "WF7",
                f"phase index {before.bookkeeping.phase_index} -> "
                f"{after.bookkeeping.phase_index} did not increment by exactly 1",
            )
        )
    return violations


def check_all_state(
    state: State, ruleset: RuleSet, *, allow_terminal: bool = False
) -> list[Violation]:
    """Run every single-state WF check (WF1-4, WF5, WF6, WF7 ranges)."""
    return [
        *check_wf1_occupancy_injectivity(state),
        *check_wf2_domain(state),
        *check_wf3_cooldown_membership(state),
        *check_wf4_king_count(state, allow_terminal=allow_terminal),
        *check_wf5_reservation_order(state),
        *check_wf6_reservation_referential_integrity(state),
        *check_wf7_bookkeeping_ranges(state, ruleset),
    ]
