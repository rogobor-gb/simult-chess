"""Stage F — fizzle resolution (spec §6.2; INVARIANTS.md R1, R2, R16).

Both fizzle sources are functions of declarations alone — computable before
any collision — because whether a token vacates depends only on whether it
*starts* moving (spec §3), not on whether it survives.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from simult_chess.core import geometry
from simult_chess.core.moves import DeclaredMove
from simult_chess.core.types import State
from simult_chess.rules.ruleset import RuleSet

FizzleCause = Literal["F1", "F2"]


@dataclass(frozen=True, slots=True)
class FizzleOutcome:
    """One fizzled move and why (inv §9 trace: "fizzled: [...] # F1/F2 outcomes")."""

    move: DeclaredMove
    cause: FizzleCause


@dataclass(frozen=True, slots=True)
class FizzleResult:
    """The output of Stage F: which declared moves fizzle, and why."""

    outcomes: tuple[FizzleOutcome, ...]

    @property
    def fizzled(self) -> frozenset[DeclaredMove]:
        """The set of fizzled moves, dropping their F1/F2 provenance."""
        return frozenset(outcome.move for outcome in self.outcomes)

    def executes(self, move: DeclaredMove) -> bool:
        """Whether `move`'s token vacates this phase (spec §3)."""
        return move not in self.fizzled


def _is_pawn_diagonal_capture(move: DeclaredMove) -> bool:
    if move.token.typ != "p":
        return False
    if len(move.trajectory.path) != 2:
        return False
    origin, destination = move.trajectory.path
    return origin.file != destination.file


def _resolve_f2(
    declared: tuple[DeclaredMove, ...], ruleset: RuleSet
) -> frozenset[DeclaredMove]:
    """F2 — opposing pawn pushes onto a shared square both fizzle (spec §6.2)."""
    by_destination: dict[object, list[DeclaredMove]] = {}
    for move in declared:
        by_destination.setdefault(move.trajectory.destination, []).append(move)

    fizzled: set[DeclaredMove] = set()
    for movers in by_destination.values():
        if len(movers) < 2:
            continue
        colors = {m.color for m in movers}
        if len(colors) < 2:
            continue  # same-color destination collisions are barred at declaration (L5)
        if ruleset.pawn_same_square_fizzle_scope == "both_pawns":
            if all(m.token.typ == "p" for m in movers):
                fizzled.update(movers)
        else:
            fizzled.update(movers)
    return frozenset(fizzled)


def _resolve_f1(
    declared: tuple[DeclaredMove, ...],
    f2_fizzled: frozenset[DeclaredMove],
    state: State,
    tie_break: Sequence[DeclaredMove] | None,
) -> frozenset[DeclaredMove]:
    """F1 — a pawn diagonal capture fizzles iff its target executes (Lemma 6.2)."""
    occupant = geometry.occupant_lookup(state.board)
    move_by_token = {m.token: m for m in declared}
    pawn_captures = [m for m in declared if _is_pawn_diagonal_capture(m)]

    def target_move_of(move: DeclaredMove) -> DeclaredMove | None:
        target_token = occupant(move.trajectory.destination)
        if target_token is None:
            return None
        return move_by_token.get(target_token)

    # The only cycles in the F1 dependency digraph are mutual 2-cycles,
    # which are exactly (E) edge-conflicts (Lemma 6.2): both execute and are
    # annihilated later in Stage A, so neither fizzles via F1.
    cycle_members: set[DeclaredMove] = set()
    for move in pawn_captures:
        partner = target_move_of(move)
        if partner is not None and target_move_of(partner) is move:
            cycle_members.add(move)
            cycle_members.add(partner)

    memo: dict[DeclaredMove, bool] = {}

    def executes(move: DeclaredMove | None) -> bool:
        if move is None:
            return False
        if move in f2_fizzled:
            return False
        if move in cycle_members:
            return True
        if not _is_pawn_diagonal_capture(move):
            return True
        return not resolve(move)

    def resolve(move: DeclaredMove) -> bool:
        # Only ever called for non-cycle pawn-capture moves (see `executes`).
        if move in memo:
            return memo[move]
        result = executes(target_move_of(move))
        memo[move] = result
        return result

    non_cycle_captures = [m for m in pawn_captures if m not in cycle_members]
    order = list(tie_break) if tie_break is not None else non_cycle_captures
    for move in order:
        resolve(move)

    return frozenset(m for m in non_cycle_captures if resolve(m))


def resolve_fizzles(
    declared: tuple[DeclaredMove, ...],
    state: State,
    ruleset: RuleSet,
    *,
    tie_break: Sequence[DeclaredMove] | None = None,
) -> FizzleResult:
    """Resolve Stage F: F2 pawn convergence, then F1 vacated-capture (spec §6.2).

    Parameters
    ----------
    declared : tuple[DeclaredMove, ...]
        :math:`M`, the full multiset of declared moves.
    state : State
        The declaration-time state (F1 needs the board to find capture targets).
    ruleset : RuleSet
        Supplies `pawn_same_square_fizzle_scope` (inv R2, `[K]`).
    tie_break : Sequence[DeclaredMove] | None
        An optional processing order for the F1 backward induction (Lemma
        6.2's provably-commuting choice). The result is invariant to it —
        `resolve_fizzles` is memoized over a DAG, so any permutation of the
        non-cyclic pawn-capture moves yields identical output (inv M2a).
    """
    f2_fizzled = _resolve_f2(declared, ruleset)
    f1_fizzled = _resolve_f1(declared, f2_fizzled, state, tie_break)
    outcomes = tuple(
        FizzleOutcome(move=m, cause="F2") for m in declared if m in f2_fizzled
    ) + tuple(FizzleOutcome(move=m, cause="F1") for m in f1_fizzled)
    return FizzleResult(outcomes=outcomes)
