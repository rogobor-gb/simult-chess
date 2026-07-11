"""The transition operator Φ: pure assembly of Stages F/A/B/C/D (spec §5, §14).

.. math:: s' = \\Phi(s, \\pi_W, \\pi_B)

`phi` is a pure function of its arguments: same `(state, program_white,
program_black, ruleset)` in, bit-identical `(state, outcome, trace)` out —
no wall-clock, no global mutable state, no unseeded randomness (inv M1).
"""

from __future__ import annotations

from dataclasses import dataclass

from simult_chess.core import legality
from simult_chess.core.moves import DeclaredMove, extract_declared_moves
from simult_chess.core.stages import closure
from simult_chess.core.stages.annihilate import AnnihilationEvent
from simult_chess.core.stages.closure import Outcome
from simult_chess.core.stages.defense import RecaptureFired
from simult_chess.core.stages.fizzle import FizzleOutcome
from simult_chess.core.types import (
    Bookkeeping,
    Cancel,
    Color,
    Move,
    PieceType,
    Program,
    Reservation,
    Reserve,
    Square,
    State,
    Token,
)
from simult_chess.referee.serialize import public_position_key
from simult_chess.rules import registry
from simult_chess.rules.ruleset import RuleSet

_BLACK_AGE_OFFSET = 1_000_000
"""[C]: guarantees age-stamp uniqueness across R_W/R_B (WF5) for reservations
declared in the same (simultaneous) phase, without needing — or being able
to construct — a true combined declaration order between two players'
independent programs."""


@dataclass(frozen=True, slots=True)
class PhiTrace:
    """The event log of one Φ call (inv §9's trace substructure)."""

    fizzled: tuple[FizzleOutcome, ...]
    executing: tuple[DeclaredMove, ...]
    annihilated: tuple[AnnihilationEvent, ...]
    survivors: tuple[DeclaredMove, ...]
    captured: tuple[tuple[Token, Square], ...]
    fired: tuple[RecaptureFired, ...]
    promoted: frozenset[int]


@dataclass(frozen=True, slots=True)
class PhiResult:
    """The output of one Φ call: the successor state, outcome, and trace."""

    state: State
    outcome: Outcome
    trace: PhiTrace


def _extract_new_reservations(
    phase_index: int, program_white: Program, program_black: Program
) -> tuple[Reservation, ...]:
    reservations: list[Reservation] = []
    for offset, program in ((0, program_white), (_BLACK_AGE_OFFSET, program_black)):
        for index, action in enumerate(program):
            if isinstance(action, Reserve):
                reservations.append(
                    Reservation(
                        defender=action.defender,
                        protege=action.protege,
                        age=(phase_index, offset + index),
                    )
                )
    return tuple(reservations)


def _extract_promotion_choices(
    program_white: Program, program_black: Program
) -> dict[int, PieceType]:
    choices: dict[int, PieceType] = {}
    for program in (program_white, program_black):
        for action in program:
            if isinstance(action, Move) and action.promotion is not None:
                choices[action.token.id] = action.promotion
    return choices


def _extract_cancellations(
    program_white: Program, program_black: Program
) -> frozenset[Reservation]:
    cancelled: set[Reservation] = set()
    for program in (program_white, program_black):
        for action in program:
            if isinstance(action, Cancel):
                cancelled.add(action.reservation)
    return frozenset(cancelled)


def phi(
    state: State,
    program_white: Program,
    program_black: Program,
    ruleset: RuleSet,
) -> PhiResult:
    """Resolve one phase: :math:`\\Phi(s,\\pi_W,\\pi_B) \\to (s',\\text{trace})`."""
    if not legality.is_legal_program(state, program_white, Color.WHITE, ruleset):
        raise ValueError("program_white violates L(s,π) (spec §4.4)")
    if not legality.is_legal_program(state, program_black, Color.BLACK, ruleset):
        raise ValueError("program_black violates L(s,π) (spec §4.4)")

    declared = extract_declared_moves(state, program_white, program_black)

    fizzle_resolver = registry.get_fizzle_resolver(ruleset)
    fizzle_result = fizzle_resolver(declared, state, ruleset)
    executing = tuple(m for m in declared if fizzle_result.executes(m))

    annihilation_matcher = registry.get_annihilation_matcher(ruleset)
    annihilation_result = annihilation_matcher(executing, ruleset)
    survivors = tuple(m for m in executing if annihilation_result.survives(m))

    new_reservations = _extract_new_reservations(
        state.bookkeeping.phase_index, program_white, program_black
    )
    reservations_white = state.reservations_white + tuple(
        r for r in new_reservations if r.defender.color is Color.WHITE
    )
    reservations_black = state.reservations_black + tuple(
        r for r in new_reservations if r.defender.color is Color.BLACK
    )

    defense_resolver = registry.get_defense_resolver(ruleset)
    defense_result = defense_resolver(
        executing, survivors, state, reservations_white, reservations_black, ruleset
    )

    promotion_choices = _extract_promotion_choices(program_white, program_black)
    final_board = closure.apply_promotions(defense_result.occupancy, promotion_choices)
    promoted_ids = frozenset(promotion_choices.keys())

    displaced_tokens = closure.compute_displaced_tokens(
        survivors, defense_result, final_board
    )
    displaced_ids = frozenset(t.id for t in displaced_tokens)
    recapturer_ids = frozenset(fired.defender.id for fired in defense_result.fired)
    new_cooldown = closure.compute_cooldown(displaced_tokens, recapturer_ids, ruleset)

    captured_ids = frozenset(t.id for t in defense_result.captured_tokens)
    annihilated_ids = frozenset(
        t.id for t in closure.annihilated_tokens(annihilation_result)
    )
    dead_ids = captured_ids | annihilated_ids

    cancellations = _extract_cancellations(program_white, program_black)
    kept_reservations = closure.update_reservations(
        reservations_white + reservations_black,
        state.bookkeeping.phase_index,
        displaced_ids,
        dead_ids,
        cancellations,
        ruleset,
    )
    final_reservations_white = tuple(
        r for r in kept_reservations if r.defender.color is Color.WHITE
    )
    final_reservations_black = tuple(
        r for r in kept_reservations if r.defender.color is Color.BLACK
    )

    new_castling_rights = closure.update_castling_rights(
        state.bookkeeping.castling_rights, state, displaced_ids
    )

    any_capture = bool(dead_ids)
    any_pawn_displacement = any(
        m.token.typ == "p" and m.token.id not in captured_ids for m in survivors
    )
    new_no_progress = closure.update_no_progress_counter(
        state.bookkeeping.no_progress_counter,
        any_capture=any_capture,
        any_pawn_displacement=any_pawn_displacement,
    )

    provisional_bookkeeping = Bookkeeping(
        castling_rights=new_castling_rights,
        repetition_ledger=state.bookkeeping.repetition_ledger,
        no_progress_counter=new_no_progress,
        phase_index=state.bookkeeping.phase_index + 1,
    )
    provisional_state = State(
        board=final_board,
        cooldown=new_cooldown,
        reservations_white=final_reservations_white,
        reservations_black=final_reservations_black,
        bookkeeping=provisional_bookkeeping,
    )
    position_key = public_position_key(provisional_state)
    new_ledger = closure.update_repetition_ledger(
        state.bookkeeping.repetition_ledger, position_key
    )

    final_state = State(
        board=final_board,
        cooldown=new_cooldown,
        reservations_white=final_reservations_white,
        reservations_black=final_reservations_black,
        bookkeeping=Bookkeeping(
            castling_rights=new_castling_rights,
            repetition_ledger=new_ledger,
            no_progress_counter=new_no_progress,
            phase_index=state.bookkeeping.phase_index + 1,
        ),
    )

    outcome = closure.detect_terminal(final_board)
    if outcome == "ongoing":
        if new_ledger.get(position_key, 0) >= 3:
            outcome = "draw"
        elif new_no_progress >= ruleset.horizon:
            outcome = "draw"

    trace = PhiTrace(
        fizzled=fizzle_result.outcomes,
        executing=executing,
        annihilated=annihilation_result.events,
        survivors=survivors,
        captured=defense_result.captured,
        fired=defense_result.fired,
        promoted=promoted_ids,
    )
    return PhiResult(state=final_state, outcome=outcome, trace=trace)
