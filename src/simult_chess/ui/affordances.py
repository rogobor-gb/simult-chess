"""Live affordances for a program under construction (Phase 16.2).

From ``(state, colour, ruleset, partial program)`` this computes what a UI needs
to guide a human building a program *before* submit: the legal destinations of
each own token, the legal castles, the admissible reservation pairings (letting
L6 handle the aggressive-dual future-destination case), the cancellable
reservations, and a **threat overlay**.

Every "is this legal?" question is answered by appending the candidate action to
the partial program and asking `core.legality.check_partial_program` — never a
reimplementation of the rules. Candidate *enumeration* reuses
`agents.candidates`' geometry, so this module adds no new rule logic.

**The threat overlay is declaration-time attack potential, not safety.** A
token in `threatened_tokens` is reachable by a pseudo-legal opponent capture
next phase; under simultaneity that attack may still fizzle, be annihilated, or
be pre-empted by a defensive recapture. It answers "what could be hit", not
"what will die".

Serves the Tk window (16.3) and a future web client identically.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from simult_chess.agents.candidates import exhaustive_move_and_castle_candidates
from simult_chess.core.legality import check_partial_program
from simult_chess.core.types import (
    Cancel,
    Castle,
    CastleSide,
    Color,
    Move,
    Program,
    Reserve,
    Square,
    State,
)
from simult_chess.rules.ruleset import RuleSet


@dataclass(frozen=True, slots=True)
class Affordances:
    """What is legal to add to a partial program right now, plus threats.

    Parameters
    ----------
    legal_destinations : Mapping[int, frozenset[Square]]
        Own ``token id`` → the squares it may move to such that appending that
        move keeps the partial program legal. A token already used in the
        partial program (L3) maps to an empty set.
    legal_castles : frozenset[CastleSide]
        Castle sides that may be added to the partial program.
    reservation_pairings : frozenset[tuple[int, int]]
        Admissible ``(defender id, protege id)`` reservations, including the
        aggressive dual (the protégé's declared destination this phase).
    cancellable_reservations : frozenset[int]
        Indices into ``state.reservations(colour)`` whose Cancel is admissible
        (empty when ``cancellation_enabled`` is off — L6 decides).
    threatened_tokens : frozenset[int]
        Own ``token id``s reachable by a pseudo-legal opponent capture next
        phase — *attack potential, not safety* (see module docstring).
    """

    legal_destinations: Mapping[int, frozenset[Square]]
    legal_castles: frozenset[CastleSide]
    reservation_pairings: frozenset[tuple[int, int]]
    cancellable_reservations: frozenset[int]
    threatened_tokens: frozenset[int]


def threat_overlay(state: State, color: Color) -> frozenset[int]:
    """Own ``token id``s a pseudo-legal opponent capture could reach next phase.

    Declaration-time attack potential (see module docstring), computed from the
    opponent's pseudo-legal Move geometry landing on one of `color`'s tokens.
    """
    square_to_token = {square: token for token, square in state.board.items()}
    threatened: set[int] = set()
    for action in exhaustive_move_and_castle_candidates(state, color.opponent):
        if not isinstance(action, Move):
            continue
        victim = square_to_token.get(action.trajectory.destination)
        if victim is not None and victim.color is color:
            threatened.add(victim.id)
    return frozenset(threatened)


def affordances(
    state: State,
    color: Color,
    ruleset: RuleSet,
    partial: Program = (),
) -> Affordances:
    """Compute all live affordances for `color`'s `partial` program.

    Pure function of the arguments; safe to call on every UI edit. Complexity is
    a handful of `check_partial_program` calls per candidate action — trivial
    for a 16-token board, and it shares the exact rule clauses with submit.
    """
    destinations: dict[int, set[Square]] = {}
    castles: set[CastleSide] = set()
    for action in exhaustive_move_and_castle_candidates(state, color):
        if check_partial_program(state, (*partial, action), color, ruleset):
            continue
        if isinstance(action, Move):
            destinations.setdefault(action.token.id, set()).add(
                action.trajectory.destination
            )
        elif isinstance(action, Castle):
            castles.add(action.side)

    pairings: set[tuple[int, int]] = set()
    own = [token for token in state.board if token.color is color]
    for defender in own:
        for protege in own:
            if protege is defender:
                continue
            reserve = Reserve(defender=defender, protege=protege)
            if not check_partial_program(state, (*partial, reserve), color, ruleset):
                pairings.add((defender.id, protege.id))

    cancellable: set[int] = set()
    for index, reservation in enumerate(state.reservations(color)):
        cancel = Cancel(reservation=reservation)
        if not check_partial_program(state, (*partial, cancel), color, ruleset):
            cancellable.add(index)

    return Affordances(
        legal_destinations={
            token_id: frozenset(squares) for token_id, squares in destinations.items()
        },
        legal_castles=frozenset(castles),
        reservation_pairings=frozenset(pairings),
        cancellable_reservations=frozenset(cancellable),
        threatened_tokens=threat_overlay(state, color),
    )
