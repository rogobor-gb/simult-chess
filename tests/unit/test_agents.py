from __future__ import annotations

import random

from conftest import build_state

from simult_chess.agents.candidates import cancel_candidates
from simult_chess.agents.greedy import greedy_program
from simult_chess.agents.random_legal import random_legal_program
from simult_chess.core import legality
from simult_chess.core.types import (
    Cancel,
    Color,
    Move,
    Reservation,
    Square,
    State,
    Token,
)
from simult_chess.referee.setup import standard_starting_state
from simult_chess.rules.ruleset import RuleSet

RULESET = RuleSet()


def _state_with_a_white_reservation() -> tuple[State, Reservation]:
    """Contact pawn defence (e3-pawn defends d4-pawn), both kings present so
    a legal displacement exists (which makes a Cancel-only program L2-illegal)."""
    white_king = Token(id=100, color=Color.WHITE, typ="k")
    black_king = Token(id=200, color=Color.BLACK, typ="k")
    d4_pawn = Token(id=1, color=Color.WHITE, typ="p")
    e3_pawn = Token(id=2, color=Color.WHITE, typ="p")
    reservation = Reservation(defender=e3_pawn, protege=d4_pawn, age=(0, 0))
    state = build_state(
        {
            white_king: Square(0, 0),
            black_king: Square(7, 7),
            d4_pawn: Square(3, 3),
            e3_pawn: Square(4, 2),
        },
        reservations_white=(reservation,),
    )
    return state, reservation


def test_random_legal_program_produces_legal_programs() -> None:
    state = standard_starting_state()
    rng = random.Random(0)
    for _ in range(50):
        program = random_legal_program(state, Color.WHITE, RULESET, rng)
        assert legality.is_legal_program(state, program, Color.WHITE, RULESET)


def test_random_legal_program_is_deterministic_given_the_same_seed() -> None:
    state = standard_starting_state()
    program_a = random_legal_program(state, Color.WHITE, RULESET, random.Random(42))
    program_b = random_legal_program(state, Color.WHITE, RULESET, random.Random(42))
    assert program_a == program_b


def test_random_legal_program_handles_a_sparse_board() -> None:
    king = Token(id=1, color=Color.WHITE, typ="k")
    state = build_state({king: Square(4, 0)})
    program = random_legal_program(state, Color.WHITE, RULESET, random.Random(0))
    assert legality.is_legal_program(state, program, Color.WHITE, RULESET)


def test_cancel_candidates_one_per_standing_reservation() -> None:
    # D3: the shared candidate generation now emits Cancel (was the root of
    # the Phase 11b structural-0.000 cancellation caveat).
    state, reservation = _state_with_a_white_reservation()
    cancels = cancel_candidates(state, Color.WHITE)
    assert cancels == [Cancel(reservation=reservation)]
    # Black has no standing reservation here, so no Cancel candidate.
    assert cancel_candidates(state, Color.BLACK) == []


def test_random_legal_can_now_construct_a_cancel_program() -> None:
    # With a reservation standing, some seed must sample a Move+Cancel
    # combination -- the fuzzer can finally reach the mechanic. Every emitted
    # program stays legal.
    state, _ = _state_with_a_white_reservation()
    saw_cancel = False
    for seed in range(200):
        program = random_legal_program(state, Color.WHITE, RULESET, random.Random(seed))
        assert legality.is_legal_program(state, program, Color.WHITE, RULESET)
        if any(isinstance(a, Cancel) for a in program):
            saw_cancel = True
    assert saw_cancel, "random_legal never constructed a Cancel over 200 seeds"


def test_greedy_program_prefers_the_highest_value_capture() -> None:
    king = Token(id=1, color=Color.WHITE, typ="k")
    rook = Token(id=2, color=Color.WHITE, typ="r")
    enemy_pawn = Token(id=3, color=Color.BLACK, typ="p")
    enemy_queen = Token(id=4, color=Color.BLACK, typ="q")
    state = build_state(
        {
            king: Square(0, 0),
            rook: Square(4, 4),
            enemy_pawn: Square(4, 1),
            enemy_queen: Square(4, 7),
        }
    )
    program = greedy_program(state, Color.WHITE, RULESET, random.Random(0))
    assert len(program) == 1
    action = program[0]
    assert isinstance(action, Move)
    assert action.token == rook
    assert action.trajectory.destination == Square(4, 7)  # captures the queen


def test_greedy_program_is_legal_and_single_action() -> None:
    state = standard_starting_state()
    rng = random.Random(1)
    program = greedy_program(state, Color.WHITE, RULESET, rng)
    assert len(program) == 1
    assert legality.is_legal_program(state, program, Color.WHITE, RULESET)
