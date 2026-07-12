from __future__ import annotations

import random

from conftest import build_state

from simult_chess.agents.greedy import greedy_program
from simult_chess.agents.random_legal import random_legal_program
from simult_chess.core import legality
from simult_chess.core.types import Color, Move, Square, Token
from simult_chess.referee.setup import standard_starting_state
from simult_chess.rules.ruleset import RuleSet

RULESET = RuleSet()


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
