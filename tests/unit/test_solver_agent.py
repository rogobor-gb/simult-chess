from __future__ import annotations

import random

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")

from conftest import build_state  # noqa: E402

from simult_chess.core import legality  # noqa: E402
from simult_chess.core.types import Color, Square, Token  # noqa: E402
from simult_chess.referee.setup import standard_starting_state  # noqa: E402
from simult_chess.rules.ruleset import RuleSet  # noqa: E402
from simult_chess.solver.agent import matrix_1ply  # noqa: E402

RULESET = RuleSet()


def test_matrix_1ply_produces_a_legal_program_for_both_colors() -> None:
    state = standard_starting_state()
    for color in (Color.WHITE, Color.BLACK):
        program = matrix_1ply(state, color, RULESET, random.Random(0))
        assert legality.is_legal_program(state, program, color, RULESET)


def test_matrix_1ply_is_deterministic_given_the_same_seed() -> None:
    state = standard_starting_state()
    program_a = matrix_1ply(state, Color.WHITE, RULESET, random.Random(7))
    program_b = matrix_1ply(state, Color.WHITE, RULESET, random.Random(7))
    assert program_a == program_b


def test_matrix_1ply_handles_a_sparse_board() -> None:
    king_w = Token(id=1, color=Color.WHITE, typ="k")
    king_b = Token(id=2, color=Color.BLACK, typ="k")
    state = build_state({king_w: Square(0, 0), king_b: Square(7, 7)})
    program = matrix_1ply(state, Color.WHITE, RULESET, random.Random(0))
    assert legality.is_legal_program(state, program, Color.WHITE, RULESET)


def test_matrix_1ply_takes_a_cooled_undefended_queen() -> None:
    # A capture is only *unconditionally* forced -- immune to whatever the
    # opponent replies with in the same simultaneous phase -- if the target
    # cannot flee at all. A merely undefended piece can still just move
    # away in the same phase (spec's own point: simultaneity means nothing
    # is "hanging" the way it is in sequential chess, section 8.3); a
    # *cooled* piece is genuinely immobile (L4), so this is the clean case.
    # Sample many times and check the capture dominates the equilibrium mix.
    king_w = Token(id=1, color=Color.WHITE, typ="k")
    king_b = Token(id=2, color=Color.BLACK, typ="k")
    rook_w = Token(id=3, color=Color.WHITE, typ="r")
    queen_b = Token(id=4, color=Color.BLACK, typ="q")
    state = build_state(
        {
            king_w: Square(0, 0),
            king_b: Square(7, 7),
            rook_w: Square(0, 4),
            queen_b: Square(0, 7),
        },
        cooldown=frozenset({queen_b}),
    )
    rng = random.Random(0)
    capture_count = 0
    trials = 20
    for _ in range(trials):
        program = matrix_1ply(state, Color.WHITE, RULESET, rng)
        if any(
            getattr(action, "trajectory", None) is not None
            and action.trajectory.destination == Square(0, 7)
            for action in program
        ):
            capture_count += 1
    assert capture_count >= trials * 0.6
