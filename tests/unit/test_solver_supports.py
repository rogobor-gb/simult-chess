from __future__ import annotations

import random

import pytest

pytest.importorskip("numpy")

from conftest import build_state  # noqa: E402

from simult_chess.core import legality  # noqa: E402
from simult_chess.core.types import (  # noqa: E402
    Cancel,
    Color,
    Reservation,
    Square,
    Token,
)
from simult_chess.referee.setup import standard_starting_state  # noqa: E402
from simult_chess.rules.ruleset import RuleSet  # noqa: E402
from simult_chess.solver.supports import enumerate_support  # noqa: E402

RULESET = RuleSet()


def test_enumerate_support_returns_only_legal_programs() -> None:
    state = standard_starting_state()
    support = enumerate_support(state, Color.WHITE, RULESET, random.Random(0))
    assert support
    for program in support:
        assert legality.is_legal_program(state, program, Color.WHITE, RULESET)


def test_enumerate_support_is_deterministic_given_the_same_seed() -> None:
    state = standard_starting_state()
    support_a = enumerate_support(state, Color.WHITE, RULESET, random.Random(7))
    support_b = enumerate_support(state, Color.WHITE, RULESET, random.Random(7))
    assert support_a == support_b


def test_enumerate_support_respects_max_programs_cap() -> None:
    state = standard_starting_state()
    support = enumerate_support(
        state, Color.WHITE, RULESET, random.Random(1), max_programs=3
    )
    assert len(support) <= 3


def test_enumerate_support_includes_two_action_programs_when_legal() -> None:
    # Two independent, non-conflicting pawn pushes should both survive as a
    # legal 2-action program somewhere in a large-enough support.
    state = standard_starting_state()
    support = enumerate_support(
        state,
        Color.WHITE,
        RULESET,
        random.Random(3),
        max_single_actions=8,
        max_programs=64,
    )
    assert any(len(program) == 2 for program in support)


def test_enumerate_support_never_empty_when_a_displacement_exists() -> None:
    # Regression: pure capture-value sorting could truncate the pool down to
    # zero Move/Castle candidates (every survivor a same-value Reserve action
    # that happened to win the shuffled tie-break), making every resulting
    # program fail L2 -- found via a 500-game minimatch sweep (4 games hit it
    # at the default max_single_actions=8). A tiny cap makes the failure mode
    # far more likely, so stress it across many seeds here.
    state = standard_starting_state()
    for seed in range(200):
        support = enumerate_support(
            state, Color.WHITE, RULESET, random.Random(seed), max_single_actions=2
        )
        assert support, f"seed {seed}: empty support despite a legal displacement"
        for program in support:
            assert legality.is_legal_program(state, program, Color.WHITE, RULESET)


def test_enumerate_support_handles_a_sparse_board() -> None:
    king = Token(id=1, color=Color.WHITE, typ="k")
    state = build_state({king: Square(4, 0)})
    support = enumerate_support(state, Color.WHITE, RULESET, random.Random(0))
    assert support
    for program in support:
        assert legality.is_legal_program(state, program, Color.WHITE, RULESET)


def test_enumerate_support_can_include_a_cancel_when_a_reservation_stands() -> None:
    # D3: matrix_1ply's support now reaches Cancel (spec §9). A Cancel is
    # L2-illegal alone while a displacement exists, so it surfaces only as the
    # second action of a (Move, Cancel) pair. Small position -> the single
    # cancel candidate survives truncation; search seeds for the pairing.
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
    saw_cancel = False
    for seed in range(50):
        support = enumerate_support(
            state, Color.WHITE, RULESET, random.Random(seed), max_programs=64
        )
        for program in support:
            assert legality.is_legal_program(state, program, Color.WHITE, RULESET)
            if any(isinstance(a, Cancel) for a in program):
                saw_cancel = True
    assert saw_cancel, "matrix_1ply support never included a Cancel over 50 seeds"
