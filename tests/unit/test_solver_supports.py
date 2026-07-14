from __future__ import annotations

import random

import pytest

pytest.importorskip("numpy")

from conftest import build_state  # noqa: E402

from simult_chess.core import legality  # noqa: E402
from simult_chess.core.types import Color, Square, Token  # noqa: E402
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


def test_enumerate_support_handles_a_sparse_board() -> None:
    king = Token(id=1, color=Color.WHITE, typ="k")
    state = build_state({king: Square(4, 0)})
    support = enumerate_support(state, Color.WHITE, RULESET, random.Random(0))
    assert support
    for program in support:
        assert legality.is_legal_program(state, program, Color.WHITE, RULESET)
