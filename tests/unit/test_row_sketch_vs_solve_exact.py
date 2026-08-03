"""v3 18c.1 cross-check: the row-sketch node solver's `read_out` value
against `solve_exact`'s independently-computed exact value, on the two
18b fixtures with genuine cross-validated ground truth
(`matching_pennies_dodge`, `three_way_dodge` -- `tests/unit/test_solver_
exact.py`, `test_solver_depth_more_fixtures.py`). A stronger check than
anything available when the original slot-1-only search was validated:
that test suite only had a search-based convergence proof to lean on, not
an independent backward-induction solve.

Both solved the same way `solve_exact` itself is (`RuleSet(n_actions=1)`,
`max_depth=1`), with a `_UniformPriorEvaluator` matching `max_depth`'s own
0.0-fallback convention for any state still "ongoing" at the cutoff --
apples to apples with `solve_exact`'s own `is_depth_limited` leaves."""

from __future__ import annotations

import random

import pytest

from simult_chess.core.types import (
    Bookkeeping,
    CastlingRights,
    Color,
    Square,
    State,
    Token,
)
from simult_chess.learn.action_grid import NO_SECOND_INDEX, slot2_legal_actions
from simult_chess.learn.row_sketch import make_root, read_out, run_simulations
from simult_chess.rules.ruleset import RuleSet
from simult_chess.solver.exact import solve_exact

RULESET = RuleSet(n_actions=1)


class _UniformPriorEvaluator:
    def evaluate_leaf(
        self, state: State, ruleset: RuleSet
    ) -> tuple[float, dict[int, float], dict[int, float], object]:
        return 0.0, {}, {}, None

    def slot2_prior(
        self,
        context: object,
        color: Color,
        state: State,
        ruleset: RuleSet,
        first_index: int,
        first: object,
    ) -> dict[int, float]:
        actions, single_legal = slot2_legal_actions(state, color, ruleset, first)  # type: ignore[arg-type]
        keys = list(actions) + ([NO_SECOND_INDEX] if single_legal else [])
        return dict.fromkeys(keys, 1.0)


_WHITE_KING = Token(id=1, color=Color.WHITE, typ="k")
_BLACK_KING = Token(id=2, color=Color.BLACK, typ="k")
_BLOCKER_A1 = Token(id=10, color=Color.WHITE, typ="p")
_BLOCKER_A3 = Token(id=11, color=Color.WHITE, typ="p")
_BLOCKER_B2 = Token(id=12, color=Color.WHITE, typ="p")
_BLOCKER_B3 = Token(id=13, color=Color.WHITE, typ="p")
_KNIGHT = Token(id=4, color=Color.BLACK, typ="n")


def _dodge_state() -> State:
    return State(
        board={
            _WHITE_KING: Square(0, 1),
            _BLACK_KING: Square(7, 7),
            _BLOCKER_A1: Square(0, 0),
            _BLOCKER_A3: Square(0, 2),
            _BLOCKER_B2: Square(1, 1),
            _BLOCKER_B3: Square(1, 2),
            _KNIGHT: Square(2, 2),
        },
        cooldown=frozenset(),
        reservations_white=(),
        reservations_black=(),
        bookkeeping=Bookkeeping(
            castling_rights=CastlingRights(False, False, False, False),
            repetition_ledger={},
            no_progress_counter=0,
            phase_index=0,
        ),
    )


def test_matching_pennies_dodge_readout_matches_solve_exact() -> None:
    exact_value = solve_exact(
        _dodge_state(), RULESET, max_nodes=5000, max_depth=1
    ).root.value
    evaluator = _UniformPriorEvaluator()
    root = make_root(_dodge_state())
    run_simulations(
        root, RULESET, evaluator, 2000, random.Random(0),
        max_depth=1, pool_size=16, pool_seed_size=16,
    )
    readout = read_out(root, evaluator, RULESET)
    assert readout.value == pytest.approx(exact_value, abs=1e-3)


_T_WHITE_KING = Token(id=1, color=Color.WHITE, typ="k")
_T_BLOCKER_A1 = Token(id=10, color=Color.WHITE, typ="p")
_T_BLOCKER_A3 = Token(id=11, color=Color.WHITE, typ="p")
_T_BLOCKER_B2 = Token(id=12, color=Color.WHITE, typ="p")
_T_BLACK_KING = Token(id=2, color=Color.BLACK, typ="k")
_T_KNIGHT_1 = Token(id=4, color=Color.BLACK, typ="n")
_T_KNIGHT_2 = Token(id=5, color=Color.BLACK, typ="n")


def _three_way_state() -> State:
    return State(
        board={
            _T_WHITE_KING: Square(0, 1),
            _T_BLOCKER_A1: Square(0, 0),
            _T_BLOCKER_A3: Square(0, 2),
            _T_BLOCKER_B2: Square(1, 1),
            _T_BLACK_KING: Square(7, 7),
            _T_KNIGHT_1: Square(2, 2),
            _T_KNIGHT_2: Square(3, 3),
        },
        cooldown=frozenset(),
        reservations_white=(),
        reservations_black=(),
        bookkeeping=Bookkeeping(
            castling_rights=CastlingRights(False, False, False, False),
            repetition_ledger={},
            no_progress_counter=0,
            phase_index=0,
        ),
    )


def test_three_way_dodge_readout_matches_solve_exact() -> None:
    exact_value = solve_exact(
        _three_way_state(), RULESET, max_nodes=20_000, max_depth=1
    ).root.value
    evaluator = _UniformPriorEvaluator()
    root = make_root(_three_way_state())
    run_simulations(
        root, RULESET, evaluator, 2000, random.Random(0),
        max_depth=1, pool_size=19, pool_seed_size=19,
    )
    readout = read_out(root, evaluator, RULESET)
    assert readout.value == pytest.approx(exact_value, abs=1e-3)
