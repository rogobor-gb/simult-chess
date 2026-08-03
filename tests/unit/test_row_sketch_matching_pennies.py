"""v3 18c.1: the row-sketch node solver, validated against the same
fixtures `test_search_matching_pennies.py` already uses -- not re-derived,
deliberately the same constructions, so this is testing whether the *new*
node representation converges where the *old* one (H4: biased in-tree
regret via a `Q[a]` running-mean baseline) measurably didn't.

**The decisive comparison.** `test_search_matching_pennies.py`'s two
asymmetric fixtures (18a′.1: weighted RPS, unequal-support) are `xfail`-
marked: even after 18a′.2's decaying-step-size mitigation, mean TV
distance to the known equilibrium is ~0.23-0.25 at the production budget
(`SearchConfig().simulations = 128`), against a 0.07 tolerance. Those
tests' own docstrings say the real fix needs "(E3) explicit row
evaluation... really part of 18c's architecture." This file's tests on
the *same* fixtures assert **pass**, not `xfail` -- and do, with the
target read off `read_out` (18c.1's actual deliverable, the max-entropy
equilibrium of the fully-observed stage-matrix sketch) rather than the
search's own in-tree `average_strategy()`: at TV distances on the order
of 1e-4 to 1e-9, not merely under 0.07. This is the expected shape of the
result, not a coincidence -- for these small, single-stage (`max_depth=1`)
fixtures with a deterministic synthetic evaluator, 128 rounds of row/
column sketching observes every pool cell many times over, so the
post-hoc entropy solve on the (near-)fully-populated exact matrix
recovers the equilibrium almost exactly, while the noisier per-round
regret-matching accumulation alone (still checked below, informationally)
lands in between the old ~0.23-0.25 and `read_out`'s near-exact result.
"""

from __future__ import annotations

import random
import statistics

import pytest

from simult_chess.core.types import (
    Bookkeeping,
    CastlingRights,
    Color,
    Move,
    Square,
    State,
    Token,
)
from simult_chess.learn.action_grid import NO_SECOND_INDEX, slot2_legal_actions
from simult_chess.learn.config import SearchConfig
from simult_chess.learn.row_sketch import make_root, read_out, run_simulations
from simult_chess.rules.ruleset import RuleSet

RULESET = RuleSet(n_actions=1)

# --- Matching-Pennies dodge (identical construction to test_search_
# matching_pennies.py's own _dodge_state) -----------------------------------

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


class _UniformPriorEvaluator:
    """Identical role to `test_search_matching_pennies.py`'s own class of
    the same name: no informative prior, isolates the row-sketch mechanism
    itself from any prior-driven shortcut."""

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


def test_mp_dodge_readout_value_matches_minus_one_half() -> None:
    # Pool sized generously (16 >= this fixture's 4/11 legal actions) so a
    # uniform (uninformative) prior can't accidentally truncate a critical
    # action out by an arbitrary tie-break -- see row_sketch.seed_program_
    # pool's own docstring on this exact risk.
    evaluator = _UniformPriorEvaluator()
    root = make_root(_dodge_state())
    run_simulations(
        root, RULESET, evaluator, SearchConfig().simulations, random.Random(0),
        max_depth=1, pool_size=16, pool_seed_size=16,
    )
    readout = read_out(root, evaluator, RULESET)
    assert readout.value == pytest.approx(-0.5, abs=1e-6)


def test_mp_dodge_readout_policy_matches_the_known_uniform_mixing() -> None:
    evaluator = _UniformPriorEvaluator()
    root = make_root(_dodge_state())
    run_simulations(
        root, RULESET, evaluator, SearchConfig().simulations, random.Random(1),
        max_depth=1, pool_size=16, pool_seed_size=16,
    )
    readout = read_out(root, evaluator, RULESET)
    assert root.white is not None and root.black is not None

    flee_index = next(
        i
        for i, program in enumerate(root.white.programs)
        if program[0].token == _WHITE_KING
    )
    stay_mass = sum(
        p for i, p in enumerate(readout.row_strategy) if i != flee_index
    )
    assert readout.row_strategy[flee_index] == pytest.approx(0.5, abs=1e-4)
    assert stay_mass == pytest.approx(0.5, abs=1e-4)

    aim_indices = [
        i
        for i, program in enumerate(root.black.programs)
        if isinstance(program[0], Move)
        and program[0].token == _KNIGHT
        and program[0].trajectory.path[-1] in (Square(0, 1), Square(1, 0))
    ]
    assert len(aim_indices) == 2
    aim_mass = sum(readout.col_strategy[i] for i in aim_indices)
    assert aim_mass == pytest.approx(1.0, abs=1e-4)
    for i in aim_indices:
        assert readout.col_strategy[i] == pytest.approx(0.5, abs=1e-4)


# --- Weighted rock-paper-scissors and unequal-support-size (identical
# construction to test_search_matching_pennies.py's own 18a′.1 fixtures) ----

_ASYM_WHITE_KING = Token(id=101, color=Color.WHITE, typ="k")
_ASYM_BLACK_KING = Token(id=102, color=Color.BLACK, typ="k")
_ASYM_WHITE_KNIGHT = Token(id=103, color=Color.WHITE, typ="n")
_ASYM_BLACK_KNIGHT = Token(id=104, color=Color.BLACK, typ="n")

_BLACK_DEST_3 = {
    Square(6, 4): 0,  # g5
    Square(5, 5): 1,  # f6
    Square(5, 7): 2,  # f8
}


def _asym_state(
    white_king: Square, white_knight: Square, black_king: Square, black_knight: Square
) -> State:
    return State(
        board={
            _ASYM_WHITE_KING: white_king,
            _ASYM_WHITE_KNIGHT: white_knight,
            _ASYM_BLACK_KING: black_king,
            _ASYM_BLACK_KNIGHT: black_knight,
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


class _MatrixGameEvaluator:
    """Identical role to `test_search_matching_pennies.py`'s own class of
    the same name: a fixed payoff matrix, keyed by which corner square
    each side's knight ended up on."""

    def __init__(
        self,
        matrix: list[list[float]],
        white_dest: dict[Square, int],
        black_dest: dict[Square, int],
        scale: float,
    ) -> None:
        self.matrix = matrix
        self.white_dest = white_dest
        self.black_dest = black_dest
        self.scale = scale

    def evaluate_leaf(
        self, state: State, ruleset: RuleSet
    ) -> tuple[float, dict[int, float], dict[int, float], object]:
        row = self.white_dest.get(state.board[_ASYM_WHITE_KNIGHT], 0)
        col = self.black_dest.get(state.board[_ASYM_BLACK_KNIGHT], 0)
        return self.matrix[row][col] * self.scale, {}, {}, None

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


def _aggregate_by_destination(
    strategy: object, programs: tuple, knight: Token, dest: dict[Square, int], n: int
) -> list[float]:
    out = [0.0] * n
    for i, p in enumerate(strategy):
        category = 0
        for action in programs[i]:
            if isinstance(action, Move) and action.token == knight:
                category = dest.get(action.trajectory.path[-1], 0)
        out[category] += float(p)
    return out


def _total_variation(p: list[float], q: list[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q, strict=True))


_RPS_MATRIX = [[0.0, -1.0, 2.0], [1.0, 0.0, -1.0], [-2.0, 1.0, 0.0]]
_RPS_EQUILIBRIUM = [0.25, 0.5, 0.25]
_RPS_SCALE = 0.5
_RPS_WHITE_DEST = {
    Square(1, 3): 0,  # a2 -> b4
    Square(2, 2): 1,  # a2 -> c3
    Square(2, 0): 2,  # a2 -> c1
}


def _rps_state() -> State:
    return _asym_state(Square(0, 0), Square(0, 1), Square(7, 7), Square(7, 6))


def test_weighted_rps_readout_converges_at_the_production_budget() -> None:
    # test_search_matching_pennies.py::test_weighted_rps_average_strategy_
    # converges_to_known_equilibrium is xfail-marked at this exact budget
    # (mean TV ~0.23-0.24, vs a 0.07 tolerance). This asserts pass, at a
    # tolerance two orders of magnitude tighter.
    evaluator = _MatrixGameEvaluator(
        _RPS_MATRIX, _RPS_WHITE_DEST, _BLACK_DEST_3, _RPS_SCALE
    )
    n_simulations = SearchConfig().simulations
    white_tv = []
    black_tv = []
    value_err = []
    for seed in range(10):
        root = make_root(_rps_state())
        run_simulations(
            root, RULESET, evaluator, n_simulations, random.Random(seed), max_depth=1
        )
        readout = read_out(root, evaluator, RULESET)
        assert root.white is not None and root.black is not None
        white_mass = _aggregate_by_destination(
            readout.row_strategy,
            root.white.programs,
            _ASYM_WHITE_KNIGHT,
            _RPS_WHITE_DEST,
            3,
        )
        black_mass = _aggregate_by_destination(
            readout.col_strategy,
            root.black.programs,
            _ASYM_BLACK_KNIGHT,
            _BLACK_DEST_3,
            3,
        )
        white_tv.append(_total_variation(white_mass, _RPS_EQUILIBRIUM))
        black_tv.append(_total_variation(black_mass, _RPS_EQUILIBRIUM))
        value_err.append(abs(readout.value - 0.0))

    assert statistics.mean(white_tv) < 1e-3
    assert statistics.mean(black_tv) < 1e-3
    assert statistics.mean(value_err) < 1e-3


_UNEQUAL_MATRIX = [[3.0, -1.0, -2.0], [-2.0, 2.0, 1.0]]
_UNEQUAL_SCALE = 1.0 / 3.0
_UNEQUAL_VALUE = -0.125 * _UNEQUAL_SCALE
_UNEQUAL_WHITE_DEST = {
    Square(1, 2): 0,  # a1 -> b3
    Square(2, 1): 1,  # a1 -> c2
}


def _unequal_support_state() -> State:
    return _asym_state(Square(7, 0), Square(0, 0), Square(7, 7), Square(7, 6))


def test_unequal_support_readout_converges_at_the_production_budget() -> None:
    # test_search_matching_pennies.py's own xfail companion for this
    # fixture. Exact ground truth (`solve_zero_sum` on _UNEQUAL_MATRIX):
    # value=-0.125, row=[0.375, 0.625], col=[0.375, 0.0, 0.625] -- column 1
    # is strictly dominated, a test of whether the estimator wastes mass on
    # it rather than just of the two supported ones.
    evaluator = _MatrixGameEvaluator(
        _UNEQUAL_MATRIX, _UNEQUAL_WHITE_DEST, _BLACK_DEST_3, _UNEQUAL_SCALE
    )
    n_simulations = SearchConfig().simulations
    value_err = []
    dominated_mass = []
    for seed in range(10):
        root = make_root(_unequal_support_state())
        run_simulations(
            root, RULESET, evaluator, n_simulations, random.Random(seed), max_depth=1
        )
        readout = read_out(root, evaluator, RULESET)
        assert root.black is not None
        black_mass = _aggregate_by_destination(
            readout.col_strategy,
            root.black.programs,
            _ASYM_BLACK_KNIGHT,
            _BLACK_DEST_3,
            3,
        )
        value_err.append(abs(readout.value - _UNEQUAL_VALUE))
        dominated_mass.append(black_mass[1])

    assert statistics.mean(value_err) < 1e-3
    assert statistics.mean(dominated_mass) < 1e-3
