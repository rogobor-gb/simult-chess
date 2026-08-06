"""v3 18d.1: Magnetic Mirror Descent (MMD) as an alternative row-sketch
update rule (`selection="mmd"`). Two layers, matching this session's own
"validate the mechanism at small scale before trusting it downstream"
practice: `_mmd_update` in isolation on toy inputs, then the decisive
check -- the same fixtures `test_row_sketch_matching_pennies.py`'s RM+/
Hedge tests already use, reused (not re-derived), checking MMD's *last
iterate* (`average_strategy()`, no averaging under `selection="mmd"`)
converges to the same known equilibria.

**A real empirical finding, not swept under the rug**: the roadmap's
literal closed form (fixed `eta`, `alpha0` annealed via `sqrt(node_
visits)`) does not converge on this fixture -- confirmed by direct
measurement across dozens of hyperparameter combinations, collapsing to
a pure vertex or converging cleanly to a biased point depending on the
regime. `g` here is a single-sample realization (`u_row`/`u_col` against
the opponent's one sampled program, not an expectation over its full
mixed strategy), and a Robbins-Monro-style decaying `eta_t = eta /
node_visits^0.6` -- the same step-size shape `search.py`'s own H4 fix
already uses for an analogous noisy-update problem -- is what actually
converges. See `row_sketch._mmd_update`'s own docstring for the full
account."""

from __future__ import annotations

import random
import statistics

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
from simult_chess.learn.row_sketch import _mmd_update, make_root, run_simulations
from simult_chess.rules.ruleset import RuleSet

RULESET = RuleSet(n_actions=1)


# --- _mmd_update, in isolation ----------------------------------------------


def test_mmd_update_returns_a_valid_probability_distribution() -> None:
    x = {0: 0.5, 1: 0.3, 2: 0.2}
    mu = {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}
    g = {0: 1.0, 1: 0.0, 2: -1.0}
    new_x, new_mu = _mmd_update(
        x, mu, g, node_visits=1, eta=1.0, alpha0=0.5, magnet_period=50
    )
    assert sum(new_x.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(p >= 0.0 for p in new_x.values())
    assert sum(new_mu.values()) == pytest.approx(1.0, abs=1e-9)


def test_mmd_update_moves_mass_toward_the_higher_payoff_action() -> None:
    x = {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}
    mu = {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}
    g = {0: 1.0, 1: 0.0, 2: -1.0}
    new_x, _ = _mmd_update(
        x, mu, g, node_visits=1, eta=1.0, alpha0=0.5, magnet_period=50
    )
    assert new_x[0] > new_x[1] > new_x[2]


def test_mmd_update_resets_the_magnet_only_on_the_scheduled_visit() -> None:
    x = {0: 0.6, 1: 0.4}
    mu = {0: 0.5, 1: 0.5}
    g = {0: 1.0, 1: 0.0}
    _, mu_unchanged = _mmd_update(
        x, mu, g, node_visits=3, eta=1.0, alpha0=0.5, magnet_period=5
    )
    assert mu_unchanged == mu
    new_x, mu_reset = _mmd_update(
        x, mu, g, node_visits=5, eta=1.0, alpha0=0.5, magnet_period=5
    )
    assert mu_reset == new_x
    assert mu_reset != mu


# --- Fixture convergence (identical construction to test_row_sketch_
# matching_pennies.py -- not re-derived) -------------------------------------

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


def test_mmd_last_iterate_converges_to_the_known_uniform_mixing() -> None:
    # RM+/Hedge's own equivalent test (test_row_sketch_matching_pennies.
    # py) converges within a single 128-simulation run; MMD's last
    # iterate has real per-run variance at comparable budgets (the
    # bias-variance tradeoff documented in _mmd_update's own docstring),
    # so this averages across seeds, matching the RPS/unequal-support
    # tests' own established pattern for a noisy statistic -- not a
    # weaker claim about the *equilibrium*, just about how many
    # independent runs it takes to see it clearly.
    evaluator = _UniformPriorEvaluator()
    white_flee = []
    black_aim = []
    for seed in range(20):
        root = make_root(_dodge_state())
        run_simulations(
            root, RULESET, evaluator, 3000, random.Random(seed),
            max_depth=1, pool_size=16, pool_seed_size=16, selection="mmd",
        )
        assert root.white is not None and root.black is not None
        x_white = root.white.average_strategy()
        x_black = root.black.average_strategy()
        flee_index = next(
            i for i, p in enumerate(root.white.programs) if p[0].token == _WHITE_KING
        )
        aim_indices = [
            i
            for i, p in enumerate(root.black.programs)
            if p[0].token == _KNIGHT
            and p[0].trajectory.path[-1] in (Square(0, 1), Square(1, 0))
        ]
        white_flee.append(x_white[flee_index])
        black_aim.append(sum(x_black[i] for i in aim_indices))

    assert statistics.mean(white_flee) == pytest.approx(0.5, abs=0.1)
    # Black's two threatening jumps should absorb essentially all mass --
    # a much sharper claim than White's, and one the earlier scratch
    # exploration found MMD gets right consistently (mean 1.0, stdev 0).
    assert statistics.mean(black_aim) > 0.9
