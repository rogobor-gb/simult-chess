"""v3 18d.2 (foundational fix): `nashconv.row_sketch_nashconv_exact`, cross-
validated against `solve_exact` on the matching_pennies_dodge fixture.
`restricted_support_nashconv_exact` cannot score a `row_sketch` policy at
all (it unconditionally drives `learn.search`'s own tree) -- this is the
function 18d.2's arm comparison actually needs.
"""

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
from simult_chess.learn import row_sketch
from simult_chess.learn.action_grid import NO_SECOND_INDEX, slot2_legal_actions
from simult_chess.learn.nashconv import row_sketch_nashconv_exact
from simult_chess.rules.ruleset import RuleSet
from simult_chess.solver.exact import solve_exact

RULESET = RuleSet(n_actions=1)
_NO_CASTLING = CastlingRights(
    white_kingside=False,
    white_queenside=False,
    black_kingside=False,
    black_queenside=False,
)

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
            castling_rights=_NO_CASTLING,
            repetition_ledger={},
            no_progress_counter=0,
            phase_index=0,
        ),
    )


class _UniformPriorEvaluator:
    """Same construction `test_row_sketch_mmd.py` uses: an empty prior makes
    `seed_program_pool` fall back to uniform-over-every-legal-action, so the
    seeded pool covers this fixture's full (small) legal action set
    regardless of a real, untrained network's arbitrary prior -- isolating
    the selection rule's own convergence from prior-seeding luck."""

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


@pytest.mark.parametrize(
    "selection", ["regret_matching", "rm_plus", "hedge", "optimistic_hedge", "mmd"]
)
def test_row_sketch_nashconv_exact_is_small_when_the_pool_covers_every_legal_program(
    selection: str,
) -> None:
    exact_value = solve_exact(
        _dodge_state(), RULESET, max_nodes=5000, max_depth=1
    ).root.value
    evaluator = _UniformPriorEvaluator()
    rng = random.Random(0)
    root = row_sketch.make_root(_dodge_state())
    row_sketch.run_simulations(
        root,
        RULESET,
        evaluator,
        400,
        rng,
        epsilon=0.02,
        selection=selection,
        pool_size=16,
        pool_seed_size=16,
    )
    result = row_sketch_nashconv_exact(root, RULESET, "matching_pennies_dodge")

    # solved_value is an independent consistency check: row_sketch_nashconv_
    # exact calls solve_exact itself, so this must match the value computed
    # directly above regardless of selection.
    assert result.solved_value == pytest.approx(exact_value, abs=1e-9)
    assert result.actual_value == pytest.approx(exact_value, abs=0.35)
    assert result.nashconv < 0.6


def test_row_sketch_nashconv_exact_is_large_when_the_pool_misses_legal_programs() -> (
    None
):
    # A deliberately tiny pool (fewer entries than the fixture's 11 legal
    # black programs) can't represent the equilibrium -- best response
    # searched over the *full* exhaustive pool should find real, large
    # exploitability, not silently report near-zero NashConv.
    evaluator = _UniformPriorEvaluator()
    rng = random.Random(0)
    root = row_sketch.make_root(_dodge_state())
    row_sketch.run_simulations(
        root,
        RULESET,
        evaluator,
        200,
        rng,
        epsilon=0.02,
        selection="regret_matching",
        pool_size=2,
        pool_seed_size=2,
    )
    result = row_sketch_nashconv_exact(root, RULESET, "matching_pennies_dodge")
    assert result.nashconv > 0.45
