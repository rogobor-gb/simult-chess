"""The correctness cornerstone of SM-MCTS (Phase 13b, docs/LEARNING_DESIGN.md
§1.2/§2.2): the spec's own Matching-Pennies king-dodge subgame (spec §8, line
270) has **no pure equilibrium** -- a max-operator (AlphaZero-style) backup
would collapse it to a pure choice; the design's whole case for regret-
matching SM-MCTS over decoupled UCT rests on this example. This test proves
the search's average strategy converges to the unique equilibrium: **uniform
mixing on both sides, value -1/2 to the defending side** (spec: "unique value
1/2, optimal play uniform mixing", stated from the attacker's win-probability
side; from the defender/White's utility side the value is -1/2, symmetric,
since either pure choice is punished with probability 1/2 under optimal
attacker play).

Construction (verified against the live engine below, not just derived by
hand): White king at a2 is walled in by four pawns (a1, a3, b2, b3) so its
*only* legal move is to b1 -- the wall is what makes this a genuine 2-choice
dilemma rather than "always flee to any of 7 safe squares" (an earlier,
broken draft of this test discovered that mistake: an unwalled king has many
strictly-safer escapes that dominate the intended dilemma entirely). A black
knight at c3 can jump to a2 ("aim x1", capturing the king iff it stayed) or
to b1 ("aim x2", meeting a fleeing king there -- a (V) vertex conflict, still
removing the king). The wall pawns' own short-range incidental moves (a3-a4,
b3-b4, and b2xc3 -- a "capture" of the attacking knight that always whiffs,
since the knight itself moves away the same phase, the vacated-square rule,
R6) give White additional ways to satisfy L2's mandatory-displacement clause
without touching the king; the knight's other 6 jumps and the black king's
own moves give Black additional legal-but-irrelevant alternatives. (A second
broken draft used a dedicated long-range filler rook for this instead of the
wall pawns -- its full-board slide happened to reach the black king's own
square, so playing it while Black played aim_x1 triggered *mutual* regicide,
a "draw" [value 0] instead of the intended "black_wins" [value -1] -- an
accidental extra escape route, same failure shape as the first draft, just
one square further out. The wall pawns' one-to-two-square range cannot reach
anything relevant, which is why they replace it.) None of this clutter is
filtered out of the search -- real self-play sees the same kind on any real
board.

Consequently the test asserts convergence in **aggregate on White's side**,
not per index: the three non-king single-action programs above are *value-
degenerate* with "stay" (all leave the king at a2, so all share the same
expected value under any fixed opponent distribution, confirmed below);
only the king's own move is "flee". Sum of the stay-equivalent arms'
probability mass should converge to ~0.5, against the flee arm's ~0.5, and
Black's two threatening knight jumps should each converge to ~0.5 while
every other legal action (Black's other 6 jumps, its king's moves) is
*dominated* (never as good as threatening) and should vanish.

A miss leaves the position "ongoing" (not a genuine T1 terminal), so the test
runs the search with `max_depth=1`: the resulting phase is always evaluated
as an immediate depth-limited leaf at value 0, isolating exactly the one-shot
matrix game spec §8 describes -- real self-play (Stage D) never bounds depth
this way.
"""

from __future__ import annotations

import random
import statistics

import numpy as np
import pytest

from simult_chess.core import legality
from simult_chess.core.phi import phi
from simult_chess.core.types import (
    Action,
    Bookkeeping,
    CastlingRights,
    Color,
    Move,
    Program,
    Square,
    State,
    Token,
    Trajectory,
)
from simult_chess.learn.action_grid import NO_SECOND_INDEX, slot2_legal_actions
from simult_chess.learn.config import SearchConfig
from simult_chess.learn.search import Evaluator, make_root, run_simulations
from simult_chess.rules.ruleset import RuleSet
from simult_chess.solver.lp import solve_zero_sum

RULESET = RuleSet(n_actions=1)

_WHITE_KING = Token(id=1, color=Color.WHITE, typ="k")
_BLACK_KING = Token(id=2, color=Color.BLACK, typ="k")
_BLOCKER_A1 = Token(id=10, color=Color.WHITE, typ="p")
_BLOCKER_A3 = Token(id=11, color=Color.WHITE, typ="p")
_BLOCKER_B2 = Token(id=12, color=Color.WHITE, typ="p")
_BLOCKER_B3 = Token(id=13, color=Color.WHITE, typ="p")
_KNIGHT = Token(id=4, color=Color.BLACK, typ="n")

_X1 = Square(0, 1)  # a2: the king's walled-in starting square ("aim x1")
_X2 = Square(1, 0)  # b1: the king's one legal flee square ("aim x2")
_KNIGHT_SQUARE = Square(2, 2)  # c3


def _dodge_state() -> State:
    return State(
        board={
            _WHITE_KING: _X1,
            _BLACK_KING: Square(7, 7),  # h8 -- out of range of every wall pawn
            _BLOCKER_A1: Square(0, 0),
            _BLOCKER_A3: Square(0, 2),
            _BLOCKER_B2: Square(1, 1),
            _BLOCKER_B3: Square(1, 2),
            _KNIGHT: _KNIGHT_SQUARE,
        },
        cooldown=frozenset(),
        reservations_white=(),
        reservations_black=(),
        bookkeeping=Bookkeeping(
            castling_rights=CastlingRights(
                white_kingside=False,
                white_queenside=False,
                black_kingside=False,
                black_queenside=False,
            ),
            repetition_ledger={},
            no_progress_counter=0,
            phase_index=0,
        ),
    )


_KING_FLEE = Move(token=_WHITE_KING, trajectory=Trajectory(path=(_X1, _X2)))
# One of the wall's own short-range pushes -- the "stay" representative used
# by the engine-verification test below. `slot1_legal_actions` (used by the
# convergence test) discovers this and the wall's other two incidental moves
# on its own; all three are value-degenerate with "stay" (see module
# docstring).
_STAY_REPRESENTATIVE = Move(
    token=_BLOCKER_A3, trajectory=Trajectory(path=(Square(0, 2), Square(0, 3)))
)
_KNIGHT_AIM_X1 = Move(
    token=_KNIGHT, trajectory=Trajectory(path=(_KNIGHT_SQUARE, _X1), is_jump=True)
)
_KNIGHT_AIM_X2 = Move(
    token=_KNIGHT, trajectory=Trajectory(path=(_KNIGHT_SQUARE, _X2), is_jump=True)
)


def test_the_engine_actually_implements_matching_pennies_here() -> None:
    """Ground truth: verify the hand-built position's payoff matrix against
    the live `phi`, not just against a derivation -- this is the contract the
    convergence test below relies on."""
    state = _dodge_state()
    expected = {
        ("stay", "aim_x1"): "black_wins",  # king captured in place
        ("stay", "aim_x2"): "ongoing",  # knight lands on an empty square
        ("flee", "aim_x1"): "ongoing",  # king vacated; knight lands on empty
        ("flee", "aim_x2"): "black_wins",  # (V) conflict at b1: king removed
    }
    programs: dict[str, Program] = {
        "stay": (_STAY_REPRESENTATIVE,),
        "flee": (_KING_FLEE,),
        "aim_x1": (_KNIGHT_AIM_X1,),
        "aim_x2": (_KNIGHT_AIM_X2,),
    }
    for (white_choice, black_choice), outcome in expected.items():
        assert legality.is_legal_program(
            state, programs[white_choice], Color.WHITE, RULESET
        )
        assert legality.is_legal_program(
            state, programs[black_choice], Color.BLACK, RULESET
        )
        result = phi(state, programs[white_choice], programs[black_choice], RULESET)
        assert result.outcome == outcome, (white_choice, black_choice, result.outcome)


def test_the_wall_leaves_the_king_exactly_one_legal_move() -> None:
    # The property that makes this a genuine 2-choice dilemma, not the
    # "many safe escapes" mistake an earlier draft of this test made.
    from simult_chess.learn.action_grid import slot1_legal_actions

    state = _dodge_state()
    white_actions = slot1_legal_actions(state, Color.WHITE, RULESET)
    king_moves = [a for a in white_actions.values() if a == _KING_FLEE]
    other_king_moves = [
        a
        for a in white_actions.values()
        if isinstance(a, Move) and a.token == _WHITE_KING and a != _KING_FLEE
    ]
    assert king_moves == [_KING_FLEE]
    assert other_king_moves == []


class _UniformPriorEvaluator:
    """No informative prior (uniform over the legal set), value 0 for any
    non-terminal leaf. Isolates the regret-matching machinery itself: the
    equilibrium must be found from the payoff structure alone, not smuggled
    in via a prior."""

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


def test_average_strategy_converges_to_uniform_mixing_both_sides() -> None:
    evaluator: Evaluator = _UniformPriorEvaluator()
    state = _dodge_state()
    root = make_root(state)
    rng = random.Random(0)
    run_simulations(
        root, RULESET, evaluator, 6000, rng, prior_weight=1.0, max_depth=1
    )

    assert root.white is not None and root.black is not None
    white_strategy = root.white.average_strategy()
    black_strategy = root.black.average_strategy()

    flee_index = next(i for i, a in root.white.actions.items() if a == _KING_FLEE)
    stay_mass = sum(p for i, p in white_strategy.items() if i != flee_index)
    flee_mass = white_strategy[flee_index]
    assert stay_mass == pytest.approx(0.5, abs=0.07)
    assert flee_mass == pytest.approx(0.5, abs=0.07)

    aim_x1_index = next(
        i for i, a in root.black.actions.items() if a == _KNIGHT_AIM_X1
    )
    aim_x2_index = next(
        i for i, a in root.black.actions.items() if a == _KNIGHT_AIM_X2
    )
    assert black_strategy[aim_x1_index] == pytest.approx(0.5, abs=0.07)
    assert black_strategy[aim_x2_index] == pytest.approx(0.5, abs=0.07)
    # Black's other legal knight jumps and its king's moves are dominated
    # (never threaten the walled king) -- they should carry little mass.
    other_black_mass = 1.0 - black_strategy[aim_x1_index] - black_strategy[aim_x2_index]
    assert other_black_mass < 0.1


def test_root_value_converges_to_minus_one_half_for_the_defender() -> None:
    # The unique game value (spec §8, line 280): under optimal (uniform) play
    # on both sides, the defender is caught with probability 1/2, so White's
    # expected value is -1/2 -- not 0 (an "always escape" reading a broken
    # construction would give) and not -1 (a pure-strategy collapse a
    # max-operator backup would produce).
    evaluator: Evaluator = _UniformPriorEvaluator()
    state = _dodge_state()
    root = make_root(state)
    rng = random.Random(1)
    run_simulations(
        root, RULESET, evaluator, 6000, rng, prior_weight=1.0, max_depth=1
    )
    assert root.white is not None
    mean_q = sum(
        root.white.q[a] * root.white.visits[a] for a in root.white.actions
    ) / sum(root.white.visits.values())
    assert mean_q == pytest.approx(-0.5, abs=0.07)


# ---------------------------------------------------------------------------
# Phase 18a'.1 (docs/LEARNING_ROADMAP_v3.md): the asymmetric ground-truth
# fixture. Matching Pennies above is *symmetric* -- regret matching, RM+,
# Hedge and expected-value substitution are all equivariant under relabelling
# of actions, so on a symmetric fixture the equilibrium is a fixed point of
# the search dynamics whether or not the in-tree regret estimator is biased
# (v3 Cor. 8.7). The tests below use fixtures whose equilibria are
# non-uniform and whose relabelling-symmetry group is trivial, so a biased
# estimator (H4: `_update`'s `stats.q[a]` baseline is a stale running mean,
# not a fresh per-simulation counterfactual) or a prior-contaminated policy
# average (H5: `_simulate` accumulates the prior-blended `sigma`, not pure
# `_regret_matching_strategy`'s RM output, into `strategy_sum`) has somewhere
# to visibly disagree with the known answer.
#
# Construction, mirroring the Matching-Pennies fixture's own idiom (a small
# number of hand-placed pieces, verified against the live engine rather than
# only derived by hand): a knight in a board corner has a small, exactly
# enumerable jump count for free (a1 -> {b3, c2}, a2 -> {b4, c3, c1}, and by
# symmetry h7 -> {g5, f6, f8}), so one corner knight per colour *is* the
# restricted action pool -- no filler pieces are needed to cap it. Each
# fixture's king (and, for White, the knight's own two idle jump-adjacent
# squares are not used at all) is free to move because any legal action that
# leaves the knight at its starting square is deliberately mapped to the same
# payoff-matrix row/column as the knight's own first destination -- the exact
# aggregation idiom the Matching-Pennies test already uses for its
# value-degenerate "stay" arms (module docstring above), not a new mechanism.
# Verified directly (see the scratch check this test's construction was
# validated against): both fixtures below give White/Black exactly the
# intended knight-jump action count plus only idle king moves, and every
# (white_program, black_program) pair resolves to `outcome == "ongoing"`
# (the pieces never interact), so every leaf value in these tests comes from
# the synthetic evaluator's matrix lookup, never from `phi`'s own win/loss
# machinery.


_ASYM_WHITE_KING = Token(id=101, color=Color.WHITE, typ="k")
_ASYM_BLACK_KING = Token(id=102, color=Color.BLACK, typ="k")
_ASYM_WHITE_KNIGHT = Token(id=103, color=Color.WHITE, typ="n")
_ASYM_BLACK_KNIGHT = Token(id=104, color=Color.BLACK, typ="n")

# Black's three corner jumps from h7 are shared by both fixtures below.
_BLACK_DEST_3 = {
    Square(6, 4): 0,  # g5
    Square(5, 5): 1,  # f6
    Square(5, 7): 2,  # f8
}


def _asym_state(
    white_king: Square,
    white_knight: Square,
    black_king: Square,
    black_knight: Square,
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
            castling_rights=CastlingRights(
                white_kingside=False,
                white_queenside=False,
                black_kingside=False,
                black_queenside=False,
            ),
            repetition_ledger={},
            no_progress_counter=0,
            phase_index=0,
        ),
    )


class _MatrixGameEvaluator:
    """Synthetic evaluator whose leaf value is a lookup into a fixed payoff
    matrix, keyed by which corner square each side's knight ended up on (see
    module note above): `matrix[row][col]` is White's utility, `row`/`col`
    coming from `white_dest`/`black_dest` (default 0 -- the knight's own
    "idle"/first-mapped bucket -- for any square not in the map, e.g. the
    knight not having moved because a king move was sampled instead).
    `scale` rescales the matrix into `Evaluator.evaluate_leaf`'s documented
    `[-1, 1]` range; a positive rescaling leaves the equilibrium unchanged
    and only scales the value."""

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
    strategy: dict[int, float],
    actions: dict[int, Action],
    knight: Token,
    dest: dict[Square, int],
    n_categories: int,
) -> list[float]:
    """Collapse a search-derived slot-1 average strategy (indexed by grid
    index) into per-matrix-row/column mass, using the *same* `dest` map the
    matching `_MatrixGameEvaluator` used to compute leaf values -- so an idle
    king move and the knight's own first-mapped destination are aggregated
    consistently between what the search actually evaluated and what this
    test measures, exactly mirroring the Matching-Pennies test's own
    aggregate "stay" mass."""
    out = [0.0] * n_categories
    for index, probability in strategy.items():
        action = actions[index]
        category = 0
        if isinstance(action, Move) and action.token == knight:
            category = dest.get(action.trajectory.path[-1], 0)
        out[category] += probability
    return out


def _total_variation(p: list[float], q: list[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q, strict=True))


# --- Fixture 1: weighted rock-paper-scissors (v3 §1, 18a'.1) ---------------
# U = [[0,-1,2],[1,0,-1],[-2,1,0]], antisymmetric (M5's val=0 anchor still
# applies), x* = y* = (1/4, 1/2, 1/4). Rescaled by 1/2 to fit evaluate_leaf's
# documented [-1, 1] range; equilibrium strategies are scale-invariant.

_RPS_MATRIX = [[0.0, -1.0, 2.0], [1.0, 0.0, -1.0], [-2.0, 1.0, 0.0]]
_RPS_EQUILIBRIUM = [0.25, 0.5, 0.25]
_RPS_VALUE = 0.0
_RPS_SCALE = 0.5

_RPS_WHITE_DEST = {
    Square(1, 3): 0,  # a2 -> b4
    Square(2, 2): 1,  # a2 -> c3
    Square(2, 0): 2,  # a2 -> c1
}


def _rps_state() -> State:
    return _asym_state(
        white_king=Square(0, 0),  # a1
        white_knight=Square(0, 1),  # a2: jumps to b4, c3, c1
        black_king=Square(7, 7),  # h8
        black_knight=Square(7, 6),  # h7: jumps to g5, f6, f8
    )


def test_weighted_rps_ground_truth_matches_the_roadmap_matrix() -> None:
    """Certify the fixture's own claimed equilibrium against the project's
    exact LP solver (`solver.lp.solve_zero_sum`), independent of the search
    -- this is what the search-convergence test below is actually checked
    against, not a hand-derivation."""
    solution = solve_zero_sum(np.array(_RPS_MATRIX))
    assert solution.value == pytest.approx(_RPS_VALUE, abs=1e-9)
    assert list(solution.row_strategy) == pytest.approx(_RPS_EQUILIBRIUM, abs=1e-6)
    assert list(solution.col_strategy) == pytest.approx(_RPS_EQUILIBRIUM, abs=1e-6)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "v3 H4/H5: H5 (prior-contaminated reported policy) is fixed "
        "(18a'.3); H4 (in-tree regret estimator bias) is IMPROVED, not "
        "resolved (18a'.2, see search.py's module docstring for the full "
        "account of what was tried, including a reverted attempt that was "
        "found to break Matching Pennies). At the LIGHT-profile production "
        "simulation budget (M=128), mean TV distance is now ~0.24-0.27 "
        "(down from ~0.32-0.39 pre-fix) vs this test's 0.07 tolerance -- "
        "real progress, still failing. Remove this marker once H4 is fully "
        "resolved (roadmap's preferred (E3) explicit row evaluation, "
        "properly part of 18c, is the most likely path)."
    ),
)
def test_weighted_rps_average_strategy_converges_to_known_equilibrium() -> None:
    """v3 18a'.1's headline fixture, run at the actual LIGHT-profile
    self-play simulation budget (`SearchConfig().simulations`, = 128 -- the
    number every real game actually spends per move), not an arbitrarily
    large one. **Empirically confirmed to still fail at this budget** even
    after 18a'.2/18a'.3 (mean TV distance ~0.24-0.27, improved from
    ~0.32-0.39 pre-fix but still far outside this test's tolerance): earlier
    diagnostic sweeps (pre-18a'.2) over M in {128, 256, 512, ..., 6000,
    24000, 96000} showed the average strategy's distance to `x*` shrinking
    with more simulations rather than plateauing, consistent with H5's own
    quantification (v3 §0: 17.7% prior share at M=64, 13.5% at M=128) -- a
    slowly-vanishing-but-materially-large-at-the-production-budget artifact,
    not a fixed floor. Per the roadmap: *"This test should fail on the
    current estimator. If it passes, that is itself a finding and must be
    reported"* -- `strict=True` makes an unexpected future pass fail loudly
    rather than silently going stale."""
    evaluator = _MatrixGameEvaluator(
        _RPS_MATRIX, _RPS_WHITE_DEST, _BLACK_DEST_3, _RPS_SCALE
    )
    n_simulations = SearchConfig().simulations
    white_tv: list[float] = []
    black_tv: list[float] = []
    for seed in range(30):
        state = _rps_state()
        root = make_root(state)
        run_simulations(
            root, RULESET, evaluator, n_simulations, random.Random(seed),
            prior_weight=1.0, max_depth=1,
        )
        assert root.white is not None and root.black is not None
        white_mass = _aggregate_by_destination(
            root.white.average_strategy(), root.white.actions,
            _ASYM_WHITE_KNIGHT, _RPS_WHITE_DEST, 3,
        )
        black_mass = _aggregate_by_destination(
            root.black.average_strategy(), root.black.actions,
            _ASYM_BLACK_KNIGHT, _BLACK_DEST_3, 3,
        )
        white_tv.append(_total_variation(white_mass, _RPS_EQUILIBRIUM))
        black_tv.append(_total_variation(black_mass, _RPS_EQUILIBRIUM))

    assert statistics.mean(white_tv) < 0.07
    assert statistics.mean(black_tv) < 0.07


# --- Fixture 2: val != 0, unequal support sizes (v3 §1, 18a'.1) ------------
# 2x3 matrix (White has 2 actions, Black 3), so the two sides' support sizes
# differ structurally; Black's column 1 is dominated (0 mass at equilibrium)
# -- also exercises whether the estimator wastes mass on a dominated action.
# Ground truth from `solve_zero_sum`, not hand-derived (see the ground-truth
# test below). Rescaled by 1/3 (the matrix's max abs entry) to fit
# evaluate_leaf's [-1, 1] range.

_UNEQUAL_MATRIX = [[3.0, -1.0, -2.0], [-2.0, 2.0, 1.0]]
_UNEQUAL_SCALE = 1.0 / 3.0

_UNEQUAL_WHITE_DEST = {
    Square(1, 2): 0,  # a1 -> b3
    Square(2, 1): 1,  # a1 -> c2
}


def _unequal_support_state() -> State:
    return _asym_state(
        white_king=Square(7, 0),  # h1
        white_knight=Square(0, 0),  # a1: jumps to b3, c2 (only 2, by corner geometry)
        black_king=Square(7, 7),  # h8
        black_knight=Square(7, 6),  # h7: jumps to g5, f6, f8
    )


def test_unequal_support_ground_truth_via_exact_lp_solver() -> None:
    """This fixture's equilibrium is derived from the project's own exact
    solver, not hand-picked, precisely to avoid an arithmetic-derivation
    error masquerading as a search-convergence finding."""
    solution = solve_zero_sum(np.array(_UNEQUAL_MATRIX))
    assert solution.value < 0.0  # sanity: val != 0, unlike the RPS fixture
    assert 0.0 < solution.row_strategy[0] < 1.0  # White's support is size 2
    assert solution.col_strategy[1] == pytest.approx(0.0, abs=1e-6)  # dominated


@pytest.mark.xfail(
    strict=True,
    reason=(
        "v3 H4/H5, same status as the RPS fixture above -- H5 fixed, H4 "
        "improved but not resolved (see search.py's module docstring). At "
        "the LIGHT-profile production simulation budget (M=128), mean TV "
        "distance is now ~0.20 (White) / ~0.30 (Black), still far outside "
        "this test's 0.07 tolerance. Remove this marker once H4 is fully "
        "resolved."
    ),
)
def test_unequal_support_average_strategy_converges_to_known_equilibrium() -> None:
    """Companion to the RPS fixture above: `val != 0` and structurally
    unequal support sizes, run at the same LIGHT-profile simulation budget
    (`SearchConfig().simulations` = 128). **Empirically confirmed to still
    fail at this budget** (mean TV distance ~0.20-0.30) for the same
    reason."""
    solution = solve_zero_sum(np.array(_UNEQUAL_MATRIX))
    white_equilibrium = list(solution.row_strategy)
    black_equilibrium = list(solution.col_strategy)

    evaluator = _MatrixGameEvaluator(
        _UNEQUAL_MATRIX, _UNEQUAL_WHITE_DEST, _BLACK_DEST_3, _UNEQUAL_SCALE
    )
    n_simulations = SearchConfig().simulations
    white_tv: list[float] = []
    black_tv: list[float] = []
    for seed in range(30):
        state = _unequal_support_state()
        root = make_root(state)
        run_simulations(
            root, RULESET, evaluator, n_simulations, random.Random(seed),
            prior_weight=1.0, max_depth=1,
        )
        assert root.white is not None and root.black is not None
        white_mass = _aggregate_by_destination(
            root.white.average_strategy(), root.white.actions,
            _ASYM_WHITE_KNIGHT, _UNEQUAL_WHITE_DEST, 2,
        )
        black_mass = _aggregate_by_destination(
            root.black.average_strategy(), root.black.actions,
            _ASYM_BLACK_KNIGHT, _BLACK_DEST_3, 3,
        )
        white_tv.append(_total_variation(white_mass, white_equilibrium))
        black_tv.append(_total_variation(black_mass, black_equilibrium))

    assert statistics.mean(white_tv) < 0.07
    assert statistics.mean(black_tv) < 0.07
