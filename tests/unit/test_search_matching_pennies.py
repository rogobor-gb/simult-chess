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

import pytest

from simult_chess.core import legality
from simult_chess.core.phi import phi
from simult_chess.core.types import (
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
from simult_chess.learn.search import Evaluator, make_root, run_simulations
from simult_chess.rules.ruleset import RuleSet

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
