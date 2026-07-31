"""v3 18b: `solver.exact.solve_exact`, cross-validated against the project's
own already-established ground truth rather than trusted on its own.

**Tractability, measured not assumed** (this session's own scoping pass for
18b): even the *proven*, already-shipped Matching-Pennies fixture
(`test_search_matching_pennies.py`'s `_dodge_state`) explodes past
thousands of reachable states once its two "ongoing" branches are allowed
to run to real completion instead of being depth-truncated -- board
mobility on the real 8x8 board grows too fast. `solve_exact` therefore
supports `max_depth` (any state still `"ongoing"` at the limit becomes a
0.0-valued leaf, `learn.selfplay.ReplayBuffer`'s own established
"undecided -> neutral" convention, not a new one). The primary test below
reuses that exact position at `max_depth=1` and checks `solve_exact`
reproduces the value the *search*-based convergence test in `test_search_
matching_pennies.py` already established empirically (-0.5) -- a strong
cross-check, since the two are entirely independent code paths (full
backward induction vs. regret-matching search) converging on the same
known answer.
"""

from __future__ import annotations

import pytest

from simult_chess.core.types import (
    Bookkeeping,
    CastlingRights,
    Color,
    Square,
    State,
    Token,
)
from simult_chess.invariants.harness import run_phase
from simult_chess.rules.ruleset import RuleSet
from simult_chess.solver.exact import IntractableGraphError, solve_exact

RULESET = RuleSet(n_actions=1)

_WHITE_KING = Token(id=1, color=Color.WHITE, typ="k")
_BLACK_KING = Token(id=2, color=Color.BLACK, typ="k")
_BLOCKER_A1 = Token(id=10, color=Color.WHITE, typ="p")
_BLOCKER_A3 = Token(id=11, color=Color.WHITE, typ="p")
_BLOCKER_B2 = Token(id=12, color=Color.WHITE, typ="p")
_BLOCKER_B3 = Token(id=13, color=Color.WHITE, typ="p")
_KNIGHT = Token(id=4, color=Color.BLACK, typ="n")


def _dodge_state() -> State:
    """Identical construction to `test_search_matching_pennies.py`'s own
    `_dodge_state` -- deliberately not re-derived, to keep the two tests
    testing the *same* position."""
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


def test_matches_the_search_based_convergence_tests_known_value() -> None:
    """`test_search_matching_pennies.py::test_root_value_converges_to_minus_
    one_half_for_the_defender` establishes -0.5 empirically (6000
    simulations, `abs=0.07` tolerance). `solve_exact` should reproduce it
    *exactly* (backward induction + LP, no sampling noise)."""
    graph = solve_exact(_dodge_state(), RULESET, max_nodes=5000, max_depth=1)
    assert graph.root.value == pytest.approx(-0.5, abs=1e-9)


def test_root_has_the_expected_action_counts() -> None:
    """Sanity check on the graph structure itself, independent of the
    value: White's pool is `{flee, 3 wall-pawn incidentals}` = 4; Black's
    is `{aim_x1, aim_x2, 6 dominated knight jumps, 3 king moves}` -- wait,
    `test_search_matching_pennies.py` documents only "Black's other legal
    knight jumps and its king's moves" without asserting a fixed count, so
    this only checks the two counts actually measured when this test was
    written, not re-derived from the spec independently."""
    graph = solve_exact(_dodge_state(), RULESET, max_nodes=5000, max_depth=1)
    assert len(graph.root.white_programs) == 4
    assert len(graph.root.black_programs) == 11


def test_raises_intractable_without_a_depth_limit() -> None:
    """The documented tractability finding, pinned as a regression test:
    this exact (proven-small-in-every-other-sense) fixture is NOT solvable
    to genuine completion without `max_depth` -- if this ever stops
    raising, `enumerate_legal_programs`'s cost model or this position's
    branching changed enough to be worth knowing about."""
    with pytest.raises(IntractableGraphError):
        solve_exact(_dodge_state(), RULESET, max_nodes=5000, max_depth=None)


def test_depth_limited_leaves_are_flagged_not_silently_treated_as_exact() -> None:
    graph = solve_exact(_dodge_state(), RULESET, max_nodes=5000, max_depth=1)
    depth_limited = [n for n in graph.nodes.values() if n.is_depth_limited]
    genuine_terminal = [
        n for n in graph.nodes.values() if n.is_terminal and not n.is_depth_limited
    ]
    # Both kinds of leaf actually occur in this fixture -- if either count
    # were 0 this test wouldn't be exercising what it claims to.
    assert len(depth_limited) > 0
    assert len(genuine_terminal) > 0


def test_every_root_program_pair_replays_clean_through_the_invariant_harness() -> None:
    """v3 18b DoD: 'every solved position replays through the invariant
    harness with zero violations at any severity.' Every root-level
    (white, black) program pair `solve_exact` actually fed to `phi` while
    building the graph -- not just the ones with equilibrium mass -- is
    replayed through `invariants.harness.run_phase` in strict mode, which
    raises on any WF/L/R/T violation."""
    graph = solve_exact(_dodge_state(), RULESET, max_nodes=5000, max_depth=1)
    root = graph.root
    state = root.state
    for program_white in root.white_programs:
        for program_black in root.black_programs:
            result = run_phase(
                state, program_white, program_black, RULESET, mode="strict"
            )
            assert result.violations == ()
