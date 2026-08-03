"""v3 18b DoD: "the exact solver reproduces M5's proven val=0 on the
existing symmetric fixtures."

`tests/property/test_m5_symmetric_value.py` already proves this via
`solver.stage_matrix.build_stage_matrix` over a *restricted* support
(`solver.supports.enumerate_support`). This is the same claim checked a
different, stronger way: `solve_exact`'s own *exhaustive* program
enumeration and full backward induction, from scratch, with no shared
code path to the existing M5 test beyond `solver.lp.solve_zero_sum`
itself.

The argument this pins as a regression test: `_symmetric_midgame_fixture_
knight_pawn` (`test_m5_symmetric_value.py`'s own fixture) satisfies
`mirror_state(s) == s`, and both terminal payoffs (`chi` swaps colours,
so a `white_wins` leaf mirrors to a `black_wins` leaf and vice versa) and
the `max_depth` fallback (0.0, self-symmetric) are chi-antisymmetric by
construction -- so the *entire* reachable subgraph `solve_exact` builds
from this root is chi-symmetric, forcing `val = 0` exactly, at *any*
depth cutoff, not just as a property of one restricted support. Run at
`RuleSet(n_actions=1)` for tractability (this session's own measured
tractability ceiling for `RuleSet()`'s Reserve/Cancel combinatorics on a
6-piece position) -- a materially different setting from the existing M5
test's default `RuleSet()`, so this is a genuinely independent check of
the same antisymmetry claim, not a re-run of the same computation.
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
from simult_chess.rules.ruleset import RuleSet
from simult_chess.solver.exact import solve_exact

RULESET = RuleSet(n_actions=1)
_NO_CASTLING = CastlingRights(
    white_kingside=False,
    white_queenside=False,
    black_kingside=False,
    black_queenside=False,
)


def _symmetric_midgame_fixture_knight_pawn() -> State:
    # Identical construction to test_m5_symmetric_value.py's own fixture of
    # the same name -- deliberately not re-derived, so this is testing the
    # same position, not a lookalike.
    king_id, knight_id, pawn_id = 1, 2, 3
    board = {
        Token(id=king_id, color=Color.WHITE, typ="k"): Square(0, 0),
        Token(id=king_id, color=Color.BLACK, typ="k"): Square(0, 7),
        Token(id=knight_id, color=Color.WHITE, typ="n"): Square(3, 3),
        Token(id=knight_id, color=Color.BLACK, typ="n"): Square(3, 4),
        Token(id=pawn_id, color=Color.WHITE, typ="p"): Square(4, 1),
        Token(id=pawn_id, color=Color.BLACK, typ="p"): Square(4, 6),
    }
    return State(
        board=board,
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


def test_exact_value_is_zero_at_depth_zero() -> None:
    state = _symmetric_midgame_fixture_knight_pawn()
    graph = solve_exact(state, RULESET, max_nodes=50_000, max_depth=0)
    assert graph.root.value == pytest.approx(0.0, abs=1e-9)


def test_exact_value_is_zero_at_depth_one() -> None:
    state = _symmetric_midgame_fixture_knight_pawn()
    graph = solve_exact(state, RULESET, max_nodes=50_000, max_depth=1)
    assert graph.root.value == pytest.approx(0.0, abs=1e-9)


@pytest.mark.slow
def test_exact_value_is_zero_at_depth_two() -> None:
    # ~256,000 nodes, ~25s -- the antisymmetry argument holds at any depth,
    # so this is a strictly stronger check than depth 0/1, kept out of the
    # fast gate rather than trimmed away.
    state = _symmetric_midgame_fixture_knight_pawn()
    graph = solve_exact(state, RULESET, max_nodes=300_000, max_depth=2)
    assert graph.root.value == pytest.approx(0.0, abs=1e-9)
