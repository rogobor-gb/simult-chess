"""v3 18b DoD: the depth profile "reported with distributions, not just
means" -- which needs more than one fixture to mean anything.
`test_solver_depth.py` computed the profile on exactly one position
(Matching-Pennies dodge); this file adds two more small, hand-constructed
positions (the roadmap's own "two-to-four-phase constructed positions"
category, same wall-boxing idiom as the existing fixtures), genuinely
different in shape, not palette-swaps of the same dilemma:

- **`dominant_strategy_contrast`**: the *same* King-boxed-to-one-escape
  setup as the Matching-Pennies dodge fixture, but the attacking knight is
  repositioned (`d2` instead of `c3`) so it threatens only the flee square,
  never the stay square -- confirmed by construction (`d2 -> a2` is not a
  valid knight offset). White's dominant strategy is simply "always stay";
  no mixing is needed. This is the contrasting, *boring* end of the depth
  spectrum: `gamma = mu = 0.0` exactly, deliberately unlike the flagship
  fixture's `gamma = 1.0`.
- **`three_way_dodge`**: the King has *two* flee squares (only three wall
  pawns instead of four), and Black has *two* knights, each covering a
  disjoint subset of the King's three options (one knight covers stay-or-
  flee-to-b1, exactly like the flagship fixture; the second knight covers
  the second flee square, `b3`, and nothing else, confirmed by construction
  -- `d4 -> a2`/`d4 -> b1` are not valid knight offsets). A genuinely
  richer dilemma than the 2-choice flagship: `value = -1/3` (distinct from
  -1/2), `gamma = 1.0` (still maximal -- simultaneity matters just as much
  with a third option in play), `mu = 2/3` (White benefits *more* from
  randomising here than in the 2-choice fixture).

Both are solved by `solve_exact` the same way as the flagship fixture
(`RuleSet(n_actions=1)`, `max_depth=1`) -- internally consistent with the
already-validated methodology, though unlike the flagship fixture neither
has an independent search-based cross-check (that pre-existed for
Matching-Pennies specifically; building one for each new fixture here was
judged not worth the added session cost given `solve_exact` is already
cross-validated once)."""

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
from simult_chess.solver.depth import compute_depth_profile
from simult_chess.solver.exact import solve_exact

RULESET = RuleSet(n_actions=1)
_NO_CASTLING = CastlingRights(
    white_kingside=False,
    white_queenside=False,
    black_kingside=False,
    black_queenside=False,
)


def _sq(name: str) -> Square:
    return Square(ord(name[0]) - ord("a"), int(name[1]) - 1)


# --- dominant_strategy_contrast ---------------------------------------------

_C_WHITE_KING = Token(id=1, color=Color.WHITE, typ="k")
_C_BLOCKER_A1 = Token(id=10, color=Color.WHITE, typ="p")
_C_BLOCKER_A3 = Token(id=11, color=Color.WHITE, typ="p")
_C_BLOCKER_B2 = Token(id=12, color=Color.WHITE, typ="p")
_C_BLOCKER_B3 = Token(id=13, color=Color.WHITE, typ="p")
_C_BLACK_KING = Token(id=2, color=Color.BLACK, typ="k")
_C_KNIGHT = Token(id=4, color=Color.BLACK, typ="n")


def _contrast_state() -> State:
    return State(
        board={
            _C_WHITE_KING: _sq("a2"),
            _C_BLOCKER_A1: _sq("a1"),
            _C_BLOCKER_A3: _sq("a3"),
            _C_BLOCKER_B2: _sq("b2"),
            _C_BLOCKER_B3: _sq("b3"),
            _C_BLACK_KING: _sq("h8"),
            _C_KNIGHT: _sq("d2"),
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


def test_knight_at_d2_cannot_reach_a2_confirming_the_dominant_strategy_setup() -> (
    None
):
    # d2 -> a2 is offset (-3, 1) -- not a knight move ({1,2}/{2,1}) -- the
    # precondition the whole "White always stays" claim rests on.
    origin = _sq("d2")
    target = _sq("a2")
    assert (abs(target.file - origin.file), abs(target.rank - origin.rank)) not in {
        (1, 2),
        (2, 1),
    }


def test_dominant_strategy_contrast_has_zero_gamma_and_mu() -> None:
    graph = solve_exact(_contrast_state(), RULESET, max_nodes=20_000, max_depth=1)
    root = graph.root
    assert root.stage_matrix is not None
    profile = compute_depth_profile(
        "dominant_strategy_contrast",
        root.stage_matrix,
        root.white_programs,
        root.black_programs,
    )
    assert profile.value == pytest.approx(0.0, abs=1e-9)
    assert profile.gamma == pytest.approx(0.0, abs=1e-9)
    assert profile.mu == pytest.approx(0.0, abs=1e-9)


# --- three_way_dodge ---------------------------------------------------------

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
            _T_WHITE_KING: _sq("a2"),
            _T_BLOCKER_A1: _sq("a1"),
            _T_BLOCKER_A3: _sq("a3"),
            _T_BLOCKER_B2: _sq("b2"),
            _T_BLACK_KING: _sq("h8"),
            _T_KNIGHT_1: _sq("c3"),
            _T_KNIGHT_2: _sq("d4"),
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


def test_second_knight_covers_only_the_second_flee_square() -> None:
    # d4 -> a2 is (-3, -2) and d4 -> b1 is (-2, -3) -- neither is a knight
    # offset, so the second knight's coverage is disjoint from the first's
    # (c3, unchanged from the flagship fixture, still covers a2 and b1).
    origin = _sq("d4")
    for target_name in ("a2", "b1"):
        target = _sq(target_name)
        offset = (abs(target.file - origin.file), abs(target.rank - origin.rank))
        assert offset not in {(1, 2), (2, 1)}


def test_three_way_dodge_differs_from_the_two_way_flagship() -> None:
    graph = solve_exact(_three_way_state(), RULESET, max_nodes=20_000, max_depth=1)
    root = graph.root
    assert root.stage_matrix is not None
    profile = compute_depth_profile(
        "three_way_dodge", root.stage_matrix, root.white_programs, root.black_programs
    )
    # A distinct value from the flagship fixture's -0.5 -- confirms this is
    # a genuinely different position, not a relabelled copy.
    assert profile.value != pytest.approx(-0.5, abs=1e-9)
    assert profile.value == pytest.approx(-1.0 / 3.0, abs=1e-9)
    # Simultaneity still matters maximally with a third option in play.
    assert profile.gamma == 1.0
    # White benefits *more* from randomising here than in the 2-choice
    # fixture (mu=0.5 there) -- richer, not just different.
    assert profile.mu > 0.5
