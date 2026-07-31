"""v3 18b.4: the strategic-depth profile, computed on the one fixture this
session's tractability findings left well-validated (the Matching-Pennies
dodge position, cross-checked exactly against the search-based convergence
test's own -0.5 in `test_solver_exact.py`).

**Scope note, stated plainly rather than glossed over**: this session did
not manage to construct additional *named* endgame fixtures (K+R vs K,
K+P vs K, K+N+N vs K) with a meaningfully non-degenerate exact solution --
every attempted construction either exceeded `solve_exact`'s tractable
node count, or (once tightly enough boxed to solve) resolved almost
entirely via the `max_depth` fallback (>90% of leaves `is_depth_limited`,
i.e. mostly reporting "undecided", not a real forced result) rather than
genuine `phi` terminals. That is itself a reportable finding about how
hard tightly-bounded, genuinely-resolving small endgames are to construct
by hand on this board, not a result to force past with a low-quality
fixture. `phi` (intra-program coordination value) and `delta`/`rho_can`
(the reservation mechanic's shadow price) are consequently **not**
reported here -- this fixture has no Reserve/Cancel actions available at
all (`RuleSet(n_actions=1)`), so those functionals are structurally
inapplicable to it, not merely unmeasured.
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
from simult_chess.solver.depth import compute_depth_profile
from simult_chess.solver.exact import solve_exact

RULESET = RuleSet(n_actions=1)

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


def test_depth_profile_of_the_matching_pennies_dodge_fixture() -> None:
    graph = solve_exact(_dodge_state(), RULESET, max_nodes=5000, max_depth=1)
    root = graph.root
    assert root.stage_matrix is not None
    profile = compute_depth_profile(
        "matching_pennies_dodge",
        root.stage_matrix,
        root.white_programs,
        root.black_programs,
    )

    # Value: matches the search-based convergence test exactly (already
    # pinned in test_solver_exact.py; re-asserted here as a precondition
    # for the depth-profile numbers below to mean what they claim to).
    assert profile.value == pytest.approx(-0.5, abs=1e-9)

    # gamma > 0: simultaneity itself matters here -- this *is* the spec's
    # own flagship example for exactly that claim (a max-operator/
    # alternating-game backup would collapse this to a pure, wrong answer).
    assert profile.gamma > 0.0

    # mu > 0: White strictly benefits from being allowed to randomise
    # over a best deterministic guarantee.
    assert profile.mu > 0.0

    # No Reserve/Cancel actions exist under RuleSet(n_actions=1): both
    # usage rates are exactly 0, and phi is the *trivial* 0 (no second
    # slot to coordinate at all), not an unmeasured None.
    assert profile.rho_res_white == 0.0
    assert profile.rho_can_white == 0.0
    assert profile.rho_res_black == 0.0
    assert profile.rho_can_black == 0.0
    assert profile.phi == 0.0
    assert profile.delta is None  # no Reserve-deleted comparison supplied

    print(f"\ndepth profile: {profile}")
