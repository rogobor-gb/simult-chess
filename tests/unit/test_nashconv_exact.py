"""v3 18b.3: search-derived NashConv against a genuinely *exact* equilibrium
(not `build_stage_matrix`'s one-ply material surrogate, and not restricted to
the chi-symmetric val=0 fixtures `test_nashconv.py` already covers).

The Matching-Pennies dodge fixture (`test_solver_exact.py`'s `_dodge_state`)
is the one position this session's tractability findings left well cross-
validated: `solve_exact`'s root value (-0.5, `max_depth=1`) reproduces the
independent search-based convergence test's own empirically-established
value exactly. Its root `stage_matrix` and exhaustive program sets are
therefore a trustworthy exact ground truth to measure a learned agent's
search-derived NashConv against -- `restricted_support_nashconv_exact`
(`learn.nashconv`) takes them directly, with no chi-symmetry assumption
(this fixture isn't chi-symmetric: White is boxed, Black is not)."""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from simult_chess.core.types import (  # noqa: E402
    Bookkeeping,
    CastlingRights,
    Color,
    Square,
    State,
    Token,
)
from simult_chess.learn.config import NetConfig, SearchConfig  # noqa: E402
from simult_chess.learn.nashconv import restricted_support_nashconv_exact  # noqa: E402
from simult_chess.learn.net import SimultChessNet  # noqa: E402
from simult_chess.rules.ruleset import RuleSet  # noqa: E402
from simult_chess.solver.exact import solve_exact  # noqa: E402

RULESET = RuleSet(n_actions=1)
_CPU = torch.device("cpu")

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


def _tiny_net() -> SimultChessNet:
    return SimultChessNet(NetConfig(residual_blocks=1, filters=8, policy_channels=4))


def _solved_root():  # noqa: ANN201
    graph = solve_exact(_dodge_state(), RULESET, max_nodes=5000, max_depth=1)
    return graph.root


def test_solved_value_matches_the_solver_exact_cross_validated_minus_half() -> None:
    # Precondition for everything below to mean what it claims: the matrix
    # this test feeds to NashConv must itself be the same -0.5-valued exact
    # matrix `test_solver_exact.py` already cross-validated against search.
    root = _solved_root()
    net = _tiny_net()
    assert root.stage_matrix is not None
    result = restricted_support_nashconv_exact(
        net,
        root.state,
        RULESET,
        root.stage_matrix,
        root.white_programs,
        root.black_programs,
        SearchConfig(simulations=4),
        "matching_pennies_dodge",
        random.Random(0),
        device=_CPU,
    )
    assert result.solved_value == pytest.approx(-0.5, abs=1e-9)


def test_nashconv_is_nonnegative_on_the_untrained_net() -> None:
    root = _solved_root()
    net = _tiny_net()
    assert root.stage_matrix is not None
    result = restricted_support_nashconv_exact(
        net,
        root.state,
        RULESET,
        root.stage_matrix,
        root.white_programs,
        root.black_programs,
        SearchConfig(simulations=4),
        "matching_pennies_dodge",
        random.Random(1),
        device=_CPU,
    )
    assert result.nashconv >= -1e-9
    assert result.best_response_white >= result.best_response_black - 1e-9


def test_best_response_brackets_the_actual_value() -> None:
    root = _solved_root()
    net = _tiny_net()
    assert root.stage_matrix is not None
    result = restricted_support_nashconv_exact(
        net,
        root.state,
        RULESET,
        root.stage_matrix,
        root.white_programs,
        root.black_programs,
        SearchConfig(simulations=4),
        "matching_pennies_dodge",
        random.Random(2),
        device=_CPU,
    )
    assert result.best_response_white >= result.actual_value - 1e-9
    assert result.best_response_black <= result.actual_value + 1e-9
