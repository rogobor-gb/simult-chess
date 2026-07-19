"""Exact restricted-support NashConv on M5 fixtures (Phase 13b, design §6.3)."""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from simult_chess.core.collision import mirror_state  # noqa: E402
from simult_chess.core.types import (  # noqa: E402
    Bookkeeping,
    CastlingRights,
    Color,
    Square,
    State,
    Token,
)
from simult_chess.learn.config import NetConfig, SearchConfig  # noqa: E402
from simult_chess.learn.nashconv import restricted_support_nashconv  # noqa: E402
from simult_chess.learn.net import SimultChessNet  # noqa: E402
from simult_chess.rules.ruleset import RuleSet  # noqa: E402

RULESET = RuleSet()
_CPU = torch.device("cpu")
_NO_CASTLING = CastlingRights(
    white_kingside=False,
    white_queenside=False,
    black_kingside=False,
    black_queenside=False,
)


def _midgame_knight_pawn_fixture() -> State:
    # The same chi-symmetric fixture as
    # tests/property/test_m5_symmetric_value.py.
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


def _tiny_net() -> SimultChessNet:
    return SimultChessNet(NetConfig(residual_blocks=1, filters=8, policy_channels=4))


def test_fixture_is_actually_chi_symmetric() -> None:
    state = _midgame_knight_pawn_fixture()
    assert mirror_state(state) == state


def test_nashconv_solved_value_matches_m5_zero() -> None:
    # M5's own proven property (test_m5_symmetric_value.py): the exact
    # matrix-game value on a chi-symmetric fixture with chi-closed supports
    # is 0. This module's `solved_value` must reproduce that -- a free
    # sanity anchor for the NashConv computation itself.
    state = _midgame_knight_pawn_fixture()
    net = _tiny_net()
    result = restricted_support_nashconv(
        net,
        state,
        RULESET,
        SearchConfig(simulations=4),
        "midgame_knight_pawn",
        random.Random(0),
        device=_CPU,
    )
    assert result.solved_value == pytest.approx(0.0, abs=1e-6)


def test_nashconv_is_nonnegative() -> None:
    # NashConv >= 0 always (it's zero exactly at a Nash equilibrium); a
    # negative value would indicate a bug in the best-response computation.
    state = _midgame_knight_pawn_fixture()
    net = _tiny_net()
    result = restricted_support_nashconv(
        net,
        state,
        RULESET,
        SearchConfig(simulations=4),
        "midgame_knight_pawn",
        random.Random(1),
        device=_CPU,
    )
    assert result.nashconv >= -1e-9
    assert result.best_response_white >= result.best_response_black - 1e-9


def test_nashconv_best_response_brackets_the_actual_value() -> None:
    # By definition: White's best response value is >= the actual value
    # under the current joint policy (White could not do worse by playing
    # its own strategy than its best pure response); symmetric for Black.
    state = _midgame_knight_pawn_fixture()
    net = _tiny_net()
    result = restricted_support_nashconv(
        net,
        state,
        RULESET,
        SearchConfig(simulations=4),
        "midgame_knight_pawn",
        random.Random(2),
        device=_CPU,
    )
    assert result.best_response_white >= result.actual_value - 1e-9
    assert result.best_response_black <= result.actual_value + 1e-9
