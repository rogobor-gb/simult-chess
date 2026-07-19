"""Learning diagnostics: entropy, calibration, color-symmetry (Phase 13b,
design §6.4)."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from simult_chess.learn.config import NetConfig, SearchConfig  # noqa: E402
from simult_chess.learn.diagnostics import (  # noqa: E402
    color_symmetry_spot_check,
    entropy,
    stage_policy_entropy,
    value_calibration,
)
from simult_chess.learn.net import SimultChessNet  # noqa: E402
from simult_chess.learn.selfplay import (  # noqa: E402
    ReplayBuffer,
    SelfPlayGame,
    play_one_selfplay_game,
)
from simult_chess.referee.setup import standard_starting_state  # noqa: E402
from simult_chess.rules.ruleset import RuleSet  # noqa: E402

RULESET = RuleSet()
_CPU = torch.device("cpu")


def _tiny_net() -> SimultChessNet:
    return SimultChessNet(NetConfig(residual_blocks=1, filters=8, policy_channels=4))


def test_entropy_is_zero_for_a_pure_strategy() -> None:
    assert entropy({0: 1.0}) == 0.0


def test_entropy_is_log_n_for_uniform() -> None:
    uniform = {i: 1.0 / 4 for i in range(4)}
    assert entropy(uniform) == pytest.approx(math.log(4), abs=1e-9)


def test_entropy_handles_an_empty_distribution() -> None:
    assert entropy({}) == 0.0


def test_stage_policy_entropy_summarizes_recorded_targets() -> None:
    net = _tiny_net()
    games = [
        play_one_selfplay_game(
            standard_starting_state(),
            net,
            RULESET,
            SearchConfig(simulations=4),
            rng_seed=i,
            max_phases=3,
            device=_CPU,
        )
        for i in range(2)
    ]
    summary = stage_policy_entropy(games)
    assert summary.n_phases == sum(len(g.phases) for g in games)
    assert summary.mean_white >= 0.0
    assert summary.mean_black >= 0.0


def test_stage_policy_entropy_handles_no_phases() -> None:
    summary = stage_policy_entropy([])
    assert summary.mean_white == 0.0
    assert summary.mean_black == 0.0
    assert summary.n_phases == 0


def test_value_calibration_bins_and_reports_mae() -> None:
    net = _tiny_net()
    game = play_one_selfplay_game(
        standard_starting_state(),
        net,
        RULESET,
        SearchConfig(simulations=4),
        rng_seed=0,
        max_phases=4,
        device=_CPU,
    )
    buffer = ReplayBuffer(capacity=100)
    buffer.add_game(game)
    result = value_calibration(net, buffer.examples, n_bins=5, device=_CPU)
    assert result.mean_absolute_error >= 0.0
    total_binned = sum(b.count for b in result.bins)
    assert total_binned == len(buffer)
    for b in result.bins:
        assert -1.0 <= b.predicted_mean <= 1.0
        assert b.count > 0


def test_color_symmetry_spot_check_with_no_decisive_games_is_nan() -> None:
    game = SelfPlayGame(
        phases=(), outcome="phase_limit_reached", rng_seed=0, violations=()
    )
    result = color_symmetry_spot_check([game])
    assert result.decisive == 0
    assert math.isnan(result.p_value)


def test_color_symmetry_spot_check_counts_decisive_games_correctly() -> None:
    def _fake_game(outcome: str) -> SelfPlayGame:
        return SelfPlayGame(phases=(), outcome=outcome, rng_seed=0, violations=())

    games = [
        _fake_game("white_wins"),
        _fake_game("white_wins"),
        _fake_game("black_wins"),
        _fake_game("draw"),
        _fake_game("phase_limit_reached"),
    ]
    result = color_symmetry_spot_check(games)
    assert result.white_wins == 2
    assert result.decisive == 3
    assert not math.isnan(result.p_value)
