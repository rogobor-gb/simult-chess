"""Relative-Elo ladder + strength gate (Phase 13b, design §6.1/§6.2)."""

from __future__ import annotations

import math
import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from simult_chess.agents.greedy import greedy_program  # noqa: E402
from simult_chess.agents.random_legal import random_legal_program  # noqa: E402
from simult_chess.learn.agent import LearnedAgent  # noqa: E402
from simult_chess.learn.config import NetConfig, SearchConfig  # noqa: E402
from simult_chess.learn.evaluate import (  # noqa: E402
    LadderMatchResult,
    bootstrap_elo,
    elo_diff_from_score,
    play_ladder_match,
    strength_gate_passes,
)
from simult_chess.learn.net import SimultChessNet  # noqa: E402
from simult_chess.rules.ruleset import RuleSet  # noqa: E402

RULESET = RuleSet()
_CPU = torch.device("cpu")


def _tiny_agent() -> LearnedAgent:
    net = SimultChessNet(NetConfig(residual_blocks=1, filters=8, policy_channels=4))
    return LearnedAgent(net=net, search_config=SearchConfig(simulations=4), device=_CPU)


def test_ladder_match_is_color_balanced_and_scores_sum_correctly() -> None:
    agent = _tiny_agent()
    result = play_ladder_match(
        agent, random_legal_program, RULESET, n_games=4, base_seed=0, max_phases=4
    )
    assert len(result.scores) == 4
    assert len(result.outcomes) == 4
    assert all(s in (0.0, 0.5, 1.0) for s in result.scores)
    assert 0.0 <= result.win_rate <= 1.0
    assert result.decisive_games == sum(1 for s in result.scores if s != 0.5)


def test_ladder_match_is_deterministic_given_a_fixed_net_and_seed() -> None:
    net = SimultChessNet(NetConfig(residual_blocks=1, filters=8, policy_channels=4))
    search_config = SearchConfig(simulations=4)
    agent_a = LearnedAgent(net=net, search_config=search_config, device=_CPU)
    result_a = play_ladder_match(
        agent_a, greedy_program, RULESET, n_games=3, base_seed=5, max_phases=4
    )
    agent_b = LearnedAgent(net=net, search_config=search_config, device=_CPU)
    result_b = play_ladder_match(
        agent_b, greedy_program, RULESET, n_games=3, base_seed=5, max_phases=4
    )
    assert result_a.scores == result_b.scores
    assert result_a.outcomes == result_b.outcomes


def test_elo_diff_from_score_is_zero_at_half_and_monotonic() -> None:
    e50 = elo_diff_from_score(0.5)
    e60, e90 = elo_diff_from_score(0.6), elo_diff_from_score(0.9)
    e40, e10 = elo_diff_from_score(0.4), elo_diff_from_score(0.1)
    assert e50 == pytest.approx(0.0, abs=1e-9)
    assert e90 > e60 > e50
    assert e10 < e40 < e50
    # Clamped away from +-inf at a shutout.
    assert math.isfinite(elo_diff_from_score(1.0))


def test_bootstrap_elo_ci_contains_the_point_estimate() -> None:
    scores = [1.0, 1.0, 0.5, 0.0, 1.0, 1.0, 0.0, 1.0]
    estimate = bootstrap_elo(scores, random.Random(0), n_resamples=500)
    assert estimate.lower <= estimate.point <= estimate.upper


def test_bootstrap_elo_ci_is_seeded_and_reproducible() -> None:
    scores = [1.0, 0.5, 0.0, 1.0, 1.0]
    a = bootstrap_elo(scores, random.Random(7), n_resamples=300)
    b = bootstrap_elo(scores, random.Random(7), n_resamples=300)
    assert a == b


def test_strength_gate_fails_on_no_decisive_games() -> None:
    result = LadderMatchResult(scores=(0.5, 0.5, 0.5), outcomes=("draw",) * 3)
    passes, p_value = strength_gate_passes(result)
    assert not passes
    assert p_value == 1.0


def test_strength_gate_passes_on_a_lopsided_seeded_win_record() -> None:
    # A synthetic, obviously-significant record (not a live match -- the
    # gate's statistical logic is what's under test here).
    wins = tuple(1.0 for _ in range(30))
    losses = tuple(0.0 for _ in range(2))
    result = LadderMatchResult(scores=wins + losses, outcomes=("white_wins",) * 32)
    passes, p_value = strength_gate_passes(result)
    assert passes
    assert p_value < 0.01
