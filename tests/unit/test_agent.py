"""The learned agent (Phase 13b, docs/LEARNING_DESIGN.md §2.5): NetworkEvaluator
+ LearnedAgent, conforming to agents.base.Agent so it drops into
referee/match.py and harness/selfplay.py unchanged."""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from simult_chess.core import legality  # noqa: E402
from simult_chess.core.types import Color  # noqa: E402
from simult_chess.learn.agent import LearnedAgent, NetworkEvaluator  # noqa: E402
from simult_chess.learn.config import NetConfig, SearchConfig  # noqa: E402
from simult_chess.learn.net import SimultChessNet  # noqa: E402
from simult_chess.referee.setup import standard_starting_state  # noqa: E402
from simult_chess.rules.ruleset import RuleSet  # noqa: E402

RULESET = RuleSet()
_CPU = torch.device("cpu")


def _tiny_net() -> SimultChessNet:
    # A small trunk keeps these tests fast; correctness doesn't depend on
    # the LIGHT-sized B=6, F=64 config.
    return SimultChessNet(NetConfig(residual_blocks=1, filters=8, policy_channels=4))


def test_network_evaluator_returns_a_valid_leaf() -> None:
    net = _tiny_net()
    evaluator = NetworkEvaluator(net, device=_CPU)
    state = standard_starting_state()
    value, prior_white, prior_black, context = evaluator.evaluate_leaf(state, RULESET)
    assert -1.0 <= value <= 1.0
    assert prior_white and prior_black
    assert all(p >= 0.0 for p in prior_white.values())
    assert pytest.approx(sum(prior_white.values()), abs=1e-5) == 1.0
    assert pytest.approx(sum(prior_black.values()), abs=1e-5) == 1.0
    assert context is not None


def test_network_evaluator_slot2_prior_is_a_distribution() -> None:
    net = _tiny_net()
    evaluator = NetworkEvaluator(net, device=_CPU)
    state = standard_starting_state()
    _, prior_white, _, context = evaluator.evaluate_leaf(state, RULESET)
    from simult_chess.learn.action_grid import slot1_legal_actions

    slot1 = slot1_legal_actions(state, Color.WHITE, RULESET)
    first_index = next(iter(prior_white))
    first = slot1[first_index]
    dist = evaluator.slot2_prior(
        context, Color.WHITE, state, RULESET, first_index, first
    )
    assert dist
    assert all(p >= 0.0 for p in dist.values())
    assert pytest.approx(sum(dist.values()), abs=1e-5) == 1.0


def test_learned_agent_produces_legal_programs() -> None:
    net = _tiny_net()
    agent = LearnedAgent(
        net=net, search_config=SearchConfig(simulations=8), device=_CPU
    )
    state = standard_starting_state()
    for seed in range(5):
        rng = random.Random(seed)
        program = agent(state, Color.WHITE, RULESET, rng)
        assert legality.is_legal_program(state, program, Color.WHITE, RULESET)
        program_b = agent(state, Color.BLACK, RULESET, rng)
        assert legality.is_legal_program(state, program_b, Color.BLACK, RULESET)


def test_learned_agent_is_deterministic_given_the_same_seed() -> None:
    net = _tiny_net()
    agent = LearnedAgent(
        net=net, search_config=SearchConfig(simulations=8), device=_CPU
    )
    state = standard_starting_state()
    program_a = agent(state, Color.WHITE, RULESET, random.Random(7))
    program_b = agent(state, Color.WHITE, RULESET, random.Random(7))
    assert program_a == program_b


def test_learned_agent_handles_a_sparse_board() -> None:
    from conftest import build_state

    from simult_chess.core.types import Square, Token

    king = Token(id=1, color=Color.WHITE, typ="k")
    enemy_king = Token(id=2, color=Color.BLACK, typ="k")
    state = build_state({king: Square(4, 0), enemy_king: Square(4, 7)})
    net = _tiny_net()
    agent = LearnedAgent(
        net=net, search_config=SearchConfig(simulations=8), device=_CPU
    )
    program = agent(state, Color.WHITE, RULESET, random.Random(0))
    assert legality.is_legal_program(state, program, Color.WHITE, RULESET)


def test_learned_agent_drops_into_play_one_game() -> None:
    from simult_chess.harness.selfplay import play_one_game

    net_white = _tiny_net()
    net_black = _tiny_net()
    agent_white = LearnedAgent(
        net=net_white, search_config=SearchConfig(simulations=4), device=_CPU
    )
    agent_black = LearnedAgent(
        net=net_black, search_config=SearchConfig(simulations=4), device=_CPU
    )
    report = play_one_game(
        standard_starting_state(),
        agent_white,
        agent_black,
        RULESET,
        rng_seed=0,
        max_phases=6,
    )
    assert report.phases_played == 6
    assert report.violations == ()
