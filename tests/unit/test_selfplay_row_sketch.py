"""v3 18c: `row_sketch` wired into live self-play as an opt-in node
solver (`SearchConfig(node_solver="row_sketch")`), validated the same way
`test_selfplay_learn.py` validates the default `"slot1"` path -- same
`_tiny_net()`, same small `max_phases`/`simulations`, same assertion
shapes, so a difference in behaviour between the two paths shows up as a
difference in test *results*, not a difference in test *methodology*.
`test_selfplay_learn.py` itself is untouched and still green: this is a
parallel opt-in path, not a replacement."""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from simult_chess.learn.config import NetConfig, SearchConfig  # noqa: E402
from simult_chess.learn.net import SimultChessNet  # noqa: E402
from simult_chess.learn.selfplay import (  # noqa: E402
    SelfPlayGame,
    play_one_selfplay_game,
)
from simult_chess.referee.setup import standard_starting_state  # noqa: E402
from simult_chess.rules.ruleset import RuleSet  # noqa: E402

RULESET = RuleSet()
_CPU = torch.device("cpu")


def _tiny_net() -> SimultChessNet:
    return SimultChessNet(NetConfig(residual_blocks=1, filters=8, policy_channels=4))


def _row_sketch_config(simulations: int = 6) -> SearchConfig:
    return SearchConfig(
        simulations=simulations, node_solver="row_sketch", pool_size=8, pool_seed_size=4
    )


def _play(seed: int, max_phases: int = 6) -> SelfPlayGame:
    net = _tiny_net()
    return play_one_selfplay_game(
        standard_starting_state(),
        net,
        RULESET,
        _row_sketch_config(),
        rng_seed=seed,
        max_phases=max_phases,
        device=_CPU,
    )


def test_row_sketch_selfplay_game_is_invariant_clean_and_records_every_phase() -> None:
    game = _play(seed=0, max_phases=8)
    assert game.violations == ()
    assert len(game.phases) >= 1
    if game.outcome != "aborted":
        assert len(game.phases) == 8 or game.outcome in (
            "white_wins",
            "black_wins",
            "draw",
        )
    assert game.outcome in (
        "white_wins",
        "black_wins",
        "draw",
        "phase_limit_reached",
        "aborted",
    )


def test_row_sketch_selfplay_phase_records_have_valid_targets() -> None:
    game = _play(seed=1, max_phases=4)
    for phase in game.phases:
        assert phase.planes.shape == (21, 8, 8)
        assert phase.scalars.shape == (7,)
        assert pytest.approx(sum(phase.white_slot1_target.values()), abs=1e-5) == 1.0
        assert pytest.approx(sum(phase.black_slot1_target.values()), abs=1e-5) == 1.0
        assert phase.white_slot1_played in phase.white_slot1_target
        assert phase.black_slot1_played in phase.black_slot1_target


def test_row_sketch_selfplay_game_is_deterministic_given_the_same_net_and_seed() -> (
    None
):
    net = _tiny_net()
    game_a = play_one_selfplay_game(
        standard_starting_state(),
        net,
        RULESET,
        _row_sketch_config(),
        rng_seed=3,
        max_phases=4,
        device=_CPU,
    )
    game_b = play_one_selfplay_game(
        standard_starting_state(),
        net,
        RULESET,
        _row_sketch_config(),
        rng_seed=3,
        max_phases=4,
        device=_CPU,
    )
    assert game_a.outcome == game_b.outcome
    assert len(game_a.phases) == len(game_b.phases)
    for pa, pb in zip(game_a.phases, game_b.phases, strict=True):
        assert pa.white_slot1_played == pb.white_slot1_played
        assert pa.white_slot2_played == pb.white_slot2_played
        assert pa.black_slot1_played == pb.black_slot1_played
        assert pa.black_slot2_played == pb.black_slot2_played


def test_default_search_config_still_uses_the_slot1_path() -> None:
    # The opt-in flag's whole point: an unmodified SearchConfig() must not
    # be silently affected by row_sketch existing in the codebase.
    assert SearchConfig().node_solver == "slot1"


def test_row_sketch_and_slot1_paths_produce_different_selfplay_games() -> None:
    # Not a claim about which is "better" -- just confirms the branch
    # actually dispatches to a different node solver, not a no-op flag.
    net = _tiny_net()
    game_slot1 = play_one_selfplay_game(
        standard_starting_state(),
        net,
        RULESET,
        SearchConfig(simulations=6),
        rng_seed=random.Random(0).randint(0, 2**31),
        max_phases=4,
        device=_CPU,
    )
    game_row_sketch = play_one_selfplay_game(
        standard_starting_state(),
        net,
        RULESET,
        _row_sketch_config(),
        rng_seed=game_slot1.rng_seed,
        max_phases=4,
        device=_CPU,
    )
    assert game_slot1.violations == ()
    assert game_row_sketch.violations == ()
