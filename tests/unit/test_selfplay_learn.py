"""Self-play data generation + replay buffer (Phase 13b, design §2.4/§2.5)."""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from simult_chess.learn.config import NetConfig, SearchConfig  # noqa: E402
from simult_chess.learn.net import SimultChessNet  # noqa: E402
from simult_chess.learn.selfplay import (  # noqa: E402
    ReplayBuffer,
    SelfPlayGame,
    TrainingExample,
    play_one_selfplay_game,
)
from simult_chess.referee.setup import standard_starting_state  # noqa: E402
from simult_chess.rules.ruleset import RuleSet  # noqa: E402

RULESET = RuleSet()
_CPU = torch.device("cpu")


def _tiny_net() -> SimultChessNet:
    return SimultChessNet(NetConfig(residual_blocks=1, filters=8, policy_channels=4))


def _play(seed: int, max_phases: int = 6) -> SelfPlayGame:
    net = _tiny_net()
    return play_one_selfplay_game(
        standard_starting_state(),
        net,
        RULESET,
        SearchConfig(simulations=6),
        rng_seed=seed,
        max_phases=max_phases,
        device=_CPU,
    )


def test_selfplay_game_is_invariant_clean_and_records_every_phase() -> None:
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


def test_selfplay_phase_records_have_valid_targets() -> None:
    game = _play(seed=1, max_phases=4)
    for phase in game.phases:
        assert phase.planes.shape == (21, 8, 8)
        assert phase.scalars.shape == (7,)
        assert pytest.approx(sum(phase.white_slot1_target.values()), abs=1e-5) == 1.0
        assert pytest.approx(sum(phase.black_slot1_target.values()), abs=1e-5) == 1.0
        assert phase.white_slot1_played in phase.white_slot1_target
        assert phase.black_slot1_played in phase.black_slot1_target


def test_selfplay_game_is_deterministic_given_the_same_net_weights_and_seed() -> None:
    # Two independently-constructed nets are randomly initialized differently,
    # so pin one net's weights and reuse it -- determinism is a property of
    # (fixed weights, fixed seed), not of construction.
    net = _tiny_net()
    game_a = play_one_selfplay_game(
        standard_starting_state(),
        net,
        RULESET,
        SearchConfig(simulations=6),
        rng_seed=3,
        max_phases=4,
        device=_CPU,
    )
    game_b = play_one_selfplay_game(
        standard_starting_state(),
        net,
        RULESET,
        SearchConfig(simulations=6),
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


@pytest.mark.parametrize(
    ("outcome", "expected_z"),
    [
        ("white_wins", 1.0),
        ("black_wins", -1.0),
        ("draw", 0.0),
        ("phase_limit_reached", 0.0),
        ("aborted", 0.0),
    ],
)
def test_replay_buffer_maps_outcome_to_z(outcome: str, expected_z: float) -> None:
    import numpy as np

    from simult_chess.learn.selfplay import PhaseRecord

    phase = PhaseRecord(
        planes=np.zeros((21, 8, 8), dtype=np.float32),
        scalars=np.zeros(7, dtype=np.float32),
        white_slot1_target={0: 1.0},
        black_slot1_target={0: 1.0},
        white_slot1_played=0,
        white_slot2_played=-1,
        black_slot1_played=0,
        black_slot2_played=-1,
    )
    game = SelfPlayGame(phases=(phase,), outcome=outcome, rng_seed=0, violations=())
    buffer = ReplayBuffer(capacity=10)
    buffer.add_game(game)
    assert buffer.examples[0].z == expected_z


def test_replay_buffer_respects_capacity_as_a_ring_buffer() -> None:
    import numpy as np

    from simult_chess.learn.selfplay import PhaseRecord

    def _phase(marker: int) -> PhaseRecord:
        return PhaseRecord(
            planes=np.full((21, 8, 8), marker, dtype=np.float32),
            scalars=np.zeros(7, dtype=np.float32),
            white_slot1_target={0: 1.0},
            black_slot1_target={0: 1.0},
            white_slot1_played=marker,
            white_slot2_played=-1,
            black_slot1_played=marker,
            black_slot2_played=-1,
        )

    buffer = ReplayBuffer(capacity=3)
    for i in range(5):
        game = SelfPlayGame(
            phases=(_phase(i),), outcome="draw", rng_seed=i, violations=()
        )
        buffer.add_game(game)
    assert len(buffer) == 3
    # The oldest two (markers 0, 1) should have been evicted.
    markers = {ex.phase.white_slot1_played for ex in buffer.examples}
    assert markers == {2, 3, 4}


def test_replay_buffer_sample_is_seeded_and_bounded() -> None:
    game = _play(seed=2, max_phases=5)
    buffer = ReplayBuffer(capacity=100)
    buffer.add_game(game)
    sample = buffer.sample(1000, random.Random(0))
    assert len(sample) == len(buffer)
    assert all(isinstance(ex, TrainingExample) for ex in sample)
