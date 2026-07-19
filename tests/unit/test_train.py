"""Training loop, loss, checkpointing, and parallel self-play generation
(Phase 13b, design §2.4/§2.5)."""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from simult_chess.learn.config import NetConfig, SearchConfig  # noqa: E402
from simult_chess.learn.net import SimultChessNet  # noqa: E402
from simult_chess.learn.selfplay import ReplayBuffer  # noqa: E402
from simult_chess.learn.train import (  # noqa: E402
    TrainConfig,
    compute_loss,
    generate_self_play_games,
    load_checkpoint,
    make_optimizer,
    save_checkpoint,
    train_step,
)
from simult_chess.rules.ruleset import RuleSet  # noqa: E402

RULESET = RuleSet()
_CPU = torch.device("cpu")


def _tiny_net() -> SimultChessNet:
    return SimultChessNet(NetConfig(residual_blocks=1, filters=8, policy_channels=4))


def _buffer_from_games(n_games: int, seed_offset: int = 0) -> ReplayBuffer:
    net = _tiny_net()
    games = generate_self_play_games(
        net,
        RULESET,
        SearchConfig(simulations=4),
        seeds=list(range(seed_offset, seed_offset + n_games)),
        max_phases=4,
        max_workers=1,
    )
    buffer = ReplayBuffer(capacity=1000)
    for game in games:
        buffer.add_game(game)
    return buffer


def test_generate_self_play_games_single_process_is_invariant_clean() -> None:
    net = _tiny_net()
    games = generate_self_play_games(
        net, RULESET, SearchConfig(simulations=4), seeds=[0, 1], max_phases=4
    )
    assert len(games) == 2
    assert all(g.violations == () for g in games)


def test_compute_loss_is_finite_and_all_components_present() -> None:
    net = _tiny_net()
    buffer = _buffer_from_games(2)
    examples = buffer.sample(4, random.Random(0))
    from simult_chess.learn.train import _stack_batch

    batch = _stack_batch(examples, _CPU)
    loss, metrics = compute_loss(net, batch)
    assert torch.isfinite(loss)
    for key in (
        "loss",
        "value_loss",
        "policy_loss",
        "white_slot1_loss",
        "black_slot1_loss",
        "white_slot2_loss",
        "black_slot2_loss",
    ):
        assert key in metrics
        assert metrics[key] >= 0.0 or key == "loss"


def test_train_step_reduces_loss_on_a_fixed_batch() -> None:
    # Not a claim about learning speed -- just that one SGD step on a FIXED
    # batch (no new sampling in between) doesn't increase loss on that same
    # batch, the basic sanity property of a correctly-wired gradient step.
    net = _tiny_net()
    optimizer = make_optimizer(net, TrainConfig(learning_rate=1e-2))
    buffer = _buffer_from_games(2)
    examples = buffer.sample(8, random.Random(0))

    from simult_chess.learn.train import _stack_batch

    batch = _stack_batch(examples, _CPU)
    net.eval()
    with torch.no_grad():
        loss_before, _ = compute_loss(net, batch)

    train_step(net, optimizer, examples, _CPU)

    net.eval()
    with torch.no_grad():
        loss_after, _ = compute_loss(net, batch)
    assert float(loss_after) < float(loss_before)


def test_checkpoint_round_trip_preserves_weights_and_step(tmp_path: object) -> None:
    from pathlib import Path

    path = Path(str(tmp_path)) / "ckpt.pt"
    net = _tiny_net()
    optimizer = make_optimizer(net, TrainConfig())
    save_checkpoint(path, net, optimizer, step=17)

    net2 = _tiny_net()
    optimizer2 = make_optimizer(net2, TrainConfig())
    step = load_checkpoint(path, net2, optimizer2)

    assert step == 17
    for (name, value), (name2, value2) in zip(
        net.state_dict().items(), net2.state_dict().items(), strict=True
    ):
        assert name == name2
        assert torch.equal(value, value2)


def test_generate_self_play_games_multiprocess_matches_single_process_shape() -> None:
    # ProcessPoolExecutor requires a real, importable module (spawn can't
    # re-run a -c/stdin script) -- pytest's own test collection satisfies
    # that, so this exercises the actual parallel path, not a stub.
    net = _tiny_net()
    games = generate_self_play_games(
        net,
        RULESET,
        SearchConfig(simulations=4),
        seeds=[10, 11, 12],
        max_phases=4,
        max_workers=2,
    )
    assert len(games) == 3
    assert all(g.violations == () for g in games)
