"""Residual policy-value network (Phase 13b, docs/LEARNING_DESIGN.md §3)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from simult_chess.core.types import Color  # noqa: E402
from simult_chess.learn.action_grid import NO_SECOND_INDEX, SLOT_SIZE  # noqa: E402
from simult_chess.learn.config import NetConfig  # noqa: E402
from simult_chess.learn.net import SimultChessNet  # noqa: E402


def _net() -> SimultChessNet:
    net = SimultChessNet(NetConfig())
    net.eval()  # BatchNorm running stats -> deterministic, batch-size-1 safe
    return net


def _inputs(batch: int) -> tuple[object, object]:
    generator = torch.Generator().manual_seed(0)
    planes = torch.rand(batch, 21, 8, 8, generator=generator)
    scalars = torch.rand(batch, 7, generator=generator)
    return planes, scalars


def test_forward_shapes_both_colors() -> None:
    net = _net()
    planes, scalars = _inputs(4)
    slot1_white, slot1_black, value, policy_features = net(planes, scalars)
    assert slot1_white.shape == (4, SLOT_SIZE)
    assert slot1_black.shape == (4, SLOT_SIZE)
    assert value.shape == (4,)
    assert policy_features.shape[0] == 4


def test_one_forward_pass_gives_distinct_white_and_black_policies() -> None:
    # f_theta(s) = (p_W, p_B, v): one pass, two genuinely different heads.
    net = _net()
    planes, scalars = _inputs(2)
    slot1_white, slot1_black, _, _ = net(planes, scalars)
    assert not torch.allclose(slot1_white, slot1_black)


def test_value_head_is_in_tanh_range() -> None:
    net = _net()
    planes, scalars = _inputs(8)
    _, _, value, _ = net(planes, scalars)
    assert torch.all(value <= 1.0) and torch.all(value >= -1.0)


def test_slot2_head_shape_and_conditioning() -> None:
    net = _net()
    planes, scalars = _inputs(3)
    _, _, _, policy_features = net(planes, scalars)
    a1_a = torch.tensor([0, 1, 2])
    a1_b = torch.tensor([100, 200, 300])
    logits_a = net.slot2_logits(policy_features, a1_a, Color.WHITE)
    logits_b = net.slot2_logits(policy_features, a1_b, Color.WHITE)
    assert logits_a.shape == (3, SLOT_SIZE + 1) == (3, NO_SECOND_INDEX + 1)
    # slot-2 genuinely depends on the slot-1 action (autoregressive), so
    # different slot-1 indices must give different slot-2 logits.
    assert not torch.allclose(logits_a, logits_b)


def test_slot2_head_differs_by_color() -> None:
    net = _net()
    planes, scalars = _inputs(2)
    _, _, _, policy_features = net(planes, scalars)
    a1 = torch.tensor([5, 5])
    white_logits = net.slot2_logits(policy_features, a1, Color.WHITE)
    black_logits = net.slot2_logits(policy_features, a1, Color.BLACK)
    assert not torch.allclose(white_logits, black_logits)


def test_forward_is_deterministic_in_eval_mode() -> None:
    net = _net()
    planes, scalars = _inputs(4)
    first = net(planes, scalars)[:2]
    second = net(planes, scalars)[:2]
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_runs_on_the_encoder_output() -> None:
    from simult_chess.interop.encoding import encode_state
    from simult_chess.referee.setup import standard_starting_state
    from simult_chess.rules.ruleset import RuleSet

    planes_np, scalars_np = encode_state(standard_starting_state(), RuleSet())
    planes = torch.from_numpy(np.expand_dims(planes_np, 0))
    scalars = torch.from_numpy(np.expand_dims(scalars_np, 0))
    net = _net()
    slot1_white, slot1_black, value, _ = net(planes, scalars)
    assert slot1_white.shape == (1, SLOT_SIZE)
    assert slot1_black.shape == (1, SLOT_SIZE)
    assert value.shape == (1,)
    assert bool(torch.isfinite(slot1_white).all())
    assert bool(torch.isfinite(slot1_black).all())
    assert bool(torch.isfinite(value).all())
