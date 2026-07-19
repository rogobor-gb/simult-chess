"""Training loop, loss, checkpointing, and parallel self-play generation
(Phase 13b, design §2.4/§2.5).

**Loss** (design §2.4): ``L(theta) = (v_theta(s)-z)^2 - sum_omega
sigma_omega . log p_theta,omega(s) + c||theta||^2``. The L2 term is applied
via the optimizer's ``weight_decay`` (the standard, numerically-equivalent
realization of ``c||theta||^2`` for SGD-family optimizers), not computed
by hand in the loss -- cleaner and avoids walking every parameter tensor
each step. The policy term splits per the Stage-D scope note in
``learn.selfplay``: slot-1 is a **soft** cross-entropy against the search's
average strategy (a real target); slot-2 is a **hard** cross-entropy
against the actually-played index (ordinary behavioural cloning, since
slot-2 has no independently-refined search statistic, `learn.search`'s
module docstring).

**Self-play parallelism** (design §2.5, "parallelize self-play across the
4 P-cores"): `simult_chess.core.types.State`'s board is backed by an
immutable `mappingproxy` (a Phase-1 design choice), which does not pickle
-- confirmed empirically, not assumed -- so no `State` crosses a
`ProcessPoolExecutor` boundary. Each worker reconstructs the standard
starting state locally and loads the network from a plain `state_dict`
(tensors pickle fine); only picklable primitives (the state dict, config
dataclasses, ruleset, seeds) cross the boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from simult_chess.core.types import Color
from simult_chess.learn.action_grid import NO_SECOND_INDEX, SLOT_SIZE
from simult_chess.learn.config import NetConfig, SearchConfig
from simult_chess.learn.net import SimultChessNet
from simult_chess.learn.selfplay import (
    SelfPlayGame,
    TrainingExample,
    play_one_selfplay_game,
)
from simult_chess.referee.setup import standard_starting_state
from simult_chess.rules.ruleset import RuleSet


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """LIGHT training hyperparameters."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4  # the design's c||theta||^2, via the optimizer
    batch_size: int = 256


def make_optimizer(net: SimultChessNet, config: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )


def _dense_target(target: dict[int, float], device: torch.device) -> torch.Tensor:
    dense = torch.zeros(SLOT_SIZE, dtype=torch.float32, device=device)
    for index, prob in target.items():
        dense[index] = prob
    return dense


def _stack_batch(
    examples: Sequence[TrainingExample], device: torch.device
) -> dict[str, torch.Tensor]:
    planes = torch.from_numpy(np.stack([ex.phase.planes for ex in examples])).to(
        device
    )
    scalars = torch.from_numpy(np.stack([ex.phase.scalars for ex in examples])).to(
        device
    )
    white_slot1_target = torch.stack(
        [_dense_target(ex.phase.white_slot1_target, device) for ex in examples]
    )
    black_slot1_target = torch.stack(
        [_dense_target(ex.phase.black_slot1_target, device) for ex in examples]
    )
    white_slot1_played = torch.tensor(
        [ex.phase.white_slot1_played for ex in examples], device=device
    )
    black_slot1_played = torch.tensor(
        [ex.phase.black_slot1_played for ex in examples], device=device
    )
    white_slot2_played = torch.tensor(
        [_slot2_label(ex.phase.white_slot2_played) for ex in examples], device=device
    )
    black_slot2_played = torch.tensor(
        [_slot2_label(ex.phase.black_slot2_played) for ex in examples], device=device
    )
    z = torch.tensor([ex.z for ex in examples], dtype=torch.float32, device=device)
    return {
        "planes": planes,
        "scalars": scalars,
        "white_slot1_target": white_slot1_target,
        "black_slot1_target": black_slot1_target,
        "white_slot1_played": white_slot1_played,
        "black_slot1_played": black_slot1_played,
        "white_slot2_played": white_slot2_played,
        "black_slot2_played": black_slot2_played,
        "z": z,
    }


def _slot2_label(index: int) -> int:
    """`NO_SECOND_INDEX` (== `SLOT_SIZE`) is already the last valid class for
    the `SLOT_SIZE + 1`-way slot-2 head, so it needs no remapping -- this
    helper exists to keep that assumption named and checkable at one site."""
    assert 0 <= index <= NO_SECOND_INDEX
    return index


def compute_loss(
    net: SimultChessNet, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    """The design §2.4 loss on one batch. Returns `(total_loss, metrics)`."""
    slot1_white_logits, slot1_black_logits, value, policy_features = net(
        batch["planes"], batch["scalars"]
    )
    value_loss = functional.mse_loss(value, batch["z"])

    white_slot1_loss = -(
        batch["white_slot1_target"] * functional.log_softmax(slot1_white_logits, dim=1)
    ).sum(dim=1).mean()
    black_slot1_loss = -(
        batch["black_slot1_target"] * functional.log_softmax(slot1_black_logits, dim=1)
    ).sum(dim=1).mean()

    white_slot2_logits = net.slot2_logits(
        policy_features, batch["white_slot1_played"], Color.WHITE
    )
    black_slot2_logits = net.slot2_logits(
        policy_features, batch["black_slot1_played"], Color.BLACK
    )
    white_slot2_loss = functional.cross_entropy(
        white_slot2_logits, batch["white_slot2_played"]
    )
    black_slot2_loss = functional.cross_entropy(
        black_slot2_logits, batch["black_slot2_played"]
    )

    policy_loss = (
        white_slot1_loss + black_slot1_loss + white_slot2_loss + black_slot2_loss
    )
    total_loss = value_loss + policy_loss
    metrics = {
        "loss": float(total_loss.item()),
        "value_loss": float(value_loss.item()),
        "policy_loss": float(policy_loss.item()),
        "white_slot1_loss": float(white_slot1_loss.item()),
        "black_slot1_loss": float(black_slot1_loss.item()),
        "white_slot2_loss": float(white_slot2_loss.item()),
        "black_slot2_loss": float(black_slot2_loss.item()),
    }
    return total_loss, metrics


def train_step(
    net: SimultChessNet,
    optimizer: torch.optim.Optimizer,
    examples: Sequence[TrainingExample],
    device: torch.device,
) -> dict[str, float]:
    """One SGD step on a minibatch of `TrainingExample`s. Puts `net` in
    `train()` mode (BatchNorm batch statistics -- correct for a training
    step; callers doing inference afterward must set `net.eval()` again,
    e.g. via a fresh `NetworkEvaluator`, which does this itself) and moves
    it to `device` (idempotent if already there) -- found via the Stage-F
    pilot run: a caller that builds `net` fresh (default CPU placement) and
    generates self-play data via `generate_self_play_games` (whose workers
    reconstruct their own CPU copies from a state_dict, never touching the
    caller's own `net` object) would otherwise hit an MPS/CPU tensor
    mismatch on the very first training step, since only the *input batch*
    was being moved to `device`, not the model itself. No unit test caught
    this because they all used `device=torch.device("cpu")` throughout,
    matching a freshly-constructed net's default placement -- the mismatch
    only exists when `device` is MPS."""
    net.train()
    net.to(device)
    batch = _stack_batch(examples, device)
    optimizer.zero_grad()
    loss, metrics = compute_loss(net, batch)
    loss.backward()  # type: ignore[no-untyped-call]
    optimizer.step()
    return metrics


def save_checkpoint(
    path: Path, net: SimultChessNet, optimizer: torch.optim.Optimizer, step: int
) -> None:
    torch.save(
        {
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "net_config": net.config,
            "step": step,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    net: SimultChessNet,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    """Load a checkpoint into `net` (and `optimizer`, if given) in place.
    Returns the saved step count."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    step: int = checkpoint["step"]
    return step


def _self_play_worker(
    args: tuple[dict[str, Any], NetConfig, RuleSet, SearchConfig, int, int],
) -> SelfPlayGame:
    state_dict, net_config, ruleset, search_config, rng_seed, max_phases = args
    net = SimultChessNet(net_config)
    net.load_state_dict(state_dict)
    return play_one_selfplay_game(
        standard_starting_state(),
        net,
        ruleset,
        search_config,
        rng_seed,
        max_phases=max_phases,
        device=torch.device("cpu"),
    )


def generate_self_play_games(
    net: SimultChessNet,
    ruleset: RuleSet,
    search_config: SearchConfig,
    seeds: Sequence[int],
    *,
    max_phases: int = 500,
    max_workers: int = 1,
) -> list[SelfPlayGame]:
    """Generate one self-play game per seed, every game starting from the
    standard opening. `max_workers > 1` parallelizes across processes (design
    §2.5); each worker reconstructs the starting state and loads the network
    from a plain (picklable) `state_dict` rather than crossing a `State`
    object or a live `nn.Module` (MPS tensors, in particular, aren't safely
    shared across a process boundary) -- see the module docstring."""
    state_dict = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    args_list = [
        (state_dict, net.config, ruleset, search_config, seed, max_phases)
        for seed in seeds
    ]
    if max_workers <= 1:
        return [_self_play_worker(args) for args in args_list]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_self_play_worker, args_list))
