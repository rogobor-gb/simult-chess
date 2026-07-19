"""Residual policy-value network (Phase 13b, docs/LEARNING_DESIGN.md §3).

Input: the ``(21, 8, 8)`` planes + ``(7,)`` scalars of ``interop.encoding``.
The 7 scalars are broadcast to 7 constant planes and concatenated onto the 21
board planes (a 28-channel stem input), the standard way to feed global
features to a convolutional trunk; the trunk itself is B=6 residual blocks of
F=64 filters (LIGHT).

Policy head: the fixed factored grid of ``action_grid`` (SLOT_SIZE=9026 per
slot), **autoregressive over the two program slots** -- a slot-1 head over the
grid, then the chosen slot-1 action is embedded and concatenated to the shared
policy features for a slot-2 head (SLOT_SIZE+1 logits, the +1 being
``NO_SECOND_INDEX`` for single-action programs). Value head: 1x1 conv to one
plane -> MLP -> tanh, output in [-1, 1] (§3.4).
"""

from __future__ import annotations

import torch
from torch import nn

from simult_chess.learn.action_grid import SLOT_SIZE
from simult_chess.learn.config import NetConfig


def default_device() -> torch.device:
    """MPS when available (the LIGHT profile target), else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _ResidualBlock(nn.Module):
    """Two 3x3 convs + BatchNorm + ReLU with an identity skip (§3.4)."""

    def __init__(self, filters: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return torch.relu(x + h)


class SimultChessNet(nn.Module):
    """Policy-value network :math:`f_\\theta(s)=(p_W, p_B, v)`; one forward pass
    predicts both colours' slot-1 logits and the value, and ``slot2_logits``
    completes the autoregressive factorization given a chosen slot-1 action."""

    def __init__(self, config: NetConfig | None = None) -> None:
        super().__init__()
        self.config = config or NetConfig()
        stem_in = self.config.num_planes + self.config.num_scalars
        self.stem_conv = nn.Conv2d(
            stem_in, self.config.filters, 3, padding=1, bias=False
        )
        self.stem_bn = nn.BatchNorm2d(self.config.filters)
        self.blocks = nn.ModuleList(
            _ResidualBlock(self.config.filters)
            for _ in range(self.config.residual_blocks)
        )

        self.policy_conv = nn.Conv2d(
            self.config.filters, self.config.policy_channels, 1, bias=False
        )
        self.policy_bn = nn.BatchNorm2d(self.config.policy_channels)
        policy_dim = self.config.policy_channels * 8 * 8
        self.slot1_head = nn.Linear(policy_dim, SLOT_SIZE)
        self.a1_embedding = nn.Embedding(SLOT_SIZE, self.config.a1_embed_dim)
        self.slot2_head = nn.Linear(
            policy_dim + self.config.a1_embed_dim, SLOT_SIZE + 1
        )

        self.value_conv = nn.Conv2d(
            self.config.filters, self.config.value_channels, 1, bias=False
        )
        self.value_bn = nn.BatchNorm2d(self.config.value_channels)
        self.value_fc1 = nn.Linear(
            self.config.value_channels * 8 * 8, self.config.value_hidden
        )
        self.value_fc2 = nn.Linear(self.config.value_hidden, 1)

    def _trunk(self, planes: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        batch = planes.shape[0]
        num_scalars = scalars.shape[1]
        scalar_planes = scalars.view(batch, num_scalars, 1, 1).expand(
            batch, num_scalars, 8, 8
        )
        x = torch.cat((planes, scalar_planes), dim=1)
        x = torch.relu(self.stem_bn(self.stem_conv(x)))
        for block in self.blocks:
            x = block(x)
        return x

    def _policy_features(self, trunk: torch.Tensor) -> torch.Tensor:
        p = torch.relu(self.policy_bn(self.policy_conv(trunk)))
        return p.flatten(1)

    def _value(self, trunk: torch.Tensor) -> torch.Tensor:
        v = torch.relu(self.value_bn(self.value_conv(trunk)))
        v = torch.relu(self.value_fc1(v.flatten(1)))
        return torch.tanh(self.value_fc2(v)).squeeze(-1)

    def forward(
        self, planes: torch.Tensor, scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(slot1_logits (N, SLOT_SIZE), value (N,), policy_features)``.

        ``policy_features`` is returned so a caller (the search) can compute
        ``slot2_logits`` for a sampled slot-1 action without re-running the
        trunk.
        """
        trunk = self._trunk(planes, scalars)
        policy_features = self._policy_features(trunk)
        slot1_logits = self.slot1_head(policy_features)
        value = self._value(trunk)
        return slot1_logits, value, policy_features

    def slot2_logits(
        self, policy_features: torch.Tensor, a1_indices: torch.Tensor
    ) -> torch.Tensor:
        """Slot-2 logits ``(N, SLOT_SIZE + 1)`` conditioned on the chosen slot-1
        grid indices ``a1_indices`` (N,)."""
        embedded = self.a1_embedding(a1_indices)
        logits: torch.Tensor = self.slot2_head(
            torch.cat((policy_features, embedded), dim=1)
        )
        return logits
