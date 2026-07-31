"""Residual policy-value network (Phase 13b, docs/LEARNING_DESIGN.md §3).

Input: the ``(21, 8, 8)`` planes + ``(7,)`` scalars of ``interop.encoding``.
The 7 scalars are broadcast to 7 constant planes and concatenated onto the 21
board planes (a 28-channel stem input), the standard way to feed global
features to a convolutional trunk; the trunk itself is B=6 residual blocks of
F=64 filters (LIGHT).

Policy: :math:`f_\\theta(s)=(\\mathbf p_W, \\mathbf p_B, v)` -- **one forward
pass predicts both colours' policies** (design §2.4: the state is perfect-
information and player-independent, §3.1), via separate slot-1/slot-2 head
weights per colour sharing one trunk. Each colour's policy is the fixed
factored grid of ``action_grid`` (SLOT_SIZE=9026 per slot), **autoregressive
over the two program slots**: a slot-1 head over the grid, then the chosen
slot-1 action is embedded (one shared embedding table -- it embeds a grid
index, not a colour-specific quantity) and concatenated to the policy
features for that colour's slot-2 head (SLOT_SIZE+1 logits, the +1 being
``NO_SECOND_INDEX`` for single-action programs). Value head: 1x1 conv to one
plane -> MLP -> tanh, output in [-1, 1] (§3.4).

**v3 18a'.5: chi-antisymmetric value and slot-1 policy heads.** Rather than
*learning* V(chi(s)) = -V(s) (M3's colour-symmetry guarantee, spec) from
data, `forward` imposes it architecturally: every call also runs the trunk
on `chi_transform(planes, scalars)` (a fixed, differentiable tensor
transform -- rank-flip + colour-paired channel swap, cross-validated against
`core.collision.mirror_state`/`interop.encoding.encode_state` directly, see
`test_net_chi_symmetry.py`) and combines `value = 0.5*(g(e(s)) - g(e(chi(s))))`.
Slot-1's two colour heads are symmetrized the analogous way, using
`action_grid.MIRROR_PERMUTATION` (also independently cross-validated) to
reindex logits: `slot1_white(s) = 0.5*(head_white(e(s)) +
mirror(head_black(e(chi(s)))))`, and `slot1_black` symmetrically -- an
*identity* by construction (`chi` and the grid mirror are both involutions),
not something the loss needs to teach the net. This is also why it halves
the value head's effective sample complexity: every training example
teaches both `V(s)` and `V(chi(s))` at once.

**Scope decision, flagged for review: slot-2 is deliberately NOT
symmetrized this way.** It is a *conditional* completion given a specific
sampled slot-1 action (`slot2_logits`, called separately, after slot-1
sampling, using cached `policy_features`) -- properly symmetrizing it would
need caching *both* `e(s)` and `e(chi(s))`'s features plus mirroring the
conditioning action index, a materially bigger change to the evaluator/
search's calling convention. `policy_features`, `forward`'s fourth return
value, is therefore still exactly `e(s)` alone (not a chi-combination), so
`slot2_logits` and every existing caller of it are unaffected. Slot-2
already has a separate, larger open defect (H2: its own training update has
zero expected gradient, `learn.search`'s module docstring) that this
doesn't touch either way.
"""

from __future__ import annotations

import torch
from torch import nn

from simult_chess.core.types import Color
from simult_chess.learn.action_grid import MIRROR_PERMUTATION, SLOT_SIZE
from simult_chess.learn.config import NetConfig

_COLOR_KEYS: dict[Color, str] = {Color.WHITE: "white", Color.BLACK: "black"}

# v3 18a'.5: interop.encoding.py's plane layout, colour-paired for chi's
# token-colour inversion (channel 12, cooldown, is colour-agnostic already
# and needs only the rank flip every plane gets -- no pairing entry here).
_PIECE_PLANE_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11),
)
_RESERVATION_ACTOR_PLANE_PAIRS: tuple[tuple[int, int], ...] = ((13, 15), (14, 16))
# Sign unchanged: mu fixes file, and dfile is a file difference.
_DFILE_PLANE_PAIRS: tuple[tuple[int, int], ...] = ((17, 19),)
# Sign flips: mu negates every rank, and drank is a rank difference.
_DRANK_PLANE_PAIRS: tuple[tuple[int, int], ...] = ((18, 20),)
# Castling rights, colour-paired kingside/queenside across White<->Black.
_SCALAR_PAIRS: tuple[tuple[int, int], ...] = ((0, 2), (1, 3))


def chi_transform(
    planes: torch.Tensor, scalars: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The board-mirror/colour-swap involution chi, as a batched tensor
    transform on `interop.encoding`'s own `(N, 21, 8, 8)`/`(N, 7)` encoding
    -- equivalent to `encode_state(mirror_state(s))` but computed directly
    on the tensor (no `State` round trip inside the network), since `mirror_
    state`'s rank flip (files fixed) and colour inversion act as a fixed
    axis flip plus a fixed set of channel-pair swaps (cross-validated
    against the real `encode_state`/`mirror_state` in
    `test_net_chi_symmetry.py`, not just asserted). `chi_transform` is its
    own inverse (mu and colour inversion are both involutions)."""
    mirrored = torch.flip(planes, dims=(2,))  # rank is dim 2 of (N, C, rank, file)
    out = mirrored.clone()
    swap_pairs = (
        *_PIECE_PLANE_PAIRS,
        *_RESERVATION_ACTOR_PLANE_PAIRS,
        *_DFILE_PLANE_PAIRS,
    )
    for a, b in swap_pairs:
        out[:, a], out[:, b] = mirrored[:, b], mirrored[:, a]
    for a, b in _DRANK_PLANE_PAIRS:
        out[:, a], out[:, b] = -mirrored[:, b], -mirrored[:, a]

    out_scalars = scalars.clone()
    for a, b in _SCALAR_PAIRS:
        out_scalars[:, a], out_scalars[:, b] = scalars[:, b], scalars[:, a]
    return out, out_scalars


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
    """Policy-value network; see module docstring."""

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
        self.slot1_heads = nn.ModuleDict(
            {key: nn.Linear(policy_dim, SLOT_SIZE) for key in _COLOR_KEYS.values()}
        )
        # Shared across colours: it embeds a grid index, not a colour-specific
        # quantity, so one table suffices.
        self.a1_embedding = nn.Embedding(SLOT_SIZE, self.config.a1_embed_dim)
        self.slot2_heads = nn.ModuleDict(
            {
                key: nn.Linear(policy_dim + self.config.a1_embed_dim, SLOT_SIZE + 1)
                for key in _COLOR_KEYS.values()
            }
        )

        self.value_conv = nn.Conv2d(
            self.config.filters, self.config.value_channels, 1, bias=False
        )
        self.value_bn = nn.BatchNorm2d(self.config.value_channels)
        self.value_fc1 = nn.Linear(
            self.config.value_channels * 8 * 8, self.config.value_hidden
        )
        self.value_fc2 = nn.Linear(self.config.value_hidden, 1)

        # v3 18a'.5: not a parameter (never trained), but must move with
        # the module across .to(device) calls -- persistent=False since
        # it's a pure function of SLOT_SIZE, nothing to checkpoint.
        self.register_buffer(
            "_mirror_permutation",
            torch.tensor(MIRROR_PERMUTATION, dtype=torch.long),
            persistent=False,
        )

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(slot1_white (N, SLOT_SIZE), slot1_black (N, SLOT_SIZE),
        value (N,), policy_features)``.

        ``policy_features`` is returned so a caller (the search) can compute
        either colour's ``slot2_logits`` for a sampled slot-1 action without
        re-running the trunk -- it is always ``e(s)`` alone, never a
        chi-combination (see the module docstring's slot-2 scope note).

        v3 18a'.5: internally runs the trunk on ``s`` and ``chi(s)``
        together (one doubled-size batched pass, not two), then combines
        ``value`` and both slot-1 heads into the architecturally
        chi-antisymmetric/-symmetric forms the module docstring derives.
        """
        n = planes.shape[0]
        planes_chi, scalars_chi = chi_transform(planes, scalars)
        trunk_both = self._trunk(
            torch.cat((planes, planes_chi), dim=0),
            torch.cat((scalars, scalars_chi), dim=0),
        )
        policy_features_both = self._policy_features(trunk_both)
        value_both = self._value(trunk_both)

        policy_features, policy_features_chi = (
            policy_features_both[:n],
            policy_features_both[n:],
        )
        value = 0.5 * (value_both[:n] - value_both[n:])

        slot1_white_s = self.slot1_heads["white"](policy_features)
        slot1_black_s = self.slot1_heads["black"](policy_features)
        slot1_white_chi = self.slot1_heads["white"](policy_features_chi)
        slot1_black_chi = self.slot1_heads["black"](policy_features_chi)
        mirrored_black_chi = slot1_black_chi.index_select(1, self._mirror_permutation)
        mirrored_white_chi = slot1_white_chi.index_select(1, self._mirror_permutation)
        slot1_white = 0.5 * (slot1_white_s + mirrored_black_chi)
        slot1_black = 0.5 * (slot1_black_s + mirrored_white_chi)
        return slot1_white, slot1_black, value, policy_features

    def slot2_logits(
        self, policy_features: torch.Tensor, a1_indices: torch.Tensor, color: Color
    ) -> torch.Tensor:
        """Slot-2 logits ``(N, SLOT_SIZE + 1)`` for `color`, conditioned on the
        chosen slot-1 grid indices ``a1_indices`` (N,)."""
        embedded = self.a1_embedding(a1_indices)
        head = self.slot2_heads[_COLOR_KEYS[color]]
        logits: torch.Tensor = head(torch.cat((policy_features, embedded), dim=1))
        return logits
