"""LIGHT-profile configuration (Phase 13b, docs/LEARNING_DESIGN.md §4).

Defaults are fixed by the on-device profiling in the design doc (Apple M3,
PyTorch MPS): residual trunk B=6 x F=64, M=128 simulations/move. Kept as a
plain dataclass (no torch) so it can be imported anywhere and logged into the
report; the network and search read it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NetConfig:
    """Residual policy-value network geometry (LIGHT: B=6, F=64, §3.4)."""

    num_planes: int = 21  # interop.encoding.NUM_PLANES
    num_scalars: int = 7  # interop.encoding.NUM_SCALARS
    residual_blocks: int = 6  # LIGHT B
    filters: int = 64  # LIGHT F
    policy_channels: int = 32  # 1x1 conv channels feeding the policy heads
    value_channels: int = 1  # 1x1 conv to one plane, then MLP (§3.4)
    value_hidden: int = 64
    a1_embed_dim: int = 32  # slot-1 action embedding conditioning slot-2


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """SM-MCTS parameters (LIGHT: M=128 simulations/move, §4.3)."""

    simulations: int = 128
    # "regret_matching" (optimistic RM+, v3 18a'.4) is the default in-tree
    # rule; "hedge"/"optimistic_hedge" are configurable alternatives (also
    # 18a'.4) -- see learn.search._regret_matching_strategy/_hedge_
    # distribution. Comparing them properly on real fixtures is 18d.2's job.
    selection: str = "regret_matching"
    prior_weight: float = 1.0  # blends the network prior into the RM initializer
    temperature: float = 1.0  # early-game sampling temperature for self-play
    epsilon_floor: float = 0.02  # v3 18a'.3: explicit per-action sampling-probability
    # floor, turning "every action visited infinitely often" (SM-MCTS convergence
    # theorem hypothesis H3) from an empirical observation into a guarantee --
    # see learn.search._regret_matching_strategy's docstring.
