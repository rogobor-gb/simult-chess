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
    node_solver: str = "slot1"
    # v3 18c.1: "slot1" (default) is learn.search's slot-1-only SM-MCTS,
    # unchanged; "row_sketch" opts a caller (currently only learn.selfplay)
    # into learn.row_sketch's program-pool node solver instead -- see that
    # module's docstring for what changes. Every existing caller/test keeps
    # today's exact behaviour by not setting this.
    pool_size: int = 12  # row_sketch's own k -- ignored unless node_solver="row_sketch"
    pool_seed_size: int = 6  # row_sketch's own k1 -- ditto
    # v3 18d.1/18d.3: row_sketch._mmd_update's own parameters, ignored unless
    # node_solver="row_sketch" and selection="mmd". Defaults are the values
    # validated against the Matching-Pennies dodge fixture (row_sketch.
    # _mmd_update's own docstring). Exposed here (rather than left as
    # run_simulations' own function defaults) so a caller can vary alpha0
    # per agent -- e.g. 18d.3's tau-ladder (tau = 1/mmd_alpha0).
    mmd_eta: float = 1.0
    mmd_alpha0: float = 0.5
    mmd_magnet_period: int = 50
