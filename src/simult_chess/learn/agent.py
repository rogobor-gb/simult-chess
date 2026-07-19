"""The learned agent: network-backed SM-MCTS wired into the `Agent` Protocol
(Phase 13b, docs/LEARNING_DESIGN.md §2.5).

`NetworkEvaluator` implements `learn.search.Evaluator` against a trained
`SimultChessNet`: one forward pass per newly-expanded leaf gives both
colours' slot-1 priors and the value (§2.4 -- perfect information, one pass
predicts both colours); `slot2_prior` reuses the cached trunk features
(`policy_features`) for the cheap conditional slot-2 head call the design's
own cost model assumes (§4.3), never re-running the trunk.

`LearnedAgent` runs the search to a simulation budget, then plays: samples
slot-1 from the root's search-derived average strategy (§2.3's "average
strategy... is the move actually played in self-play"), then samples slot-2
from the network's own masked conditional prior given that slot-1 (the same
mechanism the search itself uses internally, §2.3's per-simulation slot-2
step) -- consistent with the scope decision in `learn.search` that slot-2 is
a conditional completion, not an independently regret-matched choice.
Conforms to `agents.base.Agent` so it drops into `referee.match`,
`harness.selfplay`, and the evaluation ladder unchanged.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from simult_chess.core.types import Action, Color, Program, State
from simult_chess.interop.encoding import encode_state
from simult_chess.learn.action_grid import (
    NO_SECOND_INDEX,
    decode_program,
    sample_index,
    slot2_legal_actions,
)
from simult_chess.learn.config import SearchConfig
from simult_chess.learn.net import SimultChessNet, default_device
from simult_chess.learn.search import make_root, run_simulations
from simult_chess.rules.ruleset import RuleSet


def _masked_softmax_dict(
    logits: torch.Tensor, indices: list[int]
) -> dict[int, float]:
    """Softmax of `logits` restricted to `indices`, as an `index -> prob`
    dict. Empty `indices` returns an empty dict (no legal choice at this
    slot -- callers handle that, e.g. slot-2's `NO_SECOND_INDEX` case)."""
    if not indices:
        return {}
    gathered = logits[indices]
    probs = torch.softmax(gathered, dim=0)
    return {index: float(p) for index, p in zip(indices, probs, strict=True)}


@dataclass
class NetworkEvaluator:
    """`learn.search.Evaluator` backed by a trained `SimultChessNet`. Puts
    the network in `eval()` mode at construction (BatchNorm running stats,
    not batch statistics -- correct for single-position inference); callers
    that need training-mode forward passes use the network directly, not
    through this evaluator."""

    net: SimultChessNet
    device: torch.device | None = None

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = default_device()
        self.net.to(self.device)
        self.net.eval()

    def evaluate_leaf(
        self, state: State, ruleset: RuleSet
    ) -> tuple[float, dict[int, float], dict[int, float], object]:
        from simult_chess.learn.action_grid import slot1_legal_actions

        planes_np, scalars_np = encode_state(state, ruleset)
        planes = torch.from_numpy(np.expand_dims(planes_np, 0)).to(self.device)
        scalars = torch.from_numpy(np.expand_dims(scalars_np, 0)).to(self.device)
        with torch.no_grad():
            slot1_white, slot1_black, value, policy_features = self.net(
                planes, scalars
            )
        white_legal = list(slot1_legal_actions(state, Color.WHITE, ruleset))
        black_legal = list(slot1_legal_actions(state, Color.BLACK, ruleset))
        prior_white = _masked_softmax_dict(slot1_white[0].cpu(), white_legal)
        prior_black = _masked_softmax_dict(slot1_black[0].cpu(), black_legal)
        return float(value.item()), prior_white, prior_black, policy_features

    def slot2_prior(
        self,
        context: object,
        color: Color,
        state: State,
        ruleset: RuleSet,
        first_index: int,
        first: Action,
    ) -> dict[int, float]:
        assert isinstance(context, torch.Tensor)
        actions, single_legal = slot2_legal_actions(state, color, ruleset, first)
        indices = list(actions) + ([NO_SECOND_INDEX] if single_legal else [])
        a1_index = torch.tensor([first_index], device=self.device)
        with torch.no_grad():
            logits = self.net.slot2_logits(context, a1_index, color)
        return _masked_softmax_dict(logits[0].cpu(), indices)


@dataclass
class LearnedAgent:
    """Network-backed SM-MCTS agent, conforming to `agents.base.Agent`
    (`__call__(state, color, ruleset, rng) -> Program`)."""

    net: SimultChessNet
    search_config: SearchConfig = SearchConfig()
    device: torch.device | None = None

    def __post_init__(self) -> None:
        self.evaluator = NetworkEvaluator(self.net, self.device)

    def __call__(
        self, state: State, color: Color, ruleset: RuleSet, rng: random.Random
    ) -> Program:
        root = make_root(state)
        run_simulations(
            root,
            ruleset,
            self.evaluator,
            self.search_config.simulations,
            rng,
            prior_weight=self.search_config.prior_weight,
        )
        stats = root.white if color is Color.WHITE else root.black
        assert stats is not None, "search must expand the root before playing"
        a1_index = sample_index(stats.average_strategy(), rng)
        first = stats.actions[a1_index]

        slot2_dist = self.evaluator.slot2_prior(
            root.context, color, state, ruleset, a1_index, first
        )
        a2_index = sample_index(slot2_dist, rng)
        return decode_program(first, a2_index, state, color, ruleset)
