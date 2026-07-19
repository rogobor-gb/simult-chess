"""SM-MCTS with regret-matching in-tree selection (Phase 13b, design §2).

Per §1.2, a pure-strategy (max-operator) backup is inadmissible: the spec's
Matching-Pennies king-dodge subgame has no pure equilibrium, so the backup
operator must be matrix-game minimax, approximated here by **regret
matching** (Hart & Mas-Colell 2000) run as a per-node, per-colour no-regret
bandit whose **average strategy** converges to the stage-game value (Lanctot
et al. 2013/2014) -- this module's correctness claim, proven by
``tests/unit/test_search_matching_pennies.py``.

**Scope decision (flagged for review): only slot-1 is regret-matched.**
Regret matching's bandit update needs, for the *whole* action set at a node,
a value estimate for switching to any alternative action (the "counterfactual
value" of §2.3 point 4) -- tractable for slot-1 (O(pool) actions, matching
the design's own §4.3/§4.4 cost model) but not for the full program space
(slot-1 x slot-2 is the ~8x10^7-pair space §3.3 explicitly rejects enumerating
for the *policy head*, and the same cardinality argument applies to a bandit
over it). Slot-2 is instead **sampled directly from the network's masked,
per-node-cached conditional prior** given the sampled slot-1 action -- a
conditional *completion* of the chosen first action rather than an
independently-mixed strategic choice. This keeps every simulation at the
O(pool) mask cost §4.3 measures (~2.2-2.5 ms/sim), and the correctness claim
this module proves (mixed-equilibrium convergence, e.g. Matching Pennies) is
itself a single-action-level property in the spec's own worked example --
fully covered by regret-matching the slot-1 marginal.

Decoupled, zero-sum backup: one signed scalar per simulation, White
maximizes it (its own utility) and Black minimizes it (equivalently
maximizes its own utility, the negation) -- exactly §2.3 point 4.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol

from simult_chess.core.legality import is_legal_program
from simult_chess.core.phi import phi
from simult_chess.core.types import Action, Color, Program, State
from simult_chess.learn.action_grid import (
    NO_SECOND_INDEX,
    sample_index,
    slot1_legal_actions,
    slot2_legal_actions,
)
from simult_chess.rules.ruleset import RuleSet

_OUTCOME_VALUE: dict[str, float] = {
    "white_wins": 1.0,
    "black_wins": -1.0,
    "draw": 0.0,
}


class Evaluator(Protocol):
    """Supplies leaf values and action priors -- the search's only dependency
    on the network, so the search's correctness is testable without torch
    (see the synthetic evaluator in the Matching-Pennies test)."""

    def evaluate_leaf(
        self, state: State, ruleset: RuleSet
    ) -> tuple[float, dict[int, float], dict[int, float], object]:
        """Return ``(value, slot1_prior_white, slot1_prior_black, context)``:
        ``value`` is White's utility estimate in [-1, 1]; each prior is a
        probability distribution over that colour's legal slot-1 grid
        indices (from ``slot1_legal_actions``); ``context`` is opaque
        evaluator state (e.g. cached network features) threaded back into
        ``slot2_prior``."""
        ...

    def slot2_prior(
        self,
        context: object,
        color: Color,
        state: State,
        ruleset: RuleSet,
        first_index: int,
        first: Action,
    ) -> dict[int, float]:
        """Probability distribution over `color`'s legal slot-2 completions of
        `first` (from ``slot2_legal_actions``), keys including
        ``NO_SECOND_INDEX`` when the single-action program is legal."""
        ...


@dataclass
class _ColorStats:
    actions: dict[int, Action]
    prior: dict[int, float]
    regret: dict[int, float] = field(default_factory=dict)
    strategy_sum: dict[int, float] = field(default_factory=dict)
    q: dict[int, float] = field(default_factory=dict)
    visits: dict[int, int] = field(default_factory=dict)
    node_visits: int = 0
    """Total simulations that have reached this node (not per-action) --
    used to fade the prior anchor (see `_regret_matching_strategy`). Grows
    monotonically regardless of regret sign, unlike `sum(positive(regret))`,
    which can stay near zero indefinitely for value-tied actions (their
    pairwise regrets cancel) and would otherwise pin the blended strategy at
    the prior forever (found via the Matching-Pennies convergence test:
    with several value-tied "stay" arms, the fade denominator based on
    positive regret barely grew, and the strategy never left ~uniform)."""

    def __post_init__(self) -> None:
        for index in self.actions:
            self.regret.setdefault(index, 0.0)
            self.strategy_sum.setdefault(index, 0.0)
            self.q.setdefault(index, 0.0)
            self.visits.setdefault(index, 0)

    def average_strategy(self) -> dict[int, float]:
        """:math:`\\bar\\sigma / \\sum\\bar\\sigma` -- the search-derived mixed
        strategy (§2.3): what self-play samples from and the policy target."""
        total = sum(self.strategy_sum.values())
        if total <= 0.0:
            n = len(self.actions)
            return dict.fromkeys(self.actions, 1.0 / n) if n else {}
        return {a: s / total for a, s in self.strategy_sum.items()}


@dataclass
class SearchNode:
    state: State
    is_terminal: bool
    terminal_value: float | None = None
    white: _ColorStats | None = None
    black: _ColorStats | None = None
    context: object | None = None
    children: dict[
        tuple[tuple[int, int | None], tuple[int, int | None]], SearchNode
    ] = field(default_factory=dict)

    @property
    def is_expanded(self) -> bool:
        return self.white is not None


def make_root(state: State) -> SearchNode:
    """A fresh, unexpanded root node for `state` (assumed non-terminal --
    callers must not search from a terminal state)."""
    return SearchNode(state=state, is_terminal=False)


def _regret_matching_strategy(
    stats: _ColorStats, prior_weight: float
) -> dict[int, float]:
    """Positive-regret normalization (uniform if none positive), blended with
    the network prior as a decaying anchor (design §2.3): the anchor's
    influence is O(prior_weight) against `sqrt(node_visits)`, so it dominates
    early and fades as evidence accumulates -- using `node_visits` (not
    `sum(positive(regret))`) as the fade weight is required, not cosmetic:
    for value-tied actions, positive and negative regret cancel over time,
    so sum(positive(regret)) can stay near zero indefinitely even after
    thousands of visits, which would pin the blended strategy at the prior
    forever.

    The **square root** of `node_visits` (not `node_visits` itself, matching
    PUCT's own exploration term, e.g. AlphaZero's `c*sqrt(N)/(1+n(a))`) is
    also required, not cosmetic: pure regret-matching sampling has no built-in
    optimism-under-uncertainty guarantee, so an action whose regret happens to
    sit at/near zero (RM+ clips negative regret to zero every update) gets
    exactly zero weight from the positive-regret component; a *linearly*
    decaying prior anchor makes its only remaining exploration probability
    shrink to an expected sample count under 1 within a few thousand
    iterations, so an action that draws unlucky early can go permanently
    unexplored even though it would reveal itself as excellent the moment
    it's tried (found via the Matching-Pennies convergence test itself, seed-
    dependent: one attacker jump went completely unvisited across 6000 sims
    under the linear-decay anchor, leaving the defender's best response
    undiscovered)."""
    positive = {a: max(r, 0.0) for a, r in stats.regret.items()}
    total = sum(positive.values())
    rm = (
        {a: p / total for a, p in positive.items()}
        if total > 0.0
        else dict.fromkeys(stats.actions, 1.0 / len(stats.actions))
    )
    if prior_weight <= 0.0:
        return rm
    prior_total = sum(stats.prior.values())
    normalized_prior = (
        {a: stats.prior.get(a, 0.0) / prior_total for a in stats.actions}
        if prior_total > 0.0
        else dict.fromkeys(stats.actions, 1.0 / len(stats.actions))
    )
    confidence = stats.node_visits**0.5
    denom = confidence + prior_weight
    return {
        a: (rm[a] * confidence + prior_weight * normalized_prior[a]) / denom
        for a in stats.actions
    }


def _update(stats: _ColorStats, taken: int, value: float) -> None:
    """Running-mean Q update for the taken action, then a **Regret Matching+**
    (Tammelin 2014) update for every other action, comparing its historical
    estimate against **this iteration's fresh realized value** `value`.

    `value` (not `Q[taken]` post-update) is the correct baseline: sampled
    regret's standard unbiased-estimator trick is that the realized value of
    actually playing `taken` this iteration is itself an unbiased sample of
    "my own strategy's value against the opponent's t-th sample" -- using the
    *smoothed* running average of the taken action instead (an earlier,
    buggy version of this update) mixes in stale history from other, earlier
    iterations, which is a different (and wrong) quantity. The taken action
    gets zero regret contribution by construction (regret for what I actually
    did, against what I actually did, is definitionally zero) -- not
    `Q[taken] - value`, which is nonzero whenever the running average hasn't
    fully converged to the latest sample.

    RM+'s per-update clip (`max(0, ...)`, not just at strategy-computation
    time) is required, not cosmetic: plain accumulated regret let an action
    that looked bad *only while the opponent's now-stale strategy favored
    punishing it* fall to a deep negative regret that then took thousands of
    further iterations to climb back out of, even after the opponent's own
    strategy had already shifted and that action was genuinely best again.
    Both of these were found and fixed via the Matching-Pennies convergence
    test (tests/unit/test_search_matching_pennies.py): first the stuck-regret
    pathology, then this baseline error, which independently reproduced a
    milder version of the same "stuck, wrong equilibrium" symptom."""
    stats.visits[taken] += 1
    n = stats.visits[taken]
    stats.q[taken] += (value - stats.q[taken]) / n
    for a in stats.actions:
        if a == taken:
            continue
        stats.regret[a] = max(0.0, stats.regret[a] + (stats.q[a] - value))


def _decode_program(
    first: Action, second_index: int, slot2_actions: dict[int, Action]
) -> Program:
    if second_index == NO_SECOND_INDEX:
        return (first,)
    return (first, slot2_actions[second_index])


def _simulate(
    node: SearchNode,
    ruleset: RuleSet,
    evaluator: Evaluator,
    rng: random.Random,
    prior_weight: float,
    depth: int = 0,
    max_depth: int | None = None,
) -> float:
    if node.is_terminal:
        assert node.terminal_value is not None
        return node.terminal_value

    if max_depth is not None and depth >= max_depth:
        # Depth-limited bootstrap: evaluate as an immediate leaf without
        # expanding regret-matching state, so the node is re-evaluated fresh
        # (not cached) on every future visit at the cutoff. Used to isolate a
        # single decision phase for the equilibrium-convergence proof
        # (test_search_matching_pennies.py); real self-play (Stage D) passes
        # `max_depth=None` and always plays to a genuine game terminal.
        value, _, _, _ = evaluator.evaluate_leaf(node.state, ruleset)
        return value

    if not node.is_expanded:
        value, prior_white, prior_black, context = evaluator.evaluate_leaf(
            node.state, ruleset
        )
        slot1_white = slot1_legal_actions(node.state, Color.WHITE, ruleset)
        slot1_black = slot1_legal_actions(node.state, Color.BLACK, ruleset)
        node.white = _ColorStats(actions=slot1_white, prior=prior_white)
        node.black = _ColorStats(actions=slot1_black, prior=prior_black)
        node.context = context
        return value

    assert node.white is not None and node.black is not None
    node.white.node_visits += 1
    node.black.node_visits += 1
    sigma_white = _regret_matching_strategy(node.white, prior_weight)
    sigma_black = _regret_matching_strategy(node.black, prior_weight)
    for a, p in sigma_white.items():
        node.white.strategy_sum[a] += p
    for a, p in sigma_black.items():
        node.black.strategy_sum[a] += p

    a1_white = sample_index(sigma_white, rng)
    a1_black = sample_index(sigma_black, rng)
    first_white = node.white.actions[a1_white]
    first_black = node.black.actions[a1_black]

    slot2_prior_white = evaluator.slot2_prior(
        node.context, Color.WHITE, node.state, ruleset, a1_white, first_white
    )
    slot2_prior_black = evaluator.slot2_prior(
        node.context, Color.BLACK, node.state, ruleset, a1_black, first_black
    )
    a2_white = sample_index(slot2_prior_white, rng)
    a2_black = sample_index(slot2_prior_black, rng)

    slot2_actions_white, _ = slot2_legal_actions(
        node.state, Color.WHITE, ruleset, first_white
    )
    slot2_actions_black, _ = slot2_legal_actions(
        node.state, Color.BLACK, ruleset, first_black
    )
    program_white = _decode_program(first_white, a2_white, slot2_actions_white)
    program_black = _decode_program(first_black, a2_black, slot2_actions_black)
    assert is_legal_program(node.state, program_white, Color.WHITE, ruleset)
    assert is_legal_program(node.state, program_black, Color.BLACK, ruleset)

    signature = (
        (a1_white, None if a2_white == NO_SECOND_INDEX else a2_white),
        (a1_black, None if a2_black == NO_SECOND_INDEX else a2_black),
    )
    child = node.children.get(signature)
    if child is None:
        result = phi(node.state, program_white, program_black, ruleset)
        if result.outcome != "ongoing":
            child = SearchNode(
                state=result.state,
                is_terminal=True,
                terminal_value=_OUTCOME_VALUE[result.outcome],
            )
        else:
            child = SearchNode(state=result.state, is_terminal=False)
        node.children[signature] = child

    value = _simulate(
        child, ruleset, evaluator, rng, prior_weight, depth + 1, max_depth
    )
    _update(node.white, a1_white, value)
    _update(node.black, a1_black, -value)
    return value


def run_simulations(
    root: SearchNode,
    ruleset: RuleSet,
    evaluator: Evaluator,
    n_simulations: int,
    rng: random.Random,
    *,
    prior_weight: float = 1.0,
    max_depth: int | None = None,
) -> None:
    """Run `n_simulations` SM-MCTS simulations from `root` in place.

    `max_depth` bounds recursion depth for testing/analysis (e.g. isolating
    one decision phase against a known matrix game); real self-play leaves it
    `None` and always plays to a genuine game terminal.
    """
    for _ in range(n_simulations):
        _simulate(root, ruleset, evaluator, rng, prior_weight, 0, max_depth)
