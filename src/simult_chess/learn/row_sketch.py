"""The row-sketch unified node solver (Phase 18c.1,
docs/LEARNING_ROADMAP_v3.md). A parallel node representation to
`learn.search`'s slot-1-only SM-MCTS, not a modification of it -- every
existing `search.py`/`selfplay.py`/`train.py` test keeps passing
unmodified; this module is validated standalone before any pipeline
wiring.

**Why a new node representation, not a patch.** `learn.search` regret-
matches only *slot-1 grid indices*, with slot-2 sampled fresh each round
from the network's own conditional prior and never regret-matched at all
(H2: slot-2 runs no learning algorithm). Its per-round regret update also
compares the taken action's *realized* value against a `Q[a]` running-
mean baseline for every other action (18a'.2's decaying-step-size
mitigation) -- an estimate of an action's payoff against the *average
historical* opponent, not the one actually sampled this round (H4). Both
are consequences of the same structural choice: regret-matching over
*single actions*, with the opponent's realized program folded into a
scalar baseline instead of evaluated fresh.

**The fix.** Track a pool of `k` **full programs** per colour (seeded by
the network's own slot-1-prior x slot-2-conditional-prior product, via
`seed_program_pool`), not slot-1 actions alone -- so a "program" already
carries both slots, and regret-matching the pool regret-matches slot-2
for free. Per simulation: sample the opponent's program from its current
strategy over its own pool, then evaluate `u(a, b_t) = r + V(Phi(s,a,b_t))`
for *every* `a` in my own pool against that one fixed `b_t` (k Phi calls)
-- the *exact* instantaneous regret vector this round, not a stale
running-mean baseline. `V(child)`: a fresh, cheap network leaf evaluation
if the child hasn't been visited before (matching the roadmap's own "k Phi
calls, ~2.2ms" cost estimate -- one forward pass per pool entry, not a
full recursive re-simulation of all k candidates, which would multiply
cost by up to k at every depth reached that round); a previously-expanded
child's own running value estimate otherwise.

`val_tau`/the pool-level equilibrium (the value and policy targets 18c.1
promises) are computed post-simulation only, via `read_out` --
`solver.entropy.solve_max_entropy_equilibrium`'s annealed `scipy.optimize.
minimize` solve is far too slow to call every round.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from simult_chess.core.legality import is_legal_program
from simult_chess.core.phi import phi
from simult_chess.core.types import Action, Castle, Color, Move, Program, State
from simult_chess.learn.action_grid import (
    decode_program,
    sample_index,
    slot1_legal_actions,
)
from simult_chess.learn.search import (
    Evaluator,
    _hedge_distribution,
    _rm_plus_distribution,
)
from simult_chess.rules.ruleset import RuleSet
from simult_chess.solver.entropy import solve_max_entropy_equilibrium

FloatArray = npt.NDArray[np.float64]

_OUTCOME_VALUE: dict[str, float] = {
    "white_wins": 1.0,
    "black_wins": -1.0,
    "draw": 0.0,
}


def seed_program_pool(
    evaluator: Evaluator,
    context: object,
    state: State,
    color: Color,
    ruleset: RuleSet,
    prior: dict[int, float],
    actions: dict[int, Action],
    *,
    k: int = 12,
    k1: int = 6,
) -> tuple[Program, ...]:
    """A pool of up to `k` full programs for `color` at `state`, seeded by
    the network's own slot-1-prior x slot-2-conditional-prior product --
    18c.1's replacement for `solver.supports.enumerate_support`'s capture-
    value heuristic (that pool is seeded for a *different* purpose, an
    offline restricted-support solve; this one seeds in-tree regret
    matching, so it must track what the network actually favours, not a
    hand-picked material heuristic).

    Ranks every (slot-1, slot-2) combination reachable from the top `k1`
    slot-1 actions by prior, by joint probability `p1 * p2`, keeps the top
    `k` legal ones. Guarantees at least one legal Move/Castle-headed
    program survives (L2, spec S4.4.2: a program must contain a Move/
    Castle whenever one is legally available) -- the same fallback
    `enumerate_support` uses, for the same reason: pure prior-ranking could
    otherwise truncate every Move/Castle candidate out if the network's
    top-`k1` slot-1 picks happen to be all Reserve/Cancel starters.
    """
    if sum(prior.values()) <= 0.0:
        # No informative prior (e.g. a synthetic evaluator used to isolate
        # the regret-matching machinery itself, `search.py`'s own
        # `_UniformPriorEvaluator`) -- fall back to uniform over every
        # legal slot-1 action, the same convention `search._ColorStats`'s
        # own blending already uses for an empty/zero-sum prior.
        prior = dict.fromkeys(actions, 1.0)
    top_slot1 = sorted(prior.items(), key=lambda kv: kv[1], reverse=True)[:k1]
    candidates: list[tuple[float, Program]] = []
    seen: set[Program] = set()
    for a1_index, p1 in top_slot1:
        first = actions[a1_index]
        slot2_dist = evaluator.slot2_prior(
            context, color, state, ruleset, a1_index, first
        )
        for a2_index, p2 in slot2_dist.items():
            program = decode_program(first, a2_index, state, color, ruleset)
            if program in seen or not is_legal_program(state, program, color, ruleset):
                continue
            seen.add(program)
            candidates.append((p1 * p2, program))
    candidates.sort(key=lambda c: c[0], reverse=True)
    pool = [program for _, program in candidates[:k]]

    if not any(
        isinstance(action, Move | Castle) for program in pool for action in program
    ):
        for action in actions.values():
            if isinstance(action, Move | Castle) and is_legal_program(
                state, (action,), color, ruleset
            ):
                if pool:
                    pool[-1] = (action,)
                else:
                    pool.append((action,))
                break

    return tuple(pool)


@dataclass
class _ProgramStats:
    """Like `search._ColorStats`, but keyed by **pool index** over full
    `Program`s instead of slot-1 grid indices -- no `q`/`visits` baseline
    fields, since the row-sketch replaces that mechanism entirely with
    exact per-round evaluation."""

    programs: tuple[Program, ...]
    regret: dict[int, float] = field(default_factory=dict)
    strategy_sum: dict[int, float] = field(default_factory=dict)
    last_regret_increment: dict[int, float] = field(default_factory=dict)
    node_visits: int = 0

    def __post_init__(self) -> None:
        for i in range(len(self.programs)):
            self.regret.setdefault(i, 0.0)
            self.strategy_sum.setdefault(i, 0.0)
            self.last_regret_increment.setdefault(i, 0.0)

    def average_strategy(self) -> dict[int, float]:
        total = sum(self.strategy_sum.values())
        if total <= 0.0:
            n = len(self.programs)
            return dict.fromkeys(range(n), 1.0 / n) if n else {}
        return {i: s / total for i, s in self.strategy_sum.items()}


def _pool_strategy(
    stats: _ProgramStats, epsilon: float, selection: str
) -> tuple[dict[int, float], dict[int, float]]:
    """Like `search._regret_matching_strategy`, but without the network-
    prior re-blend: the pool itself is already prior-seeded (top-k by
    `seed_program_pool`), so every candidate is already "reasonable" by
    construction -- blending the prior in *again* at the sampling-
    distribution level would be redundant. `sigma` is `rm` (optimistic RM+
    by default, or Hedge/optimistic Hedge -- same `selection` values as
    `search`) floored at `epsilon` (v3 18a'.3's guarantee, carried over
    unchanged: every pool entry visited infinitely often)."""
    n = len(stats.programs)
    if selection == "hedge":
        rm = _hedge_distribution(stats, optimistic=False)
    elif selection == "optimistic_hedge":
        rm = _hedge_distribution(stats, optimistic=True)
    else:
        rm = _rm_plus_distribution(stats)
    if epsilon <= 0.0 or not n:
        return rm, rm
    uniform = 1.0 / n
    sigma = {i: (1.0 - epsilon) * rm[i] + epsilon * uniform for i in range(n)}
    return rm, sigma


@dataclass
class RowSketchNode:
    state: State
    is_terminal: bool
    terminal_value: float | None = None
    white: _ProgramStats | None = None
    black: _ProgramStats | None = None
    context: object | None = None
    value_estimate: float | None = None
    """The most-recently-observed value of this node -- either the leaf
    network estimate (set once, at expansion) or the return of the last
    `_simulate_row_sketch` recursion into it (refined thereafter). What
    `_peek_value` returns for an already-expanded child instead of paying
    for a fresh network call."""
    stage_sketch: FloatArray | None = None
    """`k_white x k_black`, the most-recently-observed `u(i,j)` for each
    pool cell -- `NaN` where never yet sampled. Row/column entries fill in
    every round the row-sketch touches this node; `read_out` fills any
    still-`NaN` cells (cheaply, once) before solving."""
    children: dict[tuple[int, int], RowSketchNode] = field(default_factory=dict)

    @property
    def is_expanded(self) -> bool:
        return self.white is not None


def make_root(state: State) -> RowSketchNode:
    """A fresh, unexpanded root node for `state` (assumed non-terminal --
    callers must not search from a terminal state)."""
    return RowSketchNode(state=state, is_terminal=False)


def _peek_value(child: RowSketchNode, evaluator: Evaluator, ruleset: RuleSet) -> float:
    """White's utility at `child` -- cheap by construction (never a
    recursive re-simulation): the child's own running estimate if it has
    one, else one fresh network forward pass."""
    if child.is_terminal:
        assert child.terminal_value is not None
        return child.terminal_value
    if child.value_estimate is not None:
        return child.value_estimate
    value, _, _, _ = evaluator.evaluate_leaf(child.state, ruleset)
    return value


def _get_or_make_child(
    node: RowSketchNode, i: int, j: int, ruleset: RuleSet
) -> RowSketchNode:
    key = (i, j)
    child = node.children.get(key)
    if child is not None:
        return child
    assert node.white is not None and node.black is not None
    program_white = node.white.programs[i]
    program_black = node.black.programs[j]
    result = phi(node.state, program_white, program_black, ruleset)
    if result.outcome != "ongoing":
        child = RowSketchNode(
            state=result.state,
            is_terminal=True,
            terminal_value=_OUTCOME_VALUE[result.outcome],
        )
    else:
        child = RowSketchNode(state=result.state, is_terminal=False)
    node.children[key] = child
    return child


def _simulate_row_sketch(
    node: RowSketchNode,
    ruleset: RuleSet,
    evaluator: Evaluator,
    rng: random.Random,
    epsilon: float,
    selection: str,
    depth: int,
    max_depth: int | None,
    *,
    pool_size: int,
    pool_seed_size: int,
) -> float:
    if node.is_terminal:
        assert node.terminal_value is not None
        return node.terminal_value

    if max_depth is not None and depth >= max_depth:
        value, _, _, _ = evaluator.evaluate_leaf(node.state, ruleset)
        return value

    if not node.is_expanded:
        value, prior_white, prior_black, context = evaluator.evaluate_leaf(
            node.state, ruleset
        )
        slot1_white = slot1_legal_actions(node.state, Color.WHITE, ruleset)
        slot1_black = slot1_legal_actions(node.state, Color.BLACK, ruleset)
        pool_white = seed_program_pool(
            evaluator, context, node.state, Color.WHITE, ruleset, prior_white,
            slot1_white, k=pool_size, k1=pool_seed_size,
        )
        pool_black = seed_program_pool(
            evaluator, context, node.state, Color.BLACK, ruleset, prior_black,
            slot1_black, k=pool_size, k1=pool_seed_size,
        )
        node.white = _ProgramStats(programs=pool_white)
        node.black = _ProgramStats(programs=pool_black)
        node.context = context
        node.stage_sketch = np.full((len(pool_white), len(pool_black)), np.nan)
        node.value_estimate = value
        return value

    assert (
        node.white is not None
        and node.black is not None
        and node.stage_sketch is not None
    )
    node.white.node_visits += 1
    node.black.node_visits += 1

    rm_white, sigma_white = _pool_strategy(node.white, epsilon, selection)
    rm_black, sigma_black = _pool_strategy(node.black, epsilon, selection)

    t = node.white.node_visits
    for i, p in rm_white.items():
        node.white.strategy_sum[i] += t * p
    for j, p in rm_black.items():
        node.black.strategy_sum[j] += t * p

    i_t = sample_index(sigma_white, rng)
    j_t = sample_index(sigma_black, rng)

    n_white = len(node.white.programs)
    n_black = len(node.black.programs)

    # White's row sketch: u(i, j_t) for every i, Black's sampled column
    # fixed -- the exact instantaneous regret vector for White this round.
    u_row: dict[int, float] = {}
    for i in range(n_white):
        child = _get_or_make_child(node, i, j_t, ruleset)
        u = _peek_value(child, evaluator, ruleset)
        u_row[i] = u
        node.stage_sketch[i, j_t] = u

    # Black's column sketch: u(i_t, j) for every j, White's sampled row
    # fixed -- reuses the (i_t, j_t) cell already computed above.
    u_col: dict[int, float] = {}
    for j in range(n_black):
        if j == j_t:
            u_col[j] = u_row[i_t]
            continue
        child = _get_or_make_child(node, i_t, j, ruleset)
        u = _peek_value(child, evaluator, ruleset)
        u_col[j] = u
        node.stage_sketch[i_t, j] = u

    clip_at_zero = selection == "regret_matching"
    baseline_white = u_row[i_t]
    for i in range(n_white):
        g = 0.0 if i == i_t else u_row[i] - baseline_white
        updated = node.white.regret[i] + g
        node.white.regret[i] = max(0.0, updated) if clip_at_zero else updated
        node.white.last_regret_increment[i] = g

    # Black maximizes -u; regret for switching to j is the negation of
    # White's own row-sketch comparison (zero-sum).
    baseline_black = u_col[j_t]
    for j in range(n_black):
        g = 0.0 if j == j_t else -(u_col[j] - baseline_black)
        updated = node.black.regret[j] + g
        node.black.regret[j] = max(0.0, updated) if clip_at_zero else updated
        node.black.last_regret_increment[j] = g

    child = node.children[(i_t, j_t)]
    value = _simulate_row_sketch(
        child, ruleset, evaluator, rng, epsilon, selection, depth + 1, max_depth,
        pool_size=pool_size, pool_seed_size=pool_seed_size,
    )
    child.value_estimate = value
    node.stage_sketch[i_t, j_t] = value
    return value


def run_simulations(
    root: RowSketchNode,
    ruleset: RuleSet,
    evaluator: Evaluator,
    n_simulations: int,
    rng: random.Random,
    *,
    epsilon: float = 0.02,
    selection: str = "regret_matching",
    max_depth: int | None = None,
    pool_size: int = 12,
    pool_seed_size: int = 6,
) -> None:
    """Run `n_simulations` row-sketch simulations from `root` in place.

    `pool_size` (k) / `pool_seed_size` (k1) are `seed_program_pool`'s own
    parameters, threaded through so every newly-expanded node in this
    search uses the same pool geometry. `max_depth` matches `learn.search.
    run_simulations`'s own testing/analysis hook."""
    for _ in range(n_simulations):
        _simulate_row_sketch(
            root, ruleset, evaluator, rng, epsilon, selection, 0, max_depth,
            pool_size=pool_size, pool_seed_size=pool_seed_size,
        )


@dataclass(frozen=True, slots=True)
class RowSketchReadout:
    """18c.1's value and policy targets, read off a searched root (or any
    expanded node) *after* simulations finish -- never per-round, since
    `solve_max_entropy_equilibrium`'s annealed solve is too slow for that."""

    value: float
    row_strategy: FloatArray
    col_strategy: FloatArray
    row_entropy: float
    col_entropy: float


def read_out(
    node: RowSketchNode, evaluator: Evaluator, ruleset: RuleSet
) -> RowSketchReadout:
    """`val_tau` and the pool-level max-entropy equilibrium of `node`'s
    accumulated stage-matrix sketch -- 18c.1's value and policy targets.
    Any pool cell the search never happened to sample (`NaN` in `stage_
    sketch`) is filled with one fresh, cheap network leaf evaluation
    first, so the matrix fed to the solver is always fully populated."""
    assert (
        node.white is not None
        and node.black is not None
        and node.stage_sketch is not None
    )
    matrix = node.stage_sketch.copy()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isnan(matrix[i, j]):
                child = _get_or_make_child(node, i, j, ruleset)
                matrix[i, j] = _peek_value(child, evaluator, ruleset)

    equilibrium = solve_max_entropy_equilibrium(matrix)
    return RowSketchReadout(
        value=equilibrium.value,
        row_strategy=equilibrium.row_strategy,
        col_strategy=equilibrium.col_strategy,
        row_entropy=equilibrium.row_entropy,
        col_entropy=equilibrium.col_entropy,
    )
