"""Exact restricted-support NashConv on the Phase-10 M5 fixtures (Phase 13b,
design §6.3).

On a chi-symmetric fixture with a chi-closed restricted support (Black's
support is White's support mirrored, same index order -- the exact
construction `tests/property/test_m5_symmetric_value.py` uses), `solver.lp`
gives the exact matrix-game value. Against that value, this module computes
the learned agent's (search + network, not the raw network prior alone --
the design's own "learned policy") NashConv on the restricted stage game: an
**exact** number on a small, controlled set, not a bound. M5's own proven
val=0 on these fixtures is a free sanity anchor (design §6.3): a well-trained
value head should read ~0 there too.

Like `learn.evaluate`, this module is allowed to import `simult_chess.solver`
(numpy/scipy, no torch) -- the torch quarantine only covers
`core`/`rules`/`referee`/agents/harness.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

from simult_chess.core.collision import mirror_program
from simult_chess.core.phi import phi
from simult_chess.core.types import Color, Program, State
from simult_chess.interop.openspiel_adapter import enumerate_legal_programs
from simult_chess.learn.action_grid import NO_SECOND_INDEX, encode_action
from simult_chess.learn.agent import NetworkEvaluator
from simult_chess.learn.config import SearchConfig
from simult_chess.learn.net import SimultChessNet
from simult_chess.learn.row_sketch import RowSketchNode
from simult_chess.learn.search import make_root, run_simulations
from simult_chess.rules.ruleset import RuleSet
from simult_chess.solver.exact import solve_exact
from simult_chess.solver.lp import solve_zero_sum
from simult_chess.solver.stage_matrix import build_stage_matrix
from simult_chess.solver.supports import enumerate_support

_FloatArray = npt.NDArray[np.float64]

_TERMINAL_PAYOFF: dict[str, float] = {
    "white_wins": 1.0,
    "black_wins": -1.0,
    "draw": 0.0,
}


def restricted_support_policy(
    evaluator: NetworkEvaluator,
    state: State,
    ruleset: RuleSet,
    color: Color,
    support: tuple[Program, ...],
    search_config: SearchConfig,
    rng: random.Random,
) -> _FloatArray:
    """Run the search at `state` and project its (search-derived, not raw
    network prior) policy for `color` onto `support` -- a probability vector
    over `support`'s programs, renormalized within it (the "restricted" in
    restricted-support NashConv: only the mass this support can represent).
    """
    root = make_root(state)
    run_simulations(
        root,
        ruleset,
        evaluator,
        search_config.simulations,
        rng,
        prior_weight=search_config.prior_weight,
        epsilon=search_config.epsilon_floor,
        selection=search_config.selection,
    )
    stats = root.white if color is Color.WHITE else root.black
    assert stats is not None and root.context is not None
    slot1_target = stats.average_strategy()

    weights = np.zeros(len(support), dtype=np.float64)
    for i, program in enumerate(support):
        first = program[0]
        a1_index = encode_action(first, state)
        p1 = slot1_target.get(a1_index, 0.0)
        if p1 <= 0.0:
            continue
        slot2_dist = evaluator.slot2_prior(
            root.context, color, state, ruleset, a1_index, first
        )
        if len(program) == 1:
            p2 = slot2_dist.get(NO_SECOND_INDEX, 0.0)
        else:
            a2_index = encode_action(program[1], state)
            p2 = slot2_dist.get(a2_index, 0.0)
        weights[i] = p1 * p2

    total = weights.sum()
    if total <= 0.0:
        return np.full(len(support), 1.0 / len(support))
    normalized: _FloatArray = weights / total
    return normalized


@dataclass(frozen=True, slots=True)
class NashConvResult:
    """One fixture's exact restricted-support NashConv figures."""

    fixture_name: str
    solved_value: float
    actual_value: float
    best_response_white: float
    best_response_black: float
    nashconv: float


def restricted_support_nashconv_exact(
    net: SimultChessNet,
    state: State,
    ruleset: RuleSet,
    matrix: _FloatArray,
    white_support: tuple[Program, ...],
    black_support: tuple[Program, ...],
    search_config: SearchConfig,
    fixture_name: str,
    rng: random.Random,
    *,
    device: torch.device | None = None,
) -> NashConvResult:
    """`restricted_support_nashconv`'s math (v3 18b.3), generalized to a
    fixture with no chi symmetry: the caller supplies White's and Black's
    own -- possibly unrelated -- supports and the payoff matrix directly,
    rather than deriving Black's support as White's mirror and the matrix
    from `build_stage_matrix`'s one-ply material surrogate. Meant to be fed
    `solve_exact`'s exhaustive root programs and its exact (recursively
    backward-induced, not surrogate) `stage_matrix`, so `solved_value` is a
    genuinely exact game value rather than a one-ply approximation of one.

    NashConv, for a zero-sum game, reduces to
    ``max_x x^T U pi_B - min_y pi_W^T U y`` (best-response value for White
    given Black's policy, minus best-response value for Black given White's
    policy) -- zero exactly at a Nash equilibrium, positive otherwise.
    """
    evaluator = NetworkEvaluator(net, device=device)
    solved = solve_zero_sum(matrix)

    pi_white = restricted_support_policy(
        evaluator, state, ruleset, Color.WHITE, white_support, search_config, rng
    )
    pi_black = restricted_support_policy(
        evaluator, state, ruleset, Color.BLACK, black_support, search_config, rng
    )

    actual_value = float(pi_white @ matrix @ pi_black)
    best_response_white = float((matrix @ pi_black).max())
    best_response_black = float((pi_white @ matrix).min())
    nashconv = best_response_white - best_response_black

    return NashConvResult(
        fixture_name=fixture_name,
        solved_value=float(solved.value),
        actual_value=actual_value,
        best_response_white=best_response_white,
        best_response_black=best_response_black,
        nashconv=nashconv,
    )


def restricted_support_nashconv(
    net: SimultChessNet,
    state: State,
    ruleset: RuleSet,
    search_config: SearchConfig,
    fixture_name: str,
    rng: random.Random,
    *,
    device: torch.device | None = None,
) -> NashConvResult:
    """Exact restricted-support NashConv of the learned agent's policy at a
    chi-symmetric fixture (design §6.3). Black's support is White's support
    mirrored (chi-closed), matching the M5 construction, so `U = -U^T`
    (`solver.collision.mirror_program`, the same identity M5 itself proves).
    """
    support_white = enumerate_support(state, Color.WHITE, ruleset, rng)
    support_black = tuple(mirror_program(program) for program in support_white)
    matrix = build_stage_matrix(state, support_white, support_black, ruleset)
    return restricted_support_nashconv_exact(
        net,
        state,
        ruleset,
        matrix,
        support_white,
        support_black,
        search_config,
        fixture_name,
        rng,
        device=device,
    )


def row_sketch_nashconv_exact(
    root: RowSketchNode, ruleset: RuleSet, fixture_name: str
) -> NashConvResult:
    """Exact (not restricted-support) NashConv of a `row_sketch` node's own
    reported policy (v3 18d.2) -- `restricted_support_nashconv_exact` above
    cannot be reused here: it unconditionally drives `learn.search`'s tree,
    which has nothing to do with a `row_sketch` node's regret-matching/MMD
    state. `average_strategy()` already returns the right object per
    `selection` (MMD's last iterate `x`, or the linearly-averaged strategy
    for RM+/Hedge) -- no special-casing needed here.

    Best response is searched over the **full exhaustive** program pool
    (`interop.openspiel_adapter.enumerate_legal_programs`, the same
    ground-truth enumerator 18b/18c.3 rely on), not confined to `root`'s own
    seeded pool -- a strictly stronger (exact, not restricted-support)
    NashConv than `restricted_support_nashconv_exact`'s. `u(a, b)` is
    `phi()` + a terminal check at `root`'s own single ply (the same
    `max_depth=1` "undecided -> neutral" convention `solver.exact.
    solve_exact` and `solver.double_oracle` already use), memoized --
    identical pattern to `solver.double_oracle.solve_by_double_oracle`'s own
    `u`, duplicated rather than imported since it is three lines and
    `double_oracle`'s `u` is a private per-call closure, not a shared
    utility."""
    state = root.state
    assert root.white is not None and root.black is not None
    white_programs = root.white.programs
    black_programs = root.black.programs
    white_strategy = root.white.average_strategy()
    black_strategy = root.black.average_strategy()

    white_pool = enumerate_legal_programs(state, Color.WHITE, ruleset)
    black_pool = enumerate_legal_programs(state, Color.BLACK, ruleset)

    cache: dict[tuple[Program, Program], float] = {}

    def u(white_program: Program, black_program: Program) -> float:
        key = (white_program, black_program)
        cached = cache.get(key)
        if cached is not None:
            return cached
        result = phi(state, white_program, black_program, ruleset)
        value = _TERMINAL_PAYOFF.get(result.outcome, 0.0)
        cache[key] = value
        return value

    actual_value = sum(
        pw * pb * u(white_programs[i], black_programs[j])
        for i, pw in white_strategy.items()
        for j, pb in black_strategy.items()
    )
    best_response_white = max(
        sum(pb * u(candidate, black_programs[j]) for j, pb in black_strategy.items())
        for candidate in white_pool
    )
    best_response_black = min(
        sum(pw * u(white_programs[i], candidate) for i, pw in white_strategy.items())
        for candidate in black_pool
    )
    solved_value = solve_exact(state, ruleset, max_nodes=20_000, max_depth=1).root.value

    return NashConvResult(
        fixture_name=fixture_name,
        solved_value=solved_value,
        actual_value=actual_value,
        best_response_white=best_response_white,
        best_response_black=best_response_black,
        nashconv=best_response_white - best_response_black,
    )
