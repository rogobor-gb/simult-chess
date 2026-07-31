"""Exact backward-induction solver over the *full* reachable state graph
(Phase 18b, docs/LEARNING_ROADMAP_v3.md).

Unlike `solver.stage_matrix.build_stage_matrix` (a single stage matrix over
a caller-supplied *restricted* support, non-terminal successors valued by a
material-difference surrogate), this module solves a whole subgame exactly:
it enumerates every reachable state via `interop.openspiel_adapter.
enumerate_legal_programs` (exhaustive, both colours) at every node, then
backward-induces the value from the terminal states up (Shapley's operator,
Shapley 1953 — well-defined without a Banach fixed-point argument here
because T4 (`ruleset.horizon`) gives a finite effective horizon: `phase_
index` strictly increases every `phi()` call, so the reachable graph is a
DAG layered by phase depth, and backward induction over that DAG *is* the
proof of uniqueness, per v3 Vol. I §1.3).

**Tractability, measured, not assumed** (v3 18b, this session's own
scoping pass): a bare K+R-vs-K position's branching factor is already
~150-176 *distinct successor states per phase* even at `RuleSet(n_actions=1)`
— full exhaustive solving from an open endgame placement to checkmate is
completely intractable (B^4 already exceeds 500 million states). This
module is therefore meant for **small, tightly-constructed positions**
(few legal actions per side by geometric construction, exactly the
wall/corner-boxing idiom `tests/unit/test_search_matching_pennies.py`
already uses for its Matching-Pennies and asymmetric fixtures) that
resolve within a handful of phases — the roadmap's own "two-to-four-phase
constructed positions" category — not open subgames.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from simult_chess.core.phi import phi
from simult_chess.core.types import Color, Program, State
from simult_chess.interop.openspiel_adapter import enumerate_legal_programs
from simult_chess.rules.ruleset import RuleSet
from simult_chess.solver.lp import solve_zero_sum

FloatArray = npt.NDArray[np.float64]

_TERMINAL_PAYOFF: dict[str, float] = {
    "white_wins": 1.0,
    "black_wins": -1.0,
    "draw": 0.0,
}

StateKey = tuple[
    tuple[tuple[int, str, str, int, int], ...],  # board: (id, colour, type, file, rank)
    frozenset[int],  # cooldown token ids
    tuple[tuple[int, int, tuple[int, int]], ...],  # reservations_white
    tuple[tuple[int, int, tuple[int, int]], ...],  # reservations_black
    int,  # phase_index
    int,  # no_progress_counter
    object,  # castling_rights (already hashable, a frozen dataclass)
    frozenset[tuple[Hashable, int]],  # repetition_ledger items
]


def state_key(state: State) -> StateKey:
    """A hashable structural key for `state` (`State` itself isn't hashable
    -- `board` and `bookkeeping.repetition_ledger` are `MappingProxyType`).
    Used to memoize the reachable-state graph: two independently-reached
    `State` objects with the same key are the same node. Includes every
    field that can affect `phi`'s future behaviour (board, cooldown,
    reservations, phase/no-progress counters, castling rights, and the
    repetition ledger -- included for general correctness even though the
    small fixtures this module targets are far too shallow for 3-fold
    repetition to matter in practice)."""
    board_key = tuple(
        sorted(
            (token.id, token.color.value, token.typ, square.file, square.rank)
            for token, square in state.board.items()
        )
    )
    cooldown_key = frozenset(token.id for token in state.cooldown)

    def reservation_key(
        reservations: tuple[object, ...],
    ) -> tuple[tuple[int, int, tuple[int, int]], ...]:
        return tuple(
            (r.defender.id, r.protege.id, r.age)  # type: ignore[attr-defined]
            for r in reservations
        )

    return (
        board_key,
        cooldown_key,
        reservation_key(state.reservations_white),
        reservation_key(state.reservations_black),
        state.bookkeeping.phase_index,
        state.bookkeeping.no_progress_counter,
        state.bookkeeping.castling_rights,
        frozenset(state.bookkeeping.repetition_ledger.items()),
    )


@dataclass
class SolvedNode:
    """One node of a solved subgame."""

    state: State
    value: float
    """White's exact game value at this state."""
    is_terminal: bool
    is_depth_limited: bool = False
    """True only for a node whose `value` is the `max_depth` fallback
    (0.0, matching `learn.selfplay.ReplayBuffer`'s own `phase_limit_reached`/
    `aborted` -> `z=0.0` convention), *not* a real `phi` terminal -- callers
    that need a genuinely exact fixture (not a depth-bounded approximation)
    should check this is `False` everywhere in the graph."""
    white_programs: tuple[Program, ...] = ()
    black_programs: tuple[Program, ...] = ()
    stage_matrix: FloatArray | None = None
    """`U[i, j]` = the exact (already-solved) value of the state reached by
    `(white_programs[i], black_programs[j])` -- `None` for terminal nodes."""


@dataclass
class SolvedGraph:
    """The full result of `solve_exact`: every reachable node, keyed."""

    root_key: StateKey
    ruleset: RuleSet
    nodes: dict[StateKey, SolvedNode] = field(default_factory=dict)

    @property
    def root(self) -> SolvedNode:
        return self.nodes[self.root_key]


class IntractableGraphError(RuntimeError):
    """Raised when the reachable graph exceeds `max_nodes` -- a guard rail,
    not a normal code path (this module's fixtures are meant to be small
    by construction; see the module docstring's tractability note)."""


def solve_exact(
    root: State,
    ruleset: RuleSet,
    *,
    max_nodes: int = 20_000,
    max_depth: int | None = None,
) -> SolvedGraph:
    """Backward induction over the full reachable graph from `root`.

    Two passes: a forward BFS builds the graph (exhaustive `enumerate_
    legal_programs` + `phi` at every node, memoized via `state_key`), then
    backward induction processes nodes in reverse BFS order (deepest first,
    which is always a valid topological order here since `phase_index`
    strictly increases every transition) — terminal states get their known
    payoff directly; every other state's value is `solve_zero_sum`'s value
    on the stage matrix over its *exhaustive* program sets, using already-
    solved successor values, not a surrogate.

    Raises `IntractableGraphError` if the graph exceeds `max_nodes` before
    finishing the forward pass. **This is the common case, not a rare
    edge case, for anything but a fully-boxed dilemma that resolves in one
    phase**: measured directly this session (v3 18b's own scoping pass),
    even a proven-small fixture (`tests/unit/test_search_matching_pennies.
    py`'s own Matching-Pennies construction) explodes past thousands of
    nodes once its two "ongoing" branches are allowed to run to genuine
    completion rather than being depth-truncated -- board mobility on the
    real 8x8 board grows too fast once anything is left free to move for
    more than a phase or two. `max_depth`, if given, is the practical
    answer and directly matches the roadmap's own "two-to-four-phase
    constructed positions" category: any state still `"ongoing"` at the
    depth limit is treated as a leaf with value 0.0 (`SolvedNode.
    is_depth_limited=True`) -- the same "undecided -> neutral" convention
    `learn.selfplay.ReplayBuffer` already uses for `phase_limit_reached`/
    `aborted` self-play games, not a new one invented here. A fixture
    solved this way is only genuinely *exact* if construction ensures no
    node actually needs the fallback (check `is_depth_limited` is `False`
    everywhere, e.g. via `test_solver_exact.py`'s pattern) -- `max_depth`
    is a bound on how deep to search, not license to guess at unresolved
    positions.
    """
    root_key = state_key(root)
    states: dict[StateKey, State] = {root_key: root}
    depth: dict[StateKey, int] = {root_key: 0}
    terminal_payoff: dict[StateKey, float] = {}
    depth_limited: set[StateKey] = set()
    white_programs: dict[StateKey, tuple[Program, ...]] = {}
    black_programs: dict[StateKey, tuple[Program, ...]] = {}
    successor_grid: dict[StateKey, list[list[StateKey]]] = {}
    order: list[StateKey] = [root_key]

    frontier = [root_key]
    while frontier:
        next_frontier: list[StateKey] = []
        for key in frontier:
            state = states[key]
            white = tuple(enumerate_legal_programs(state, Color.WHITE, ruleset))
            black = tuple(enumerate_legal_programs(state, Color.BLACK, ruleset))
            white_programs[key] = white
            black_programs[key] = black
            grid: list[list[StateKey]] = []
            for program_white in white:
                row: list[StateKey] = []
                for program_black in black:
                    result = phi(state, program_white, program_black, ruleset)
                    successor_key = state_key(result.state)
                    # Always record the actual State for a newly-seen key
                    # (terminal or not) -- SolvedNode.state must be the real
                    # object, never a placeholder.
                    if successor_key not in states:
                        states[successor_key] = result.state
                    if result.outcome != "ongoing":
                        terminal_payoff.setdefault(
                            successor_key, _TERMINAL_PAYOFF[result.outcome]
                        )
                    elif successor_key not in depth:
                        successor_depth = depth[key] + 1
                        if max_depth is not None and successor_depth > max_depth:
                            terminal_payoff.setdefault(successor_key, 0.0)
                            depth_limited.add(successor_key)
                        else:
                            depth[successor_key] = successor_depth
                            next_frontier.append(successor_key)
                            order.append(successor_key)
                            if len(states) > max_nodes:
                                raise IntractableGraphError(
                                    f"reachable graph exceeded max_nodes={max_nodes} "
                                    "-- this fixture is not small enough for "
                                    "solve_exact (see module docstring)"
                                )
                    row.append(successor_key)
                grid.append(row)
            successor_grid[key] = grid
        frontier = next_frontier

    values: dict[StateKey, float] = dict(terminal_payoff)
    nodes: dict[StateKey, SolvedNode] = {
        key: SolvedNode(
            state=states[key],
            value=payoff,
            is_terminal=True,
            is_depth_limited=key in depth_limited,
        )
        for key, payoff in terminal_payoff.items()
    }

    for key in reversed(order):
        if key in terminal_payoff:
            continue
        white = white_programs[key]
        black = black_programs[key]
        grid = successor_grid[key]
        matrix = np.array(
            [
                [values[grid[i][j]] for j in range(len(black))]
                for i in range(len(white))
            ],
            dtype=np.float64,
        )
        solution = solve_zero_sum(matrix)
        values[key] = solution.value
        nodes[key] = SolvedNode(
            state=states[key],
            value=solution.value,
            is_terminal=False,
            white_programs=white,
            black_programs=black,
            stage_matrix=matrix,
        )

    return SolvedGraph(root_key=root_key, ruleset=ruleset, nodes=nodes)


def solved_child(graph: SolvedGraph, node: SolvedNode, i: int, j: int) -> SolvedNode:
    """The `SolvedNode` reached by `(node.white_programs[i],
    node.black_programs[j])` -- a small convenience for callers walking a
    `SolvedGraph` (e.g. to replay the equilibrium path through `phi`)."""
    result = phi(
        node.state, node.white_programs[i], node.black_programs[j], graph.ruleset
    )
    return graph.nodes[state_key(result.state)]
