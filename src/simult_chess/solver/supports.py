"""Restricted program-support enumeration
:math:`A_\\omega(s) \\subseteq \\Pi_\\omega(s)` (spec §8.4, A7).

Exact enumeration of :math:`\\Pi_\\omega(s)` is combinatorially infeasible
(spec §8.4: stage matrices reach :math:`10^6`-:math:`10^8` entries even for
modest supports), so the engine "lives on sampled stage equilibria from the
outset". This module builds a small, seeded, *explicitly pruned* candidate
set instead — a solver parameter, never a rule.
"""

from __future__ import annotations

import random

from simult_chess.agents.candidates import move_candidates, reserve_candidates
from simult_chess.core import geometry, legality
from simult_chess.core.types import Action, Color, Move, PieceType, Program, State
from simult_chess.rules.ruleset import RuleSet

_PIECE_VALUES: dict[PieceType, int] = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}


def _capture_value(state: State, color: Color, action: Action) -> int:
    """0 for a non-capturing action; the captured piece's value otherwise --
    a cheap, explicit pruning heuristic so an unopposed capture is never
    dropped purely by chance (spec §8.4's "explicit, seeded pruning
    heuristics", not blind random sampling)."""
    if not isinstance(action, Move):
        return 0
    occupant = geometry.occupant_lookup(state.board)
    target = occupant(action.trajectory.destination)
    if target is None or target.color is color:
        return 0
    return _PIECE_VALUES[target.typ]


def enumerate_support(
    state: State,
    color: Color,
    ruleset: RuleSet,
    rng: random.Random,
    *,
    max_single_actions: int = 8,
    max_programs: int = 8,
) -> tuple[Program, ...]:
    """A small, seeded, legal restricted support for `color` at `state`.

    Pruning heuristic: seeded-shuffle every individually-legal Move/Castle/
    Reserve action (for a stable seeded tie-break), then stable-sort by
    `_capture_value` descending and truncate to `max_single_actions` — the
    highest-value captures always survive the cut; ties (most commonly,
    all the quiet non-capturing moves) keep their shuffled order, giving
    seeded diversity among the rest. From that pool, form every legal
    program of size 1 and, if `ruleset.n_actions >= 2`, every legal ordered
    pair (declaration order matters — spec §6.3's annihilation ranking and
    §4.3's reservation age both depend on it, so both orderings of a pair
    are kept as distinct candidate programs when both are legal). The
    result is capped at `max_programs` by seeded sampling. Always includes
    at least one program if `color` has any legal action at all.
    """
    pool: list[Action] = [
        *move_candidates(state, color, rng),
        *reserve_candidates(state, color),
    ]
    rng.shuffle(pool)
    pool.sort(key=lambda action: _capture_value(state, color, action), reverse=True)
    pool = pool[:max_single_actions]

    programs: set[Program] = set()
    for action in pool:
        single: Program = (action,)
        if legality.is_legal_program(state, single, color, ruleset):
            programs.add(single)

    if ruleset.n_actions >= 2:
        for i, first in enumerate(pool):
            for j, second in enumerate(pool):
                if i == j:
                    continue
                pair: Program = (first, second)
                if legality.is_legal_program(state, pair, color, ruleset):
                    programs.add(pair)

    support = list(programs)
    rng.shuffle(support)
    return tuple(support[:max_programs])
