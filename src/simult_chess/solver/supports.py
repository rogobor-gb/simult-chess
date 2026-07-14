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
from simult_chess.core import legality
from simult_chess.core.types import Action, Color, Program, State
from simult_chess.rules.ruleset import RuleSet


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

    Pruning heuristic: take up to `max_single_actions` individually-legal
    Move/Castle/Reserve actions (seeded-shuffled, then truncated), form
    every legal program of size 1 and, if `ruleset.n_actions >= 2`, every
    legal ordered pair from that pool (declaration order matters — spec
    §6.3's annihilation ranking and §4.3's reservation age both depend on
    it, so both orderings of a pair are kept as distinct candidate
    programs when both are legal). The result is capped at `max_programs`
    by seeded sampling. Always includes at least one program if `color`
    has any legal action at all.
    """
    pool: list[Action] = [
        *move_candidates(state, color, rng),
        *reserve_candidates(state, color),
    ]
    rng.shuffle(pool)
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
