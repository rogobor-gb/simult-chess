"""A uniform-over-legal-programs agent — the fuzzing workhorse (dev brief Phase 6)."""

from __future__ import annotations

import random

from simult_chess.agents.candidates import move_candidates, reserve_candidates
from simult_chess.core import legality
from simult_chess.core.types import Color, Program, State
from simult_chess.rules.ruleset import RuleSet


def random_legal_program(
    state: State,
    color: Color,
    ruleset: RuleSet,
    rng: random.Random,
    *,
    max_attempts: int = 20,
) -> Program:
    """Sample a random legal program for `color` (dev brief Phase 6).

    Not a perfectly uniform sample over the full (combinatorially huge,
    spec §8.4) legal-program space: builds the pool of individually-legal
    single actions, then repeatedly samples a random-sized combination and
    validates it against `L(s,π)`, falling back to a single guaranteed-legal
    action (a Move/Castle if any exist, else whatever's available) if no
    valid combination is found within `max_attempts`. Good enough diversity
    for fuzzing; always terminates; always legal.
    """
    movers = move_candidates(state, color, rng)
    reservers = reserve_candidates(state, color)
    all_candidates = movers + reservers
    if not all_candidates:
        return ()

    for _ in range(max_attempts):
        size = rng.randint(1, min(ruleset.n_actions, len(all_candidates)))
        sample = tuple(rng.sample(all_candidates, size))
        if legality.is_legal_program(state, sample, color, ruleset):
            return sample

    if movers:
        return (rng.choice(movers),)
    return (rng.choice(all_candidates),)
