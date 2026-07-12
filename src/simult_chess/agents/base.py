"""The Agent Protocol: given a state, produce a legal program (spec §14)."""

from __future__ import annotations

import random
from typing import Protocol

from simult_chess.core.types import Color, Program, State
from simult_chess.rules.ruleset import RuleSet


class Agent(Protocol):
    """A decision-phase program source for one color.

    Takes an explicit `random.Random` rather than reading global random
    state, so a self-play game is a pure function of its seeds (dev brief
    Phase 6 DoD: "re-running a seed reproduces the game exactly").
    """

    def __call__(
        self, state: State, color: Color, ruleset: RuleSet, rng: random.Random
    ) -> Program: ...
