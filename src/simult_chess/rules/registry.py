"""Stage-implementation strategy registry (dev brief §3.2).

Each ordered resolution stage is a `Protocol`; v1's default implementation
registers under the `RuleSet` value(s) it satisfies. A future variant lever
(spec §13) becomes an alternative registration, not an edit to `phi.py`.
More stage Protocols are added here incrementally as each stage's data
structures are designed (fizzle and annihilation now; defense follows).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from simult_chess.core.moves import DeclaredMove
from simult_chess.core.stages.annihilate import (
    AnnihilationResult,
    Edge,
    resolve_annihilation,
)
from simult_chess.core.stages.fizzle import FizzleResult, resolve_fizzles
from simult_chess.core.types import State
from simult_chess.rules.ruleset import AnnihilationReading, FizzleScope, RuleSet


class FizzleResolver(Protocol):
    """Stage F: which declared moves fizzle (spec §6.2)."""

    def __call__(
        self,
        declared: tuple[DeclaredMove, ...],
        state: State,
        ruleset: RuleSet,
        *,
        tie_break: Sequence[DeclaredMove] | None = None,
    ) -> FizzleResult: ...


_FIZZLE_RESOLVERS: dict[FizzleScope, FizzleResolver] = {
    "both_pawns": resolve_fizzles,
    "any_same_square": resolve_fizzles,
}


def get_fizzle_resolver(ruleset: RuleSet) -> FizzleResolver:
    """Look up the Stage F implementation registered for `ruleset`."""
    return _FIZZLE_RESOLVERS[ruleset.pawn_same_square_fizzle_scope]


class AnnihilationMatcher(Protocol):
    """Stage A: which executing moves annihilate (spec §6.3)."""

    def __call__(
        self,
        executing: tuple[DeclaredMove, ...],
        ruleset: RuleSet,
        *,
        tie_break: Sequence[Edge] | None = None,
    ) -> AnnihilationResult: ...


_ANNIHILATION_MATCHERS: dict[AnnihilationReading, AnnihilationMatcher] = {
    "B": resolve_annihilation,
}


def get_annihilation_matcher(ruleset: RuleSet) -> AnnihilationMatcher:
    """Look up the Stage A implementation registered for `ruleset`.

    Only the v1 declaration-priority reading ("B") is implemented; the timed
    one-tick model is a declined variant lever (spec §6.3, §13.2).
    """
    try:
        return _ANNIHILATION_MATCHERS[ruleset.annihilation_reading]
    except KeyError:
        detail = (
            f"annihilation_reading={ruleset.annihilation_reading!r} has no "
            "registered Stage A implementation (spec §13.2: declined for v1)"
        )
        raise NotImplementedError(detail) from None
