"""Stage-implementation strategy registry (dev brief §3.2).

Each ordered resolution stage is a `Protocol`; v1's default implementation
registers under the `RuleSet` value(s) it satisfies. A future variant lever
(spec §13) becomes an alternative registration, not an edit to `phi.py`.
More stage Protocols are added here incrementally as each stage's data
structures are designed (fizzle now; annihilation and defense follow).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from simult_chess.core.moves import DeclaredMove
from simult_chess.core.stages.fizzle import FizzleResult, resolve_fizzles
from simult_chess.core.types import State
from simult_chess.rules.ruleset import FizzleScope, RuleSet


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
