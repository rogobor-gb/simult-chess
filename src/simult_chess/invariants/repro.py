"""Repro-dump schema and deterministic replay (INVARIANTS.md §9).

A dump is self-contained: `rng_seed + ruleset + state_pre + programs`
suffices to replay the exact `Φ` call deterministically (inv M1 purity),
with no reference to external state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from simult_chess.core.phi import PhiResult, phi
from simult_chess.core.types import Program, State
from simult_chess.core.violation import Violation
from simult_chess.referee.serialize import serialize_state
from simult_chess.rules.ruleset import RuleSet


@dataclass(frozen=True, slots=True)
class ReproDump:
    """A self-contained, replayable record of one failed check (inv §9).

    Parameters
    ----------
    violation : Violation
        The failed predicate and its detail.
    check_point : str
        One of ``"PRE"``, ``"STATE"``, ``"TRACE"`` (INVARIANTS.md §0.2).
    phase_index : int
        The phase at which the violation was observed.
    rng_seed : int
        Reproduces agent sampling upstream of this call, if any (v1's
        hand-built fixtures default to 0).
    ruleset : RuleSet
        The `RuleSet` in effect.
    state_pre, program_white, program_black : State, Program, Program
        The exact inputs to the offending `Φ` call.
    """

    violation: Violation
    check_point: str
    phase_index: int
    rng_seed: int
    ruleset: RuleSet
    state_pre: State
    program_white: Program
    program_black: Program

    def to_dict(self) -> dict[str, Any]:
        """A JSON-able export matching inv §9's schema (state_pre only, no re-parse)."""
        return {
            "invariant_id": self.violation.invariant_id,
            "detail": self.violation.detail,
            "check_point": self.check_point,
            "phase_index": self.phase_index,
            "rng_seed": self.rng_seed,
            "ruleset": {
                "n_actions": self.ruleset.n_actions,
                "horizon": self.ruleset.horizon,
                "recapture_cooldown": self.ruleset.recapture_cooldown,
                "cancellation_enabled": self.ruleset.cancellation_enabled,
                "pawn_same_square_fizzle_scope": (
                    self.ruleset.pawn_same_square_fizzle_scope
                ),
                "annihilation_reading": self.ruleset.annihilation_reading,
                "intermezzo_reading": self.ruleset.intermezzo_reading,
            },
            "state_pre": serialize_state(self.state_pre),
        }


def build_repro_dump(
    violation: Violation,
    check_point: str,
    state_pre: State,
    program_white: Program,
    program_black: Program,
    ruleset: RuleSet,
    *,
    rng_seed: int = 0,
) -> ReproDump:
    """Build a repro dump from a single failed check (inv §9)."""
    return ReproDump(
        violation=violation,
        check_point=check_point,
        phase_index=state_pre.bookkeeping.phase_index,
        rng_seed=rng_seed,
        ruleset=ruleset,
        state_pre=state_pre,
        program_white=program_white,
        program_black=program_black,
    )


def replay(dump: ReproDump) -> PhiResult:
    """Deterministically replay a dump's Φ call (inv M1 purity)."""
    return phi(dump.state_pre, dump.program_white, dump.program_black, dump.ruleset)
