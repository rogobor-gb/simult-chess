"""Executable invariant harness wired around Φ (INVARIANTS.md §0).

`run_phase` is the reference way to resolve a phase under supervision:
every `WF`/`L`/`R`/`T` check runs automatically. **strict** mode (the
default; used in unit tests and CI) raises on the first violating phase,
carrying a replayable repro dump per violation. **lenient** mode (the
headless self-play fuzzer's default) aggregates violations across many
phases without halting, so one run harvests every finding rather than
dying on the first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from simult_chess.core import legality
from simult_chess.core.phi import PhiResult, phi
from simult_chess.core.types import Color, Program, State
from simult_chess.core.violation import Violation
from simult_chess.invariants import checks, resolution_checks
from simult_chess.invariants.repro import ReproDump, build_repro_dump
from simult_chess.rules.ruleset import RuleSet

HarnessMode = Literal["strict", "lenient"]


def _check_point_of(invariant_id: str) -> str:
    if invariant_id.startswith("WF"):
        return "STATE"
    if invariant_id.startswith("L"):
        return "PRE"
    return "TRACE"


class InvariantViolationError(Exception):
    """Raised in strict mode when one or more invariant checks fail."""

    def __init__(self, violations: tuple[Violation, ...]) -> None:
        self.violations = violations
        summary = "; ".join(f"{v.invariant_id}: {v.detail}" for v in violations[:5])
        super().__init__(f"{len(violations)} invariant violation(s): {summary}")


@dataclass(frozen=True, slots=True)
class HarnessResult:
    """The outcome of one harness-checked phase resolution.

    `phi_result` is `None` iff a PRE violation stopped the phase before
    `Φ` ran at all.
    """

    violations: tuple[Violation, ...]
    phi_result: PhiResult | None
    repro_dumps: tuple[ReproDump, ...]


def run_phase(
    state: State,
    program_white: Program,
    program_black: Program,
    ruleset: RuleSet,
    *,
    mode: HarnessMode = "strict",
    rng_seed: int = 0,
) -> HarnessResult:
    """Run one Φ call with every WF/L/R/T check wired around it (INVARIANTS.md §0)."""
    pre_violations: list[Violation] = [
        *legality.check_legal_program(state, program_white, Color.WHITE, ruleset),
        *legality.check_legal_program(state, program_black, Color.BLACK, ruleset),
    ]
    if pre_violations:
        dumps = tuple(
            build_repro_dump(
                v,
                _check_point_of(v.invariant_id),
                state,
                program_white,
                program_black,
                ruleset,
                rng_seed=rng_seed,
            )
            for v in pre_violations
        )
        if mode == "strict":
            raise InvariantViolationError(tuple(pre_violations))
        return HarnessResult(
            violations=tuple(pre_violations), phi_result=None, repro_dumps=dumps
        )

    result = phi(state, program_white, program_black, ruleset)

    violations: list[Violation] = [
        *checks.check_all_state(state, ruleset, allow_terminal=False),
        *checks.check_all_state(
            result.state, ruleset, allow_terminal=result.outcome != "ongoing"
        ),
        *checks.check_wf2_type_constancy(
            state, result.state, promoted=result.trace.promoted
        ),
        *checks.check_wf7_bookkeeping_monotone(state, result.state),
        *resolution_checks.check_all_trace(
            state, result.state, result.trace, result.outcome, ruleset
        ),
    ]

    dumps = tuple(
        build_repro_dump(
            v,
            _check_point_of(v.invariant_id),
            state,
            program_white,
            program_black,
            ruleset,
            rng_seed=rng_seed,
        )
        for v in violations
    )

    if violations and mode == "strict":
        raise InvariantViolationError(tuple(violations))
    return HarnessResult(
        violations=tuple(violations), phi_result=result, repro_dumps=dumps
    )
