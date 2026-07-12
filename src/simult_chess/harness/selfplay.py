"""Headless self-play: seeded, reproducible K-game sweep with the invariant
harness in lenient mode (dev brief Phase 6 DoD).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from simult_chess.agents.base import Agent
from simult_chess.core.types import Color, State
from simult_chess.core.violation import Violation
from simult_chess.invariants.harness import run_phase
from simult_chess.invariants.severity import Severity, severity_of
from simult_chess.referee.observe import ObservationChannel
from simult_chess.rules.ruleset import RuleSet


@dataclass(frozen=True, slots=True)
class ViolationRecord:
    """One violation, localized to a phase and game seed (dev brief Phase 6 DoD)."""

    rng_seed: int
    phase_index: int
    violation: Violation


@dataclass(frozen=True, slots=True)
class GameReport:
    """One seeded game's outcome and every invariant violation found."""

    rng_seed: int
    phases_played: int
    outcome: str
    violations: tuple[ViolationRecord, ...]


@dataclass(frozen=True, slots=True)
class SweepReport:
    """Aggregated results across a self-play sweep."""

    games: tuple[GameReport, ...]

    @property
    def all_violations(self) -> tuple[ViolationRecord, ...]:
        return tuple(v for game in self.games for v in game.violations)

    def violations_of_severity(
        self, *severities: Severity
    ) -> tuple[ViolationRecord, ...]:
        wanted = set(severities)
        return tuple(
            v
            for v in self.all_violations
            if severity_of(v.violation.invariant_id) in wanted
        )

    def counts_by_invariant(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.all_violations:
            invariant_id = record.violation.invariant_id
            counts[invariant_id] = counts.get(invariant_id, 0) + 1
        return counts


def play_one_game(
    initial_state: State,
    agent_white: Agent,
    agent_black: Agent,
    ruleset: RuleSet,
    rng_seed: int,
    *,
    max_phases: int = 500,
) -> GameReport:
    """Play one seeded game with per-phase invariant checking in lenient mode.

    Determinism (dev brief Phase 6 DoD): the same `rng_seed` always
    reproduces the same game, since the agents only draw from the seeded
    `random.Random` instances constructed here.
    """
    rng_white = random.Random(rng_seed)
    rng_black = random.Random(rng_seed ^ 0x5EED)
    channel = ObservationChannel()
    state = initial_state
    outcome = "ongoing"
    violations: list[ViolationRecord] = []
    phase_index = 0

    for _ in range(max_phases):
        program_white = agent_white(state, Color.WHITE, ruleset, rng_white)
        program_black = agent_black(state, Color.BLACK, ruleset, rng_black)
        white_commitment = channel.commit(Color.WHITE, program_white)
        black_commitment = channel.commit(Color.BLACK, program_black)
        revealed_white = channel.reveal(white_commitment)
        revealed_black = channel.reveal(black_commitment)

        result = run_phase(
            state,
            revealed_white,
            revealed_black,
            ruleset,
            mode="lenient",
            rng_seed=rng_seed,
        )
        phase_index += 1
        violations.extend(
            ViolationRecord(rng_seed=rng_seed, phase_index=phase_index, violation=v)
            for v in result.violations
        )

        if result.phi_result is None:
            outcome = "aborted"
            break
        state = result.phi_result.state
        outcome = result.phi_result.outcome
        if outcome != "ongoing":
            break
    else:
        outcome = "phase_limit_reached"

    return GameReport(
        rng_seed=rng_seed,
        phases_played=phase_index,
        outcome=outcome,
        violations=tuple(violations),
    )


def run_sweep(
    initial_state_factory: Callable[[], State],
    agent_white: Agent,
    agent_black: Agent,
    ruleset: RuleSet,
    *,
    num_games: int,
    base_seed: int = 0,
    max_phases: int = 500,
) -> SweepReport:
    """Run `num_games` seeded self-play games, aggregating invariant violations."""
    games = tuple(
        play_one_game(
            initial_state_factory(),
            agent_white,
            agent_black,
            ruleset,
            base_seed + i,
            max_phases=max_phases,
        )
        for i in range(num_games)
    )
    return SweepReport(games=games)
