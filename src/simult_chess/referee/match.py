"""The phase loop: commit -> reveal -> resolve, until terminal (spec §5, §10)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from simult_chess.agents.base import Agent
from simult_chess.core.phi import PhiTrace, phi
from simult_chess.core.types import Color, State
from simult_chess.referee.observe import ObservationChannel
from simult_chess.rules.ruleset import RuleSet

MatchOutcome = Literal[
    "ongoing", "white_wins", "black_wins", "draw", "phase_limit_reached"
]


@dataclass(frozen=True, slots=True)
class PhaseRecord:
    """One resolved phase's event log entry (spec §6.7's emit_event_log)."""

    state_before: State
    state_after: State
    outcome: str
    trace: PhiTrace


@dataclass(frozen=True, slots=True)
class MatchResult:
    """The outcome of one complete match."""

    final_state: State
    outcome: MatchOutcome
    phases: tuple[PhaseRecord, ...]


def play_match(
    initial_state: State,
    agent_white: Agent,
    agent_black: Agent,
    ruleset: RuleSet,
    *,
    rng_white: random.Random,
    rng_black: random.Random,
    max_phases: int = 500,
) -> MatchResult:
    """Run phases via commit->reveal->resolve until terminal or `max_phases`.

    `max_phases` is a defensive cap, not a rule: T4's no-progress horizon
    (`ruleset.horizon`) should always force a draw well before this, so
    hitting the cap indicates a bug rather than a legitimate long game.
    """
    channel = ObservationChannel()
    state = initial_state
    outcome: MatchOutcome = "ongoing"
    phases: list[PhaseRecord] = []

    for _ in range(max_phases):
        program_white = agent_white(state, Color.WHITE, ruleset, rng_white)
        program_black = agent_black(state, Color.BLACK, ruleset, rng_black)

        white_commitment = channel.commit(Color.WHITE, program_white)
        black_commitment = channel.commit(Color.BLACK, program_black)
        revealed_white = channel.reveal(white_commitment)
        revealed_black = channel.reveal(black_commitment)

        result = phi(state, revealed_white, revealed_black, ruleset)
        phases.append(
            PhaseRecord(
                state_before=state,
                state_after=result.state,
                outcome=result.outcome,
                trace=result.trace,
            )
        )
        state = result.state
        outcome = result.outcome
        if outcome != "ongoing":
            break
    else:
        outcome = "phase_limit_reached"

    return MatchResult(final_state=state, outcome=outcome, phases=tuple(phases))
