"""Hot-seat and human-vs-agent session loops (dev brief §3.5, §4 Phase 7).

Both loops reuse `referee/observe.py`'s commit-reveal channel: each side's
program is computed/prompted in turn, but nothing is printed back until
*both* are committed. `clear_fn` runs right after a human's program is
captured, before the other side is prompted -- that is what "hides the
first mover's committed program until both are in" (spec §11.5, dev brief
§3.5) means for a single shared terminal.

All I/O is injected (`input_fn`/`print_fn`/`clear_fn`) so the loops are
unit-testable without a real terminal.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from simult_chess.agents.base import Agent
from simult_chess.core import legality
from simult_chess.core.phi import PhiTrace, phi
from simult_chess.core.types import Color, Program, State
from simult_chess.referee.observe import ObservationChannel
from simult_chess.rules.ruleset import RuleSet
from simult_chess.ui import notation
from simult_chess.ui.board_render import render_board
from simult_chess.ui.notation import NotationError, format_program

SessionOutcome = Literal[
    "ongoing", "white_wins", "black_wins", "draw", "phase_limit_reached"
]

InputFn = Callable[[str], str]
PrintFn = Callable[[str], None]
ClearFn = Callable[[], None]


def _default_clear(print_fn: PrintFn) -> ClearFn:
    return lambda: print_fn("\n" * 60)


@dataclass(frozen=True, slots=True)
class SessionPhaseRecord:
    """One resolved phase's event log entry, mirroring `referee.match.PhaseRecord`."""

    state_before: State
    state_after: State
    outcome: str
    trace: PhiTrace


@dataclass(frozen=True, slots=True)
class SessionResult:
    """The outcome of one complete interactive session."""

    final_state: State
    outcome: SessionOutcome
    phases: tuple[SessionPhaseRecord, ...]


def prompt_program(
    state: State,
    color: Color,
    ruleset: RuleSet,
    *,
    input_fn: InputFn,
    print_fn: PrintFn,
) -> Program:
    """Repeatedly prompt `color` for a program until it parses and is `L`-legal.

    On a parse failure or a legality violation, the failing detail (naming
    the `L`-clause, per Phase 7's DoD) is printed and the player is
    re-prompted -- the program is never partially accepted.
    """
    while True:
        raw = input_fn(
            f"{color.value} program (up to {ruleset.n_actions} actions, "
            f"';'-separated): "
        )
        try:
            program = notation.parse_program(raw, state, color)
        except NotationError as exc:
            print_fn(f"parse error: {exc}")
            continue
        violations = legality.check_legal_program(state, program, color, ruleset)
        if violations:
            for violation in violations:
                print_fn(f"illegal ({violation.invariant_id}): {violation.detail}")
            continue
        return program


def _resolve_phase(
    state: State, program_white: Program, program_black: Program, ruleset: RuleSet
) -> tuple[State, str, PhiTrace]:
    channel = ObservationChannel()
    white_commitment = channel.commit(Color.WHITE, program_white)
    black_commitment = channel.commit(Color.BLACK, program_black)
    revealed_white = channel.reveal(white_commitment)
    revealed_black = channel.reveal(black_commitment)
    result = phi(state, revealed_white, revealed_black, ruleset)
    return result.state, result.outcome, result.trace


def run_hot_seat(
    initial_state: State,
    ruleset: RuleSet,
    *,
    input_fn: InputFn = input,
    print_fn: PrintFn = print,
    clear_fn: ClearFn | None = None,
    max_phases: int = 500,
) -> SessionResult:
    """Two humans, one terminal: alternately prompt, hiding each from the other."""
    if clear_fn is None:
        clear_fn = _default_clear(print_fn)

    state = initial_state
    outcome: SessionOutcome = "ongoing"
    phases: list[SessionPhaseRecord] = []

    for _ in range(max_phases):
        print_fn(render_board(state))
        print_fn(f"-- phase {state.bookkeeping.phase_index}: White to declare --")
        program_white = prompt_program(
            state, Color.WHITE, ruleset, input_fn=input_fn, print_fn=print_fn
        )
        clear_fn()

        print_fn(render_board(state))
        print_fn(f"-- phase {state.bookkeeping.phase_index}: Black to declare --")
        program_black = prompt_program(
            state, Color.BLACK, ruleset, input_fn=input_fn, print_fn=print_fn
        )
        clear_fn()

        print_fn(f"White played: {format_program(program_white, state, Color.WHITE)}")
        print_fn(f"Black played: {format_program(program_black, state, Color.BLACK)}")
        new_state, result_outcome, trace = _resolve_phase(
            state, program_white, program_black, ruleset
        )
        phases.append(
            SessionPhaseRecord(
                state_before=state,
                state_after=new_state,
                outcome=result_outcome,
                trace=trace,
            )
        )
        state = new_state
        outcome = result_outcome  # type: ignore[assignment]
        if outcome != "ongoing":
            break
    else:
        outcome = "phase_limit_reached"

    print_fn(render_board(state))
    print_fn(f"Game over: {outcome}")
    return SessionResult(final_state=state, outcome=outcome, phases=tuple(phases))


def run_human_vs_agent(
    initial_state: State,
    ruleset: RuleSet,
    human_color: Color,
    agent: Agent,
    rng: random.Random,
    *,
    input_fn: InputFn = input,
    print_fn: PrintFn = print,
    clear_fn: ClearFn | None = None,
    max_phases: int = 500,
) -> SessionResult:
    """One human against an `Agent`. The agent's program is never printed
    before the human has committed theirs, matching the hot-seat's
    concealment even though the agent itself cannot "peek"."""
    if clear_fn is None:
        clear_fn = _default_clear(print_fn)
    agent_color = human_color.opponent

    state = initial_state
    outcome: SessionOutcome = "ongoing"
    phases: list[SessionPhaseRecord] = []

    for _ in range(max_phases):
        print_fn(render_board(state))
        phase = state.bookkeeping.phase_index
        print_fn(f"-- phase {phase}: your move ({human_color.value}) --")
        program_human = prompt_program(
            state, human_color, ruleset, input_fn=input_fn, print_fn=print_fn
        )
        clear_fn()
        program_agent = agent(state, agent_color, ruleset, rng)

        programs = {human_color: program_human, agent_color: program_agent}
        program_white, program_black = programs[Color.WHITE], programs[Color.BLACK]

        print_fn(f"White played: {format_program(program_white, state, Color.WHITE)}")
        print_fn(f"Black played: {format_program(program_black, state, Color.BLACK)}")
        new_state, result_outcome, trace = _resolve_phase(
            state, program_white, program_black, ruleset
        )
        phases.append(
            SessionPhaseRecord(
                state_before=state,
                state_after=new_state,
                outcome=result_outcome,
                trace=trace,
            )
        )
        state = new_state
        outcome = result_outcome  # type: ignore[assignment]
        if outcome != "ongoing":
            break
    else:
        outcome = "phase_limit_reached"

    print_fn(render_board(state))
    print_fn(f"Game over: {outcome}")
    return SessionResult(final_state=state, outcome=outcome, phases=tuple(phases))
