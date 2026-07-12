from __future__ import annotations

import random

from simult_chess.agents.random_legal import random_legal_program
from simult_chess.core.types import Color
from simult_chess.referee.setup import standard_starting_state
from simult_chess.rules.ruleset import RuleSet
from simult_chess.ui.session import run_hot_seat, run_human_vs_agent

RULESET = RuleSet()


def test_run_hot_seat_resolves_one_phase() -> None:
    state = standard_starting_state()
    inputs = iter(["e4", "e5"])
    clears: list[bool] = []

    result = run_hot_seat(
        state,
        RULESET,
        input_fn=lambda _prompt: next(inputs),
        print_fn=lambda _line: None,
        clear_fn=lambda: clears.append(True),
        max_phases=1,
    )

    assert result.outcome == "phase_limit_reached"
    assert len(result.phases) == 1
    assert len(clears) == 2  # hides white's program, then black's, from the terminal
    board_by_id = {t.id: sq for t, sq in result.final_state.board.items()}
    white_pawn = next(
        t
        for t in state.board
        if state.board[t].file == 4 and t.color is Color.WHITE and t.typ == "p"
    )
    assert str(board_by_id[white_pawn.id]) == "e4"


def test_run_hot_seat_never_echoes_a_program_before_both_commit() -> None:
    state = standard_starting_state()
    inputs = iter(["e4", "e5"])
    printed: list[str] = []

    run_hot_seat(
        state,
        RULESET,
        input_fn=lambda _prompt: next(inputs),
        print_fn=printed.append,
        clear_fn=lambda: None,
        max_phases=1,
    )

    black_declare_index = next(
        i for i, line in enumerate(printed) if "Black to declare" in line
    )
    white_played_index = next(
        i for i, line in enumerate(printed) if "White played" in line
    )
    assert white_played_index > black_declare_index


def test_run_hot_seat_reprompts_on_illegal_program() -> None:
    state = standard_starting_state()
    # first white attempt exceeds the action budget (L1); second is legal
    inputs = iter(["e4; d4; c3", "e4", "e5"])
    printed: list[str] = []

    result = run_hot_seat(
        state,
        RULESET,
        input_fn=lambda _prompt: next(inputs),
        print_fn=printed.append,
        clear_fn=lambda: None,
        max_phases=1,
    )

    assert any("illegal (L1" in line for line in printed)
    assert len(result.phases) == 1


def test_run_human_vs_agent_resolves_one_phase() -> None:
    state = standard_starting_state()
    rng = random.Random(0)

    result = run_human_vs_agent(
        state,
        RULESET,
        Color.WHITE,
        random_legal_program,
        rng,
        input_fn=lambda _prompt: "e4",
        print_fn=lambda _line: None,
        clear_fn=lambda: None,
        max_phases=1,
    )

    assert result.outcome == "phase_limit_reached"
    assert len(result.phases) == 1
