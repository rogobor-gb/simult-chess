"""Phase 16.3: the pure ProgramBuilder, the trace view, and a Tk smoke test."""

from __future__ import annotations

import pytest

from simult_chess.agents.greedy import greedy_program
from simult_chess.core.phi import phi
from simult_chess.core.types import Color, Square
from simult_chess.referee.setup import standard_starting_state
from simult_chess.rules.ruleset import RuleSet
from simult_chess.ui.window import (
    TRACE_CATEGORIES,
    ProgramBuilder,
    describe_trace,
)

RULESET = RuleSet()


def _token_at(state: object, file: int, rank: int) -> object:
    for token, sq in state.board.items():  # type: ignore[attr-defined]
        if sq == Square(file=file, rank=rank):
            return token
    raise AssertionError("no token there")


# --- click-to-build without notation ------------------------------------------


def test_builder_click_flow_produces_a_submittable_program() -> None:
    state = standard_starting_state()
    builder = ProgramBuilder(state, Color.WHITE, RULESET)
    e_pawn = _token_at(state, 4, 1)

    # "Click" the pawn, then a legal destination — no notation typed.
    assert Square(4, 3) in builder.legal_destinations(e_pawn.id)  # type: ignore[attr-defined]
    assert builder.add_move(e_pawn.id, Square(4, 3))  # type: ignore[attr-defined]
    assert builder.is_submittable()

    # An illegal destination is refused and does not grow the program.
    assert not builder.add_move(e_pawn.id, Square(0, 0))  # type: ignore[attr-defined]
    assert len(builder.program) == 1


def test_builder_undo_and_l3_block() -> None:
    state = standard_starting_state()
    builder = ProgramBuilder(state, Color.WHITE, RULESET)
    e_pawn = _token_at(state, 4, 1)
    builder.add_move(e_pawn.id, Square(4, 3))  # type: ignore[attr-defined]
    # The same token cannot be used again (L3), so it has no destinations now.
    assert builder.legal_destinations(e_pawn.id) == frozenset()  # type: ignore[attr-defined]
    builder.undo()
    assert builder.program == ()
    assert Square(4, 3) in builder.legal_destinations(e_pawn.id)  # type: ignore[attr-defined]


def test_a_full_game_can_be_played_by_clicks_against_greedy() -> None:
    """No notation is ever typed: moves come from the affordance API alone."""
    import random

    state = standard_starting_state()
    rng = random.Random(0)
    for _ in range(20):
        builder = ProgramBuilder(state, Color.WHITE, RULESET)
        # Pick any token with a legal destination and click it there.
        dests = builder.current_affordances().legal_destinations
        token_id = next(t for t, squares in dests.items() if squares)
        destination = next(iter(dests[token_id]))
        assert builder.add_move(token_id, destination)
        assert builder.is_submittable()
        agent_program = greedy_program(state, Color.BLACK, RULESET, rng)
        result = phi(state, builder.program, agent_program, RULESET)
        state = result.state
        if result.outcome != "ongoing":
            break


# --- the resolution (trace) view ----------------------------------------------


def test_describe_trace_renders_every_category() -> None:
    """DoD: every PhiTrace category has a rendering."""
    state = standard_starting_state()
    e_pawn = _token_at(state, 4, 1)
    d_pawn = _token_at(state, 3, 6)
    from simult_chess.core.types import Move, Trajectory

    white = (
        Move(  # e2-e4
            token=e_pawn,  # type: ignore[arg-type]
            trajectory=Trajectory((Square(4, 1), Square(4, 2), Square(4, 3)), False),
        ),
    )
    black = (
        Move(  # d7-d5
            token=d_pawn,  # type: ignore[arg-type]
            trajectory=Trajectory((Square(3, 6), Square(3, 5), Square(3, 4)), False),
        ),
    )
    trace = phi(state, white, black, RULESET).trace
    rendered = describe_trace(trace)
    # Exhaustive by construction: a key for every category, none missing.
    assert set(rendered) == set(TRACE_CATEGORIES)
    # This ordinary phase at least executes and survives two moves.
    assert rendered["executing"]
    assert rendered["survivors"]


# --- Tk smoke (skips headless) ------------------------------------------------


def test_window_constructs_or_skips_headless() -> None:
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    root.withdraw()
    try:
        from simult_chess.ui.window import SimultChessWindow

        window = SimultChessWindow(
            standard_starting_state(), RULESET, agent=greedy_program, root=root
        )
        assert window.builder.color is Color.WHITE
        window._render()  # does not raise
    finally:
        root.destroy()
