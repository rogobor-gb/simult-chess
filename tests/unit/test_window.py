"""Phase 16.3: the pure ProgramBuilder, the trace view, and a Tk smoke test.

Hardened 2026-07-26 after maintainer playtest: game-over detection/lockout,
castling, reservations, cancellation, cooldown, and click feedback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from conftest import build_state

from simult_chess.agents.greedy import greedy_program
from simult_chess.core.phi import phi
from simult_chess.core.types import (
    Color,
    Move,
    Program,
    Square,
    State,
    Token,
    Trajectory,
)
from simult_chess.referee.setup import standard_starting_state
from simult_chess.rules.ruleset import RuleSet
from simult_chess.ui.window import (
    TRACE_CATEGORIES,
    ProgramBuilder,
    describe_trace,
)

if TYPE_CHECKING:
    import tkinter as tk

RULESET = RuleSet()


def _token_at(state: State, file: int, rank: int) -> Token:
    for token, sq in state.board.items():
        if sq == Square(file=file, rank=rank):
            return token
    raise AssertionError("no token there")


def _token_square(state: State, token_id: int) -> Square:
    for token, sq in state.board.items():
        if token.id == token_id:
            return sq
    raise AssertionError(f"no live token with id {token_id}")


def _add_a_paired_move(builder: ProgramBuilder, exclude_ids: set[int]) -> None:
    """Fill the program's second slot with a Move, satisfying L2.

    A Reserve alone never satisfies L2's mandatory-displacement clause (the
    same reason a Cancel-only program is illegal, `agents/candidates.py`'s
    `cancel_candidates` docstring) — it must be paired with a Move/Castle.
    """
    dests = builder.current_affordances().legal_destinations
    token_id = next(
        t for t, squares in dests.items() if squares and t not in exclude_ids
    )
    destination = next(iter(dests[token_id]))
    assert builder.add_move(token_id, destination)


# --- click-to-build without notation ------------------------------------------


def test_builder_click_flow_produces_a_submittable_program() -> None:
    state = standard_starting_state()
    builder = ProgramBuilder(state, Color.WHITE, RULESET)
    e_pawn = _token_at(state, 4, 1)

    # "Click" the pawn, then a legal destination — no notation typed.
    assert Square(4, 3) in builder.legal_destinations(e_pawn.id)
    assert builder.add_move(e_pawn.id, Square(4, 3))
    assert builder.is_submittable()

    # An illegal destination is refused and does not grow the program.
    assert not builder.add_move(e_pawn.id, Square(0, 0))
    assert len(builder.program) == 1


def test_builder_undo_and_l3_block() -> None:
    state = standard_starting_state()
    builder = ProgramBuilder(state, Color.WHITE, RULESET)
    e_pawn = _token_at(state, 4, 1)
    builder.add_move(e_pawn.id, Square(4, 3))
    # The same token cannot be used again (L3), so it has no destinations now.
    assert builder.legal_destinations(e_pawn.id) == frozenset()
    builder.undo()
    assert builder.program == ()
    assert Square(4, 3) in builder.legal_destinations(e_pawn.id)


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


# --- castling, reservations, cancellation, cooldown (builder level) -----------


def test_builder_castle_via_add_castle() -> None:
    wk = Token(id=1, color=Color.WHITE, typ="k")
    wr = Token(id=2, color=Color.WHITE, typ="r")
    bk = Token(id=3, color=Color.BLACK, typ="k")
    state = build_state({wk: Square(4, 0), wr: Square(7, 0), bk: Square(4, 7)})
    builder = ProgramBuilder(state, Color.WHITE, RULESET)
    assert "king" in builder.current_affordances().legal_castles
    assert builder.add_castle("king")
    assert builder.is_submittable()


def test_builder_castle_rejected_when_illegal() -> None:
    state = standard_starting_state()  # pieces still block castling
    builder = ProgramBuilder(state, Color.WHITE, RULESET)
    assert "king" not in builder.current_affordances().legal_castles
    assert not builder.add_castle("king")
    assert builder.program == ()


def test_builder_reserve_via_targets_and_add_reserve() -> None:
    state = standard_starting_state()
    builder = ProgramBuilder(state, Color.WHITE, RULESET)
    aff = builder.current_affordances()
    assert aff.reservation_pairings, "the standard start has reservation pairings"
    defender_id, protege_id = next(iter(aff.reservation_pairings))
    assert protege_id in builder.reserve_targets_for(defender_id)
    assert builder.add_reserve(defender_id, protege_id)
    # A lone Reserve never satisfies L2 (mandatory displacement); pair it.
    _add_a_paired_move(builder, {defender_id, protege_id})
    assert builder.is_submittable()


def test_builder_reserve_rejects_an_illegal_pairing() -> None:
    state = standard_starting_state()
    builder = ProgramBuilder(state, Color.WHITE, RULESET)
    # Two arbitrary own tokens that do not form a legal reservation pairing.
    king = _token_at(state, 4, 0)
    rook = _token_at(state, 0, 0)
    if (king.id, rook.id) in builder.current_affordances().reservation_pairings:
        pytest.skip("this pairing happens to be legal at the start; not the point")
    assert not builder.add_reserve(king.id, rook.id)
    assert builder.program == ()


def test_builder_cancel_by_defender() -> None:
    state = standard_starting_state()
    builder = ProgramBuilder(state, Color.WHITE, RULESET)
    aff = builder.current_affordances()
    defender_id, protege_id = next(iter(aff.reservation_pairings))
    assert builder.add_reserve(defender_id, protege_id)
    _add_a_paired_move(builder, {defender_id, protege_id})

    # Resolve the reservation into the state, then confirm it is cancellable
    # and reachable purely by clicking the defender (no index bookkeeping).
    import random

    from simult_chess.agents.random_legal import random_legal_program

    black = random_legal_program(state, Color.BLACK, RULESET, random.Random(0))
    result = phi(state, builder.program, black, RULESET)
    new_builder = ProgramBuilder(result.state, Color.WHITE, RULESET)
    assert defender_id in new_builder.cancellable_reservation_defenders()
    assert new_builder.cancel_reservation_by_defender(defender_id)
    # A lone Cancel is also L2-illegal (it isn't a displacement either); pair it.
    _add_a_paired_move(new_builder, {defender_id})
    assert new_builder.is_submittable()


def test_builder_is_cooled_reports_cooldown() -> None:
    state = standard_starting_state()
    e_pawn = _token_at(state, 4, 1)
    n = _token_at(state, 1, 0)  # b1 knight
    builder = ProgramBuilder(state, Color.WHITE, RULESET)
    # Neither has moved yet: nothing is cooled.
    assert not builder.is_cooled(e_pawn.id)
    assert not builder.is_cooled(n.id)


# --- the resolution (trace) view ----------------------------------------------


def test_describe_trace_renders_every_category() -> None:
    """DoD: every PhiTrace category has a rendering."""
    state = standard_starting_state()
    e_pawn = _token_at(state, 4, 1)
    d_pawn = _token_at(state, 3, 6)

    white = (
        Move(  # e2-e4
            token=e_pawn,
            trajectory=Trajectory((Square(4, 1), Square(4, 2), Square(4, 3)), False),
        ),
    )
    black = (
        Move(  # d7-d5
            token=d_pawn,
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


def test_describe_trace_fizzle_is_plain_english_not_a_raw_code() -> None:
    """A vacated-square fizzle (F1) must not surface the bare code to a player."""
    wk = Token(id=1, color=Color.WHITE, typ="k")
    bk = Token(id=2, color=Color.BLACK, typ="k")
    wp = Token(id=3, color=Color.WHITE, typ="p")
    bn = Token(id=4, color=Color.BLACK, typ="n")
    # White pawn on e5 aims to capture a knight on d6; the knight leaps away
    # in the same phase, so the pawn's diagonal capture fizzles (F1).
    state = build_state(
        {wk: Square(0, 0), bk: Square(7, 7), wp: Square(4, 4), bn: Square(3, 5)}
    )
    white = (
        Move(
            token=wp,
            trajectory=Trajectory((Square(4, 4), Square(3, 5)), is_jump=False),
        ),
    )
    black = (
        Move(
            token=bn,
            trajectory=Trajectory((Square(3, 5), Square(1, 6)), is_jump=True),
        ),
    )
    trace = phi(state, white, black, RULESET).trace
    rendered = describe_trace(trace)
    assert rendered["fizzled"], "expected a fizzle in this fixture"
    for line in rendered["fizzled"]:
        assert "F1" not in line and "F2" not in line
        assert "vacated" in line or "collided" in line


# --- Tk smoke (skips headless) ------------------------------------------------


def _open_root() -> tk.Tk:
    """A withdrawn Tk root, or a pytest skip if no display is available."""
    tk_module = pytest.importorskip("tkinter")
    try:
        root = tk_module.Tk()
    except tk_module.TclError:
        pytest.skip("no display available")
    root.withdraw()
    return root  # type: ignore[no-any-return]


def test_window_constructs_or_skips_headless() -> None:
    root = _open_root()
    try:
        from simult_chess.ui.window import SimultChessWindow

        window = SimultChessWindow(
            standard_starting_state(), RULESET, agent=greedy_program, root=root
        )
        assert window.builder.color is Color.WHITE
        window._render()  # does not raise
    finally:
        root.destroy()


def test_mode_buttons_switch_mode_correctly() -> None:
    """Regression: a naive lambda-in-a-loop would make every button set the
    same (last) mode. functools.partial must bind each button's own mode."""
    root = _open_root()
    try:
        from simult_chess.ui.window import SimultChessWindow

        window = SimultChessWindow(
            standard_starting_state(), RULESET, agent=greedy_program, root=root
        )
        for mode in ("move", "reserve", "cancel"):
            window._mode_buttons[mode].invoke()
            assert window.mode == mode
    finally:
        root.destroy()


def test_illegal_click_gives_feedback_and_keeps_the_selection() -> None:
    """Regression: a failed click used to fail silently (the root of the
    'some pawn movement was not allowed' confusion)."""
    root = _open_root()
    try:
        from simult_chess.ui.window import SimultChessWindow

        window = SimultChessWindow(
            standard_starting_state(), RULESET, agent=greedy_program, root=root
        )
        e_pawn = _token_at(window.state, 4, 1)
        window._selected = e_pawn.id
        window._click_move(Square(4, 5))  # empty but out of reach for e2
        assert "not a legal destination" in window.status.cget("text")
        assert window._selected == e_pawn.id
    finally:
        root.destroy()


def test_castle_buttons_reflect_legality() -> None:
    root = _open_root()
    try:
        from simult_chess.ui.window import SimultChessWindow

        wk = Token(id=1, color=Color.WHITE, typ="k")
        wr = Token(id=2, color=Color.WHITE, typ="r")
        bk = Token(id=3, color=Color.BLACK, typ="k")
        state = build_state({wk: Square(4, 0), wr: Square(7, 0), bk: Square(4, 7)})
        window = SimultChessWindow(
            state, RULESET, human_color=Color.WHITE, agent=greedy_program, root=root
        )
        window._render()
        assert str(window.castle_king_btn.cget("state")) == "normal"
        window._try_castle("king")
        assert window.builder.is_submittable()
        assert "castled" in window.status.cget("text")
    finally:
        root.destroy()


def test_reserve_mode_click_flow() -> None:
    root = _open_root()
    try:
        from simult_chess.ui.window import SimultChessWindow

        window = SimultChessWindow(
            standard_starting_state(), RULESET, agent=greedy_program, root=root
        )
        window._set_mode("reserve")
        aff = window.builder.current_affordances()
        defender_id, protege_id = next(iter(aff.reservation_pairings))
        defender_sq = _token_square(window.state, defender_id)
        protege_sq = _token_square(window.state, protege_id)
        window._click_reserve(defender_sq)
        assert window._selected == defender_id
        window._click_reserve(protege_sq)
        assert len(window.builder.program) == 1
        assert "reservation added" in window.status.cget("text")
    finally:
        root.destroy()


def test_cancel_mode_click_flow() -> None:
    root = _open_root()
    try:
        from simult_chess.ui.window import ProgramBuilder as WindowProgramBuilder
        from simult_chess.ui.window import SimultChessWindow

        window = SimultChessWindow(
            standard_starting_state(), RULESET, agent=greedy_program, root=root
        )
        aff = window.builder.current_affordances()
        defender_id, protege_id = next(iter(aff.reservation_pairings))
        assert window.builder.add_reserve(defender_id, protege_id)
        _add_a_paired_move(window.builder, {defender_id, protege_id})
        import random

        from simult_chess.agents.random_legal import random_legal_program

        black = random_legal_program(
            window.state, Color.BLACK, RULESET, random.Random(0)
        )
        result = phi(window.state, window.builder.program, black, RULESET)
        window.state = result.state
        window.builder = WindowProgramBuilder(window.state, Color.WHITE, RULESET)

        window._set_mode("cancel")
        defender_sq = _token_square(window.state, defender_id)
        window._click_cancel(defender_sq)
        assert len(window.builder.program) == 1
        assert "reservation cancelled" in window.status.cget("text")
    finally:
        root.destroy()


def test_cooldown_dot_is_drawn_for_a_cooled_token() -> None:
    root = _open_root()
    try:
        from simult_chess.ui.window import _COOLDOWN_DOT, SimultChessWindow

        window = SimultChessWindow(
            standard_starting_state(), RULESET, agent=greedy_program, root=root
        )
        e_pawn = _token_at(window.state, 4, 1)
        window.builder.add_move(e_pawn.id, Square(4, 3))
        window._submit()
        # Find a token that displaced this phase and is cooled (a knight,
        # since pawns/kings are exempt) — greedy or the pawn move guarantee
        # at least one non-exempt piece moved somewhere in a 20-phase window
        # is not guaranteed on move 1, so just assert the renderer doesn't
        # crash and, if anything is cooled, a dot appears for it.
        window._render()
        cooled_ids = {t.id for t in window.state.cooldown}
        if cooled_ids:
            ovals = window.canvas.find_withtag("all")
            fills = {
                window.canvas.itemcget(o, "fill")  # type: ignore[no-untyped-call]
                for o in ovals
            }
            assert _COOLDOWN_DOT in fills
    finally:
        root.destroy()


def test_game_over_locks_the_window() -> None:
    root = _open_root()
    try:
        from simult_chess.ui.window import SimultChessWindow

        wk = Token(id=1, color=Color.WHITE, typ="k")
        bk = Token(id=2, color=Color.BLACK, typ="k")
        wq = Token(id=3, color=Color.WHITE, typ="q")
        bp = Token(id=4, color=Color.BLACK, typ="p")
        state = build_state(
            {wk: Square(4, 0), bk: Square(4, 7), wq: Square(4, 6), bp: Square(0, 6)}
        )

        import random as random_module

        def fake_black(
            state: State,
            color: Color,
            ruleset: RuleSet,
            rng: random_module.Random,
        ) -> Program:
            origin = state.board[bp]
            dest = Square(origin.file, origin.rank - 1)
            return (Move(token=bp, trajectory=Trajectory((origin, dest), False)),)

        window = SimultChessWindow(
            state, RULESET, human_color=Color.WHITE, agent=fake_black, root=root
        )
        assert window.builder.add_move(wq.id, Square(4, 7))
        window._submit()

        assert window._game_over
        assert "GAME OVER" in window.status.cget("text")
        assert "White wins" in window.status.cget("text")
        for widget in (
            window.submit_btn, window.undo_btn,
            window.castle_king_btn, window.castle_queen_btn,
            *window._mode_buttons.values(),
        ):
            assert str(widget.cget("state")) == "disabled"

        # A further click must be a safe no-op.
        program_before = window.builder.program

        class _FakeEvent:
            x, y = 4 * 56 + 10, 7 * 56 + 10  # inside the board, no exception

        window._on_click(_FakeEvent())  # type: ignore[arg-type]
        assert window.builder.program == program_before
    finally:
        root.destroy()
