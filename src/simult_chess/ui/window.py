"""A local Tk window: click-to-play with live legality, threats, and a trace
view (Phase 16.3).

Tkinter ships with CPython, so this adds no dependency. Most of the window is
throwaway relative to a future web client; the parts that are **not** throwaway
are the two engine-side primitives it consumes — `core.legality`'s
`check_partial_program` (16.1) and `ui.affordances` (16.2) — which a web client
needs identically.

The design keeps all game logic in a **pure** :class:`ProgramBuilder` (a
click-to-build state machine, fully testable headless) and in
:func:`describe_trace` (the resolution view's text). The Tk
:class:`SimultChessWindow` is a thin renderer over them, so the important
behaviour is tested without a display.

There is no *check* in this game — kings are simply captured — so a new player
learns why anything happened only from the **resolution view** (every
`PhiTrace` category rendered) and the **threat overlay** (declaration-time
attack potential). Both are first-class here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from simult_chess.agents.base import Agent
from simult_chess.core.legality import check_legal_program
from simult_chess.core.moves import DeclaredMove
from simult_chess.core.phi import PhiTrace, phi
from simult_chess.core.types import (
    Cancel,
    Castle,
    CastleSide,
    Color,
    Move,
    PieceType,
    Program,
    Reserve,
    Square,
    State,
    Token,
)
from simult_chess.core.violation import Violation
from simult_chess.rules.ruleset import RuleSet
from simult_chess.ui import notation
from simult_chess.ui.affordances import Affordances, affordances, threat_overlay

if TYPE_CHECKING:
    import tkinter as tk

#: Every `PhiTrace` field, so :func:`describe_trace` is exhaustive by
#: construction (a new trace field must be added here to compile the test).
TRACE_CATEGORIES: tuple[str, ...] = (
    "fizzled",
    "executing",
    "annihilated",
    "survivors",
    "captured",
    "fired",
    "promoted",
    "cancelled",
)


def _token_label(token: Token) -> str:
    return f"{token.color.value}{token.typ}#{token.id}"


def _declared_label(move: DeclaredMove) -> str:
    letter = "" if move.token.typ == "p" else move.token.typ.upper()
    origin = notation.format_square(move.trajectory.origin)
    destination = notation.format_square(move.trajectory.destination)
    return f"{letter}{origin}{destination}"


def describe_trace(trace: PhiTrace) -> dict[str, list[str]]:
    """Render every `PhiTrace` category to human-readable lines for the UI.

    Returns a dict keyed by **every** entry of :data:`TRACE_CATEGORIES` (empty
    list when nothing of that kind happened), so a new player can read why each
    move fizzled and under which clause, which pairs annihilated, which
    reservations fired, and so on — the trace view that replaces "check".
    """
    return {
        "fizzled": [
            f"{_declared_label(o.move)} fizzled ({o.cause})" for o in trace.fizzled
        ],
        "executing": [_declared_label(m) for m in trace.executing],
        "annihilated": [
            f"{_declared_label(e.white_move)} ⇔ {_declared_label(e.black_move)}"
            for e in trace.annihilated
        ],
        "survivors": [_declared_label(m) for m in trace.survivors],
        "captured": [
            f"{_token_label(t)} captured on {notation.format_square(sq)}"
            for t, sq in trace.captured
        ],
        "fired": [
            f"{_token_label(r.defender)} recaptured on "
            f"{notation.format_square(r.square)}, taking {_token_label(r.captured)}"
            for r in trace.fired
        ],
        "promoted": [f"token #{tid} promoted" for tid in sorted(trace.promoted)],
        "cancelled": [
            f"reservation {_token_label(r.defender)} → {_token_label(r.protege)}"
            " cancelled"
            for r in trace.cancelled
        ],
    }


@dataclass(slots=True)
class ProgramBuilder:
    """A pure click-to-build state machine for one player's program.

    Holds the actions chosen so far and answers, at each step, what is legal to
    add next (via :func:`ui.affordances.affordances`, i.e. `check_partial_program`)
    and whether the whole program may be submitted (full `L`). No Tk, no I/O —
    the click flow can be driven and asserted in a test without a display.
    """

    state: State
    color: Color
    ruleset: RuleSet
    actions: list[Move | Castle | Reserve | Cancel] = field(default_factory=list)

    @property
    def program(self) -> Program:
        return tuple(self.actions)

    def current_affordances(self) -> Affordances:
        """What may legally be added to the program as it stands."""
        return affordances(self.state, self.color, self.ruleset, self.program)

    def legal_destinations(self, token_id: int) -> frozenset[Square]:
        """Squares `token_id` may move to right now (empty if none/already used)."""
        return self.current_affordances().legal_destinations.get(
            token_id, frozenset()
        )

    def add_move(
        self, token_id: int, destination: Square, promotion: PieceType | None = None
    ) -> bool:
        """Append the move `token_id`→`destination` if legal; report success."""
        if destination not in self.legal_destinations(token_id):
            return False
        move = self._build_move(token_id, destination, promotion)
        if move is None:
            return False
        self.actions.append(move)
        return True

    def add_castle(self, side: CastleSide) -> bool:
        if side not in self.current_affordances().legal_castles:
            return False
        self.actions.append(Castle(side=side))
        return True

    def add_reserve(self, defender_id: int, protege_id: int) -> bool:
        pairings = self.current_affordances().reservation_pairings
        if (defender_id, protege_id) not in pairings:
            return False
        defender, protege = self._token(defender_id), self._token(protege_id)
        if defender is None or protege is None:
            return False
        self.actions.append(Reserve(defender=defender, protege=protege))
        return True

    def add_cancel(self, reservation_index: int) -> bool:
        if reservation_index not in self.current_affordances().cancellable_reservations:
            return False
        reservation = self.state.reservations(self.color)[reservation_index]
        self.actions.append(Cancel(reservation=reservation))
        return True

    def undo(self) -> None:
        if self.actions:
            self.actions.pop()

    def clear(self) -> None:
        self.actions.clear()

    def submission_errors(self) -> list[Violation]:
        """Full-`L` violations blocking submit (empty ⇒ submittable)."""
        return check_legal_program(self.state, self.program, self.color, self.ruleset)

    def is_submittable(self) -> bool:
        return not self.submission_errors()

    def threatened_own_tokens(self) -> frozenset[int]:
        return threat_overlay(self.state, self.color)

    # -- internals ----------------------------------------------------------

    def _token(self, token_id: int) -> Token | None:
        for token in self.state.board:
            if token.id == token_id:
                return token
        return None

    def _build_move(
        self, token_id: int, destination: Square, promotion: PieceType | None
    ) -> Move | None:
        """Find the pseudo-legal trajectory of `token_id` ending at `destination`."""
        from simult_chess.core import geometry

        for token in self.state.board:
            if token.id != token_id:
                continue
            for trajectory in geometry.pseudo_legal_trajectories(self.state, token):
                if trajectory.destination == destination:
                    return Move(
                        token=token, trajectory=trajectory, promotion=promotion
                    )
        return None


_CELL = 56
_PIECE_GLYPH = {"p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚"}


class SimultChessWindow:
    """A minimal Tk window: click a token, click a destination, submit.

    Human-vs-agent by default (the human is `human_color`, the agent plays the
    other side). All rules questions go through :class:`ProgramBuilder`, so the
    window itself holds no game logic. Construction requires a display; the pure
    behaviour it drives is covered by the `ProgramBuilder`/`describe_trace`
    tests, and a headless smoke test skips when Tk cannot open.
    """

    def __init__(
        self,
        state: State,
        ruleset: RuleSet,
        *,
        human_color: Color = Color.WHITE,
        agent: Agent | None = None,
        seed: int = 0,
        root: tk.Tk | None = None,
    ) -> None:
        import tkinter as tk

        self._tk = tk
        self.ruleset = ruleset
        self.human_color = human_color
        self.agent = agent
        self.rng = random.Random(seed)
        self.state = state
        self.builder = ProgramBuilder(state, human_color, ruleset)
        self._selected: int | None = None

        self.root = root if root is not None else tk.Tk()
        self.root.title("Simultaneous Chess")
        self.canvas = tk.Canvas(
            self.root, width=8 * _CELL, height=8 * _CELL, highlightthickness=0
        )
        self.canvas.grid(row=0, column=0, rowspan=6)
        self.canvas.bind("<Button-1>", self._on_click)
        self.status = tk.Label(self.root, text="", anchor="w", width=40)
        self.status.grid(row=0, column=1, sticky="w")
        self.trace = tk.Text(self.root, width=40, height=18)
        self.trace.grid(row=1, column=1, rowspan=3)
        tk.Button(self.root, text="Submit", command=self._submit).grid(
            row=4, column=1, sticky="w"
        )
        tk.Button(self.root, text="Undo", command=self._undo).grid(
            row=5, column=1, sticky="w"
        )
        self._render()

    # -- geometry ----------------------------------------------------------

    def _square_at(self, x: int, y: int) -> Square | None:
        file, rank = x // _CELL, 7 - (y // _CELL)
        if 0 <= file < 8 and 0 <= rank < 8:
            return Square(file=file, rank=rank)
        return None

    def _occupant(self, square: Square) -> Token | None:
        for token, sq in self.state.board.items():
            if sq == square:
                return token
        return None

    # -- interaction -------------------------------------------------------

    def _on_click(self, event: tk.Event) -> None:
        square = self._square_at(event.x, event.y)
        if square is None:
            return
        occupant = self._occupant(square)
        if occupant is not None and occupant.color is self.human_color:
            self._selected = occupant.id
        elif self._selected is not None:
            self.builder.add_move(self._selected, square)
            self._selected = None
        self._render()

    def _undo(self) -> None:
        self.builder.undo()
        self._selected = None
        self._render()

    def _submit(self) -> None:
        if not self.builder.is_submittable():
            self.status.config(text="illegal: " + self._first_error())
            return
        human_program = self.builder.program
        agent = self.agent
        if agent is None:
            self.status.config(text="no agent set")
            return
        agent_color = self.human_color.opponent
        agent_program = agent(self.state, agent_color, self.ruleset, self.rng)
        programs = {self.human_color: human_program, agent_color: agent_program}
        result = phi(
            self.state, programs[Color.WHITE], programs[Color.BLACK], self.ruleset
        )
        self._show_trace(result.trace)
        self.state = result.state
        self.builder = ProgramBuilder(self.state, self.human_color, self.ruleset)
        self._selected = None
        self.status.config(text=f"resolved: {result.outcome}")
        self._render()

    def _first_error(self) -> str:
        errors = self.builder.submission_errors()
        return f"{errors[0].invariant_id}: {errors[0].detail}" if errors else ""

    def _show_trace(self, trace: PhiTrace) -> None:
        self.trace.delete("1.0", self._tk.END)
        for category, lines in describe_trace(trace).items():
            for line in lines:
                self.trace.insert(self._tk.END, f"[{category}] {line}\n")

    # -- rendering ---------------------------------------------------------

    def _render(self) -> None:
        self.canvas.delete("all")
        threatened = self.builder.threatened_own_tokens()
        dests = (
            self.builder.legal_destinations(self._selected)
            if self._selected is not None
            else frozenset()
        )
        for rank in range(8):
            for file in range(8):
                square = Square(file=file, rank=rank)
                x0, y0 = file * _CELL, (7 - rank) * _CELL
                light = (file + rank) % 2 == 1
                fill = "#ecd9b0" if light else "#b58863"
                if square in dests:
                    fill = "#9bd07a"  # legal destination
                self.canvas.create_rectangle(
                    x0, y0, x0 + _CELL, y0 + _CELL, fill=fill, outline=""
                )
                token = self._occupant(square)
                if token is None:
                    continue
                colour = "#f8f8f8" if token.color is Color.WHITE else "#101010"
                outline = "#d02020" if token.id in threatened else ""
                if token.id == self._selected:
                    outline = "#2060d0"
                if outline:
                    self.canvas.create_rectangle(
                        x0 + 2, y0 + 2, x0 + _CELL - 2, y0 + _CELL - 2,
                        outline=outline, width=3,
                    )
                self.canvas.create_text(
                    x0 + _CELL // 2, y0 + _CELL // 2,
                    text=_PIECE_GLYPH[token.typ], fill=colour, font=("", 30),
                )
        ready = "✓ submittable" if self.builder.is_submittable() else "building…"
        actions = "; ".join(
            notation.format_action(a, self.state, self.human_color)
            for a in self.builder.program
        )
        text = f"{self.human_color.value}: {actions or '(empty)'}  {ready}"
        self.status.config(text=text)

    def render_ascii(self) -> str:
        """The board as text (for a headless assertion of current state)."""
        from simult_chess.ui.board_render import render_board

        return render_board(self.state)
