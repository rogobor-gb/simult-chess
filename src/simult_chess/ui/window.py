"""A local Tk window: click-to-play with live legality, threats, and a trace
view (Phase 16.3; hardened after maintainer playtest, 2026-07-26).

Tkinter ships with CPython, so this adds no dependency. Most of the window is
throwaway relative to a future web client; the parts that are **not** throwaway
are the two engine-side primitives it consumes — `core.legality`'s
`check_partial_program` (16.1) and `ui.affordances` (16.2) — which a web client
needs identically.

The design keeps all game logic in a **pure** :class:`ProgramBuilder` (a
click-to-build state machine, fully testable headless) and in
:func:`describe_trace` (the resolution view's text). The Tk
:class:`SimultChessWindow` is a thin renderer over them — it holds a `mode`
(Move / Reserve / Cancel, since a click means something different in each) and
otherwise only ever asks `ProgramBuilder` whether an action is legal, so the
important behaviour is tested without a display.

There is no *check* in this game — kings are simply captured — so a new player
learns why anything happened only from the **resolution view** (every
`PhiTrace` category rendered, in plain language) and the **threat overlay**
(declaration-time attack potential). Both are first-class here.
"""

from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

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
from simult_chess.referee.record import (
    MatchOutcome,
    TerminationReason,
    normalize_outcome_and_reason,
)
from simult_chess.rules.ruleset import RuleSet
from simult_chess.ui import notation
from simult_chess.ui.affordances import Affordances, affordances, threat_overlay

if TYPE_CHECKING:
    import tkinter as tk

#: Every `PhiTrace` field, so :func:`describe_trace` is exhaustive by
#: construction (a new trace field must be added here to compile the test).
#: Ordered narratively: what got captured/recaptured first, then collisions
#: and fizzles, then bookkeeping, with the two most redundant-to-a-player
#: categories (a move that merely displaced, whether or not it later died) last.
TRACE_CATEGORIES: tuple[str, ...] = (
    "captured",
    "fired",
    "annihilated",
    "fizzled",
    "cancelled",
    "promoted",
    "survivors",
    "executing",
)

_FRIENDLY_HEADER: dict[str, str] = {
    "captured": "Captured",
    "fired": "Recaptures (reservations that fired)",
    "annihilated": "Head-on collisions (both pieces destroyed)",
    "fizzled": "Had no effect (fizzled)",
    "cancelled": "Reservations cancelled",
    "promoted": "Promoted",
    "survivors": "Moves that went through",
    "executing": "Declared moves that displaced",
}

_FIZZLE_CAUSE_TEXT: dict[str, str] = {
    "F1": "the target had already moved away (vacated-square fizzle)",
    "F2": "it collided with another pawn arriving on the same square",
}

_PIECE_NAME: dict[PieceType, str] = {
    "p": "pawn", "n": "knight", "b": "bishop",
    "r": "rook", "q": "queen", "k": "king",
}


def _token_label(token: Token) -> str:
    return f"{token.color.value} {_PIECE_NAME[token.typ]} (#{token.id})"


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
    Fizzle causes are spelled out in plain language rather than the raw `F1`/
    `F2` invariant codes.
    """
    return {
        "captured": [
            f"{_token_label(t)} was captured on {notation.format_square(sq)}"
            for t, sq in trace.captured
        ],
        "fired": [
            f"{_token_label(r.defender)} fired its reservation and recaptured "
            f"on {notation.format_square(r.square)}, taking {_token_label(r.captured)}"
            for r in trace.fired
        ],
        "annihilated": [
            f"{_declared_label(e.white_move)} and {_declared_label(e.black_move)} "
            "collided in mid-path — both pieces are destroyed"
            for e in trace.annihilated
        ],
        "fizzled": [
            f"{_declared_label(o.move)} had no effect: "
            f"{_FIZZLE_CAUSE_TEXT.get(o.cause, o.cause)}"
            for o in trace.fizzled
        ],
        "cancelled": [
            f"the reservation {_token_label(r.defender)} → {_token_label(r.protege)} "
            "was cancelled"
            for r in trace.cancelled
        ],
        "promoted": [
            f"a pawn promoted (token #{tid})" for tid in sorted(trace.promoted)
        ],
        "survivors": [_declared_label(m) for m in trace.survivors],
        "executing": [_declared_label(m) for m in trace.executing],
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

    def reserve_targets_for(self, defender_id: int) -> frozenset[int]:
        """Own token ids `defender_id` may legally be declared to defend right now."""
        return frozenset(
            protege_id
            for d, protege_id in self.current_affordances().reservation_pairings
            if d == defender_id
        )

    def cancellable_reservation_defenders(self) -> frozenset[int]:
        """Own token ids that are defenders of a currently-cancellable reservation."""
        reservations = self.state.reservations(self.color)
        cancellable = self.current_affordances().cancellable_reservations
        return frozenset(reservations[i].defender.id for i in cancellable)

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

    def cancel_reservation_by_defender(self, defender_id: int) -> bool:
        """Cancel the (first) cancellable reservation defended by `defender_id`."""
        reservations = self.state.reservations(self.color)
        cancellable = self.current_affordances().cancellable_reservations
        for index, reservation in enumerate(reservations):
            if reservation.defender.id == defender_id and index in cancellable:
                return self.add_cancel(index)
        return False

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

    def is_cooled(self, token_id: int) -> bool:
        token = self._token(token_id)
        return token is not None and token in self.state.cooldown

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
_FILL_LIGHT, _FILL_DARK = "#ecd9b0", "#b58863"
_FILL_MOVE_DEST = "#9bd07a"        # legal move / reservation-target square
_FILL_CANCELLABLE = "#f0c060"      # a defender you can click to cancel
_OUTLINE_THREAT = "#d02020"
_OUTLINE_SELECTED = "#2060d0"
_COOLDOWN_DOT = "#707070"

Mode = Literal["move", "reserve", "cancel"]

_REASON_SENTENCE: dict[TerminationReason, str] = {
    "regicide": "a king was captured",
    "repetition": "the same position repeated three times",
    "no_progress": "no captures or pawn moves for too long (the no-progress horizon)",
    "phase_limit": "the phase limit was reached",
}
_OUTCOME_PHRASE: dict[MatchOutcome, str] = {
    "white_wins": "White wins",
    "black_wins": "Black wins",
    "draw": "Draw",
}


class SimultChessWindow:
    """A minimal Tk window: pick a mode, click pieces/squares, submit.

    Human-vs-agent by default (the human is `human_color`, the agent plays the
    other side). All rules questions go through :class:`ProgramBuilder`, so the
    window itself holds no game logic beyond *which* builder method a click
    dispatches to. Construction requires a display; the pure behaviour it
    drives is covered by the `ProgramBuilder`/`describe_trace` tests, and a
    headless smoke test skips when Tk cannot open.
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
        self.mode: Mode = "move"
        self._game_over = False

        self.root = root if root is not None else tk.Tk()
        self.root.title("Simultaneous Chess")

        self.canvas = tk.Canvas(
            self.root, width=8 * _CELL, height=8 * _CELL, highlightthickness=0
        )
        self.canvas.grid(row=0, column=0, rowspan=9)
        self.canvas.bind("<Button-1>", self._on_click)

        # row 0: transient click feedback ("not a legal destination", ...)
        self.status = tk.Label(
            self.root, text="", anchor="w", width=44, wraplength=320, justify="left"
        )
        self.status.grid(row=0, column=1, sticky="w")

        # row 1: persistent summary of the program built so far this phase
        self.program_label = tk.Label(
            self.root, text="", anchor="w", width=44, fg="#333333"
        )
        self.program_label.grid(row=1, column=1, sticky="w")

        # row 2: Move / Reserve / Cancel mode — a click means something
        # different in each, since Move/Reserve/Cancel take different numbers
        # of clicks and different kinds of targets.
        mode_frame = tk.Frame(self.root)
        mode_frame.grid(row=2, column=1, sticky="w")
        self._mode_buttons: dict[Mode, tk.Button] = {}
        mode_choices: tuple[tuple[Mode, str], ...] = (
            ("move", "Move"), ("reserve", "Reserve"), ("cancel", "Cancel"),
        )
        for mode, label in mode_choices:
            button = tk.Button(
                mode_frame, text=label, command=functools.partial(self._set_mode, mode)
            )
            button.pack(side="left")
            self._mode_buttons[mode] = button

        # row 3: castling (not reachable by clicking the king two squares over
        # — that's a distinct Action, not a Move — so it gets its own buttons)
        castle_frame = tk.Frame(self.root)
        castle_frame.grid(row=3, column=1, sticky="w")
        self.castle_king_btn = tk.Button(
            castle_frame, text="O-O", command=lambda: self._try_castle("king")
        )
        self.castle_king_btn.pack(side="left")
        self.castle_queen_btn = tk.Button(
            castle_frame, text="O-O-O", command=lambda: self._try_castle("queen")
        )
        self.castle_queen_btn.pack(side="left")

        # row 4: colour legend
        legend = (
            "green = legal target this click · orange = click to cancel\n"
            "blue outline = selected · red outline = threatened next phase\n"
            "grey dot = cooling down (cannot act this phase)"
        )
        tk.Label(
            self.root, text=legend, anchor="w", justify="left", fg="#555555",
            font=("", 9),
        ).grid(row=4, column=1, sticky="w")

        # rows 5-7: the resolution (trace) view
        self.trace = tk.Text(self.root, width=44, height=16)
        self.trace.grid(row=5, column=1, rowspan=3)

        # row 8: submit / undo
        self.submit_btn = tk.Button(self.root, text="Submit", command=self._submit)
        self.submit_btn.grid(row=8, column=1, sticky="w")
        self.undo_btn = tk.Button(self.root, text="Undo", command=self._undo)
        self.undo_btn.grid(row=8, column=1)

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

    def _set_status(self, text: str) -> None:
        self.status.config(text=text)

    # -- mode / interaction --------------------------------------------------

    def _set_mode(self, mode: Mode) -> None:
        if self._game_over:
            return
        self.mode = mode
        self._selected = None
        self._set_status(f"mode: {mode} — " + {
            "move": "click a piece, then a highlighted square",
            "reserve": "click the defender, then the piece it should defend",
            "cancel": "click a defender (orange) to cancel its reservation",
        }[mode])
        self._render()

    def _on_click(self, event: tk.Event) -> None:
        if self._game_over:
            return
        square = self._square_at(event.x, event.y)
        if square is None:
            return
        if self.mode == "move":
            self._click_move(square)
        elif self.mode == "reserve":
            self._click_reserve(square)
        else:
            self._click_cancel(square)
        self._render()

    def _click_move(self, square: Square) -> None:
        occupant = self._occupant(square)
        if occupant is not None and occupant.color is self.human_color:
            if occupant.id == self._selected:
                self._selected = None
                self._set_status("selection cleared")
                return
            self._selected = occupant.id
            name = _PIECE_NAME[occupant.typ]
            if self.builder.is_cooled(occupant.id):
                self._set_status(
                    f"that {name} is cooling down and cannot act this phase"
                )
            else:
                self._set_status(f"selected the {name} — click a highlighted square")
            return
        if self._selected is None:
            self._set_status("click one of your own pieces first")
            return
        if self.builder.add_move(self._selected, square):
            self._set_status("move added")
            self._selected = None
        else:
            self._set_status(
                "not a legal destination for that piece — pick a highlighted square"
            )
            # Keep the selection: the highlighted destinations stay visible so
            # the player can retry without reselecting the piece.

    def _click_reserve(self, square: Square) -> None:
        occupant = self._occupant(square)
        if occupant is None or occupant.color is not self.human_color:
            self._set_status("a reservation pairs two of your own pieces — click one")
            return
        if self._selected is None:
            self._selected = occupant.id
            name = _PIECE_NAME[occupant.typ]
            self._set_status(
                f"defender: the {name}. now click the piece it should defend"
            )
            return
        if occupant.id == self._selected:
            self._selected = None
            self._set_status("selection cleared")
            return
        if self.builder.add_reserve(self._selected, occupant.id):
            self._set_status("reservation added")
        else:
            self._set_status("that pairing isn't a legal reservation right now")
        self._selected = None

    def _click_cancel(self, square: Square) -> None:
        occupant = self._occupant(square)
        if occupant is None or occupant.color is not self.human_color:
            self._set_status("click one of your pieces that is defending a reservation")
            return
        if self.builder.cancel_reservation_by_defender(occupant.id):
            self._set_status("reservation cancelled")
        else:
            self._set_status("that piece isn't defending a cancellable reservation")

    def _try_castle(self, side: CastleSide) -> None:
        if self._game_over:
            return
        if self.builder.add_castle(side):
            self._set_status("castled" if side == "king" else "castled queenside")
        else:
            self._set_status("that castle isn't legal right now")
        self._selected = None
        self._render()

    def _undo(self) -> None:
        if self._game_over:
            return
        self.builder.undo()
        self._selected = None
        self._render()

    def _submit(self) -> None:
        if self._game_over:
            return
        if not self.builder.is_submittable():
            self._set_status("illegal: " + self._first_error())
            return
        agent = self.agent
        if agent is None:
            self._set_status("no agent set")
            return
        human_program = self.builder.program
        agent_color = self.human_color.opponent
        agent_program = agent(self.state, agent_color, self.ruleset, self.rng)
        programs = {self.human_color: human_program, agent_color: agent_program}
        result = phi(
            self.state, programs[Color.WHITE], programs[Color.BLACK], self.ruleset
        )
        self._show_trace(
            result.trace, previous_state=self.state, new_state=result.state
        )
        self.state = result.state
        self.builder = ProgramBuilder(self.state, self.human_color, self.ruleset)
        self._selected = None

        if result.outcome != "ongoing":
            outcome, reason = normalize_outcome_and_reason(
                self.state, self.ruleset, result.outcome
            )
            self._game_over = True
            phrase = _OUTCOME_PHRASE[outcome]
            reason_text = _REASON_SENTENCE.get(reason, reason)
            self._set_status(f"GAME OVER — {phrase}: {reason_text}")
            self._disable_controls()
        else:
            self._set_status(f"phase resolved: {result.outcome}")
        self._render()

    def _disable_controls(self) -> None:
        for button in (
            self.submit_btn, self.undo_btn,
            self.castle_king_btn, self.castle_queen_btn,
            *self._mode_buttons.values(),
        ):
            button.config(state=self._tk.DISABLED)

    def _first_error(self) -> str:
        errors = self.builder.submission_errors()
        return f"{errors[0].invariant_id}: {errors[0].detail}" if errors else ""

    def _show_trace(
        self, trace: PhiTrace, *, previous_state: State, new_state: State
    ) -> None:
        del previous_state  # kept for a future "before/after" view; unused today
        self.trace.delete("1.0", self._tk.END)
        rendered = describe_trace(trace)
        promoted_labels = self._promoted_labels(trace, new_state)
        for category in TRACE_CATEGORIES:
            lines = (
                promoted_labels if category == "promoted" else rendered[category]
            )
            if not lines:
                continue
            self.trace.insert(self._tk.END, f"— {_FRIENDLY_HEADER[category]} —\n")
            for line in lines:
                self.trace.insert(self._tk.END, f"  {line}\n")

    def _promoted_labels(self, trace: PhiTrace, new_state: State) -> list[str]:
        """Enrich `promoted` with the post-move square/type (needs live state)."""
        by_id = {token.id: (token, sq) for token, sq in new_state.board.items()}
        labels = []
        for token_id in sorted(trace.promoted):
            found = by_id.get(token_id)
            if found is None:
                labels.append(f"a pawn promoted (token #{token_id})")
                continue
            token, square = found
            square_txt = notation.format_square(square)
            labels.append(f"promoted to {_PIECE_NAME[token.typ]} on {square_txt}")
        return labels

    # -- rendering ---------------------------------------------------------

    def _render(self) -> None:
        self.canvas.delete("all")
        aff = None if self._game_over else self.builder.current_affordances()
        if aff is None:
            move_dests: frozenset[Square] = frozenset()
            threatened: frozenset[int] = frozenset()
            cancellable: frozenset[int] = frozenset()
        else:
            threatened = aff.threatened_tokens
            cancellable = (
                self.builder.cancellable_reservation_defenders()
                if self.mode == "cancel"
                else frozenset()
            )
            if self.mode == "move" and self._selected is not None:
                move_dests = aff.legal_destinations.get(self._selected, frozenset())
            elif self.mode == "reserve" and self._selected is not None:
                target_ids = self.builder.reserve_targets_for(self._selected)
                move_dests = frozenset(
                    sq for tok, sq in self.state.board.items() if tok.id in target_ids
                )
            else:
                move_dests = frozenset()

        for rank in range(8):
            for file in range(8):
                square = Square(file=file, rank=rank)
                x0, y0 = file * _CELL, (7 - rank) * _CELL
                light = (file + rank) % 2 == 1
                fill = _FILL_LIGHT if light else _FILL_DARK
                token = self._occupant(square)
                if square in move_dests:
                    fill = _FILL_MOVE_DEST
                elif token is not None and token.id in cancellable:
                    fill = _FILL_CANCELLABLE
                self.canvas.create_rectangle(
                    x0, y0, x0 + _CELL, y0 + _CELL, fill=fill, outline=""
                )
                if token is None:
                    continue
                colour = "#f8f8f8" if token.color is Color.WHITE else "#101010"
                outline = _OUTLINE_THREAT if token.id in threatened else ""
                if token.id == self._selected:
                    outline = _OUTLINE_SELECTED
                if outline:
                    self.canvas.create_rectangle(
                        x0 + 2, y0 + 2, x0 + _CELL - 2, y0 + _CELL - 2,
                        outline=outline, width=3,
                    )
                self.canvas.create_text(
                    x0 + _CELL // 2, y0 + _CELL // 2,
                    text=_PIECE_GLYPH[token.typ], fill=colour, font=("", 30),
                )
                if token in self.state.cooldown:
                    self.canvas.create_oval(
                        x0 + 4, y0 + 4, x0 + 14, y0 + 14,
                        fill=_COOLDOWN_DOT, outline="",
                    )

        if aff is not None:
            self.castle_king_btn.config(
                state=self._tk.NORMAL if "king" in aff.legal_castles
                else self._tk.DISABLED
            )
            self.castle_queen_btn.config(
                state=self._tk.NORMAL if "queen" in aff.legal_castles
                else self._tk.DISABLED
            )
            ready = "✓ submittable" if self.builder.is_submittable() else "building…"
            actions = "; ".join(
                notation.format_action(a, self.state, self.human_color)
                for a in self.builder.program
            )
            mode_tag = f"[{self.mode}]"
            self.program_label.config(
                text=f"{mode_tag} program: {actions or '(empty)'}  {ready}"
            )

    def render_ascii(self) -> str:
        """The board as text (for a headless assertion of current state)."""
        from simult_chess.ui.board_render import render_board

        return render_board(self.state)
