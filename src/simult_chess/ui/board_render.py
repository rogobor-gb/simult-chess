"""ASCII board rendering from public state only (dev brief §3.5, Phase 7).

Renders `State.board` and `State.cooldown` -- both fully public in the
perfect-information base game (spec §11.5): concealment in this variant is
only about an *in-flight, not-yet-both-committed* program, never about the
board itself. Reservations are intentionally not drawn on the board; they
are listed separately by the caller (e.g. via `State.reservations`).
"""

from __future__ import annotations

from simult_chess.core import geometry
from simult_chess.core.types import Color, PieceType, Square, State

_TYP_LETTER: dict[PieceType, str] = {
    "p": "P", "n": "N", "b": "B", "r": "R", "q": "Q", "k": "K",
}
_FILES = "abcdefgh"


def render_board(state: State, perspective: Color = Color.WHITE) -> str:
    """Render `state.board` as an ASCII grid, `perspective`'s home rank at bottom.

    A cooled token's cell is suffixed with ``*``. Empty squares render as
    ``.``.
    """
    occupant = geometry.occupant_lookup(state.board)
    ranks = range(7, -1, -1) if perspective is Color.WHITE else range(8)
    files = range(8) if perspective is Color.WHITE else range(7, -1, -1)

    lines: list[str] = []
    for rank in ranks:
        cells: list[str] = []
        for file in files:
            token = occupant(Square(file, rank))
            if token is None:
                cell = "."
            else:
                letter = _TYP_LETTER[token.typ]
                cell = letter if token.color is Color.WHITE else letter.lower()
                if token in state.cooldown:
                    cell += "*"
            cells.append(cell)
        lines.append(f"{rank + 1:>2} " + " ".join(f"{c:>2}" for c in cells))
    file_labels = "   " + " ".join(f"{_FILES[f]:>2}" for f in files)
    lines.append(file_labels)
    return "\n".join(lines)
