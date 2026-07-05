"""Core state algebra: the formal objects of spec §1-4.

Types here are deliberately permissive at construction time — e.g. ``State``
does not itself reject two tokens sharing a square. Well-formedness is a
separate, checkable predicate (``invariants/checks.py``, WF1-7), not a type
invariant, so that malformed states can be constructed in tests and caught by
the checker they are meant to exercise.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Literal

PieceType = Literal["p", "n", "b", "r", "q", "k"]
CastleSide = Literal["king", "queen"]


class Color(Enum):
    """A player color, spec §1.1 :math:`\\Omega=\\{W,B\\}`."""

    WHITE = "W"
    BLACK = "B"

    @property
    def opponent(self) -> Color:
        """Return :math:`-\\omega`, the opposing color."""
        return Color.BLACK if self is Color.WHITE else Color.WHITE


@dataclass(frozen=True, slots=True)
class Square:
    """A board square :math:`q=(c,r)`, spec §1.1.

    Parameters
    ----------
    file : int
        Column index, :math:`c \\in \\{0,\\dots,7\\}` (``a``-``h``).
    rank : int
        Row index, :math:`r \\in \\{0,\\dots,7\\}` (``1``-``8``).
    """

    file: int
    rank: int

    def __post_init__(self) -> None:
        if not (0 <= self.file <= 7 and 0 <= self.rank <= 7):
            raise ValueError(f"square out of range: file={self.file}, rank={self.rank}")

    def __repr__(self) -> str:
        return f"{'abcdefgh'[self.file]}{self.rank + 1}"


@dataclass(frozen=True, slots=True)
class Token:
    """An identity-carrying piece token, spec §1.1.

    ``id`` is the persistent identity referenced across states (reservations
    and captures track identity, not (square, type) pairs). ``color`` is
    fixed for the token's lifetime; ``typ`` is mutable only via promotion, so
    a promoted token is represented in a later ``State`` by a ``Token`` with
    the same ``id`` and a different ``typ`` — not by in-place mutation.

    Parameters
    ----------
    id : int
        Persistent token identity, unique across the game.
    color : Color
        Fixed owner color, :math:`\\mathrm{col}(p)`.
    typ : PieceType
        Current piece type, :math:`\\mathrm{typ}(p)`.
    """

    id: int
    color: Color
    typ: PieceType


@dataclass(frozen=True, slots=True)
class Trajectory:
    """A lattice path :math:`\\tau=(q_0,\\dots,q_\\ell)`, spec §1.3.

    Parameters
    ----------
    path : tuple[Square, ...]
        The trajectory including its origin, :math:`\\ell \\ge 1`.
    is_jump : bool
        ``True`` for a knight move: the token does not traverse interior
        squares or edges, so :math:`\\varepsilon(\\tau)=\\varnothing` even
        though :math:`\\sigma(\\tau)=\\{q_1\\}` (spec §1.3).
    """

    path: tuple[Square, ...]
    is_jump: bool = False

    def __post_init__(self) -> None:
        if len(self.path) < 2:
            raise ValueError("trajectory must contain an origin and at least one step")
        if self.is_jump and len(self.path) != 2:
            raise ValueError("a jump trajectory must be exactly (origin, destination)")

    @property
    def origin(self) -> Square:
        """:math:`q_0`, the trajectory's starting square."""
        return self.path[0]

    @property
    def destination(self) -> Square:
        """:math:`q_\\ell`, the trajectory's final square."""
        return self.path[-1]

    @property
    def swept(self) -> frozenset[Square]:
        """:math:`\\sigma(\\tau)=\\{q_1,\\dots,q_\\ell\\}`, origin excluded."""
        return frozenset(self.path[1:])

    @property
    def edges(self) -> frozenset[tuple[Square, Square]]:
        """:math:`\\varepsilon(\\tau)`, empty for a jump (spec §1.3)."""
        if self.is_jump:
            return frozenset()
        return frozenset(zip(self.path, self.path[1:], strict=False))


@dataclass(frozen=True, slots=True)
class Reservation:
    """A reservation :math:`\\rho=(D,Q,a)`, spec §4.3.

    Parameters
    ----------
    defender : Token
        The defender token :math:`D`.
    protege : Token
        The protégé token :math:`Q`.
    age : tuple[int, int]
        Age stamp :math:`a=(\\text{phase},\\text{intra-program index})`,
        totally ordering all reservations (spec §4.3, inv WF5).
    """

    defender: Token
    protege: Token
    age: tuple[int, int]


@dataclass(frozen=True, slots=True)
class Move:
    """Action :math:`\\mathrm{Move}(p,\\tau)`, spec §4.1."""

    token: Token
    trajectory: Trajectory


@dataclass(frozen=True, slots=True)
class Reserve:
    """Action :math:`\\mathrm{Reserve}(D,Q)`, spec §4.1."""

    defender: Token
    protege: Token


@dataclass(frozen=True, slots=True)
class Castle:
    """Action :math:`\\mathrm{Castle}(\\text{side})`, spec §4.1, §6.6."""

    side: CastleSide


@dataclass(frozen=True, slots=True)
class Cancel:
    """Action :math:`\\mathrm{Cancel}(\\rho)`, spec §4.1, §9."""

    reservation: Reservation


Action = Move | Reserve | Castle | Cancel
Program = tuple[Action, ...]


@dataclass(frozen=True, slots=True)
class CastlingRights:
    """Per-flank castling rights, part of bookkeeping :math:`\\eta` (spec §1.2).

    Rights are monotone non-increasing over a game (inv WF7) — never regained
    once lost.
    """

    white_kingside: bool = True
    white_queenside: bool = True
    black_kingside: bool = True
    black_queenside: bool = True


@dataclass(frozen=True, slots=True)
class Bookkeeping:
    """Bookkeeping :math:`\\eta=(\\text{rights},\\text{ledger},\\nu,t)`, spec §1.2.

    Parameters
    ----------
    castling_rights : CastlingRights
        Per-flank, per-color castling rights.
    repetition_ledger : Mapping[Hashable, int]
        Occurrence counts keyed on the public-position key
        :math:`K(\\beta,C)` (``referee/serialize.py``), excluding reservations.
    no_progress_counter : int
        :math:`\\nu`, phases since the last capture or pawn displacement.
    phase_index : int
        :math:`t`, the current phase number.
    """

    castling_rights: CastlingRights
    repetition_ledger: Mapping[Hashable, int]
    no_progress_counter: int
    phase_index: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repetition_ledger", MappingProxyType(dict(self.repetition_ledger))
        )


@dataclass(frozen=True, slots=True)
class State:
    """A game state :math:`s=(\\beta,\\,C,\\,R_W,\\,R_B,\\,\\eta)`, spec §1.2.

    Parameters
    ----------
    board : Mapping[Token, Square]
        Occupancy :math:`\\beta`, an injective partial map from live tokens to
        squares. Injectivity is a checkable invariant (WF1), not enforced by
        construction.
    cooldown : frozenset[Token]
        Cooldown set :math:`C \\subseteq \\mathcal P^{\\text{live}}`.
    reservations_white : tuple[Reservation, ...]
        :math:`R_W`, age-ordered.
    reservations_black : tuple[Reservation, ...]
        :math:`R_B`, age-ordered.
    bookkeeping : Bookkeeping
        :math:`\\eta`.
    """

    board: Mapping[Token, Square]
    cooldown: frozenset[Token]
    reservations_white: tuple[Reservation, ...]
    reservations_black: tuple[Reservation, ...]
    bookkeeping: Bookkeeping

    def __post_init__(self) -> None:
        object.__setattr__(self, "board", MappingProxyType(dict(self.board)))
        object.__setattr__(self, "cooldown", frozenset(self.cooldown))
        object.__setattr__(
            self, "reservations_white", tuple(self.reservations_white)
        )
        object.__setattr__(
            self, "reservations_black", tuple(self.reservations_black)
        )

    def reservations(self, color: Color) -> tuple[Reservation, ...]:
        """Return :math:`R_\\omega` for the given color."""
        if color is Color.WHITE:
            return self.reservations_white
        return self.reservations_black
