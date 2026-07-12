"""Observation channels / the commit-reveal interface (spec §11.5, dev brief §3.5).

Every mode with two deciders (local hot-seat and online) obeys one loop:
collect both programs, then reveal, then resolve — neither side's program
is visible to the other before both commit. The perfect-information base
game (this module) "reveals everything": `reveal` is immediate. Building
the interface now, even though it does nothing clever yet, is what makes
Phase 8's networking a thin *transport* for the same handshake rather than
a redesign — and what will let a hidden-information variant (spec Ch. 11)
substitute genuine concealment (salted hash commitment, selective reveal)
without touching `referee/match.py`'s phase loop.
"""

from __future__ import annotations

from dataclasses import dataclass

from simult_chess.core.types import Color, Program


@dataclass(frozen=True, slots=True)
class Commitment:
    """A committed program. Opaque in a hidden-information channel; here, not."""

    color: Color
    program: Program


class ObservationChannel:
    """Perfect-information channel: `reveal` returns the program immediately."""

    def commit(self, color: Color, program: Program) -> Commitment:
        """Register `program` as `color`'s committed action for this phase."""
        return Commitment(color=color, program=program)

    def reveal(self, commitment: Commitment) -> Program:
        """Reveal a commitment's program (base game: always available)."""
        return commitment.program
