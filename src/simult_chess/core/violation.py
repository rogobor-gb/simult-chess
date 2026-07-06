"""Shared diagnostic record for legality/invariant checks (WF-*, L-*, R-*, T-*)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Violation:
    """A single failed check.

    Parameters
    ----------
    invariant_id : str
        The `INVARIANTS.md` ID, e.g. ``"WF1"`` or ``"L5"``.
    detail : str
        Human-readable description of the offending witness.
    """

    invariant_id: str
    detail: str
