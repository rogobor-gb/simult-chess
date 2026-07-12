"""Invariant severity classification (INVARIANTS.md §0.3, §2 index)."""

from __future__ import annotations

from typing import Literal

Severity = Literal["S0", "S1", "S2", "S3"]

_SEVERITY: dict[str, Severity] = {
    "WF1": "S1",
    "WF2": "S1",
    "WF3": "S1",
    "WF4": "S1",
    "WF5": "S1",
    "WF6": "S1",
    "WF7": "S1",
    "L1": "S2",
    "L2": "S2",
    "L3": "S2",
    "L4": "S2",
    "L5": "S0",
    "L6": "S2",
    "R1": "S2",
    "R2": "S2",
    "R3": "S2",
    "R4": "S2",
    "R5": "S0",
    "R6": "S0",
    "R7": "S2",
    "R8": "S2",
    "R9": "S2",
    "R10": "S2",
    "R11": "S2",
    "R12": "S0",
    "R13": "S2",
    "R14": "S2",
    "R15": "S2",
    "R16": "S2",
    "R17": "S2",
    "R18": "S0",
    "T1": "S2",
    "T2": "S2",
    "T3": "S2",
    "T4": "S2",
    "M1": "S0",
    "M2": "S0",
    "M3": "S0",
    "M4": "S0",
    "M5": "S1",
}


def severity_of(invariant_id: str) -> Severity:
    """Look up an invariant's severity (INVARIANTS.md §0.3); unknown IDs -> S2."""
    return _SEVERITY.get(invariant_id, "S2")
