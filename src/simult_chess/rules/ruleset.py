from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FizzleScope = Literal["both_pawns", "any_same_square"]
AnnihilationReading = Literal["B", "timed"]
IntermezzoReading = Literal["ii", "i"]


@dataclass(frozen=True, slots=True)
class RuleSet:
    """Immutable parameterization of the transition operator.

    Every field whose truth couples to an invariant is flagged ``[K]`` in
    ``INVARIANTS.md §8``. Changing a field for a variant requires editing the
    coupled invariant(s) in lockstep; the checker reads this object, never a
    literal.

    Parameters
    ----------
    n_actions : int
        Actions per decision phase, :math:`N`. v1: ``2``. Couples: ``L1``.
    horizon : int
        No-progress draw horizon, :math:`H`. v1: ``50``. Couples: ``WF7``, ``T4``.
    recapture_cooldown : bool
        Whether a recapturer enters :math:`C'`. v1: ``True``. Couples: ``R13``.
    cancellation_enabled : bool
        Whether ``Cancel`` is admissible. v1: ``True`` (spec §9, resolved
        ``[C, retained]`` by the 2026-07-14 ruling A1 — value unchanged from
        v1.0's default, only its epistemic status moved from ``[OPEN]``).
        Couples: ``R17``, ``L6``.
    pawn_same_square_fizzle_scope : FizzleScope
        Scope of the same-square fizzle. v1: ``"both_pawns"`` (spec §13,
        confirmed ``[C]`` by the 2026-07-14 ruling A2 — value unchanged, only
        its epistemic status moved from ``[C, confirm]``). Couples: ``R2``.
    annihilation_reading : AnnihilationReading
        Mid-path collision semantics. v1: ``"B"`` (declaration-priority pairing).
        Couples: ``R4``.
    intermezzo_reading : IntermezzoReading
        Defensive-precedence semantics. v1 default: ``"ii"`` (unconditional).
        ``"i"`` (attacker-sequenced, spec §13.4, implemented Phase 11a) is a
        registered alternative, to be A/B-tested empirically in Phase 11b.
        Couples: ``R7``, ``M4`` (flips ``M4`` from *true* to *order-dependent*
        under ``"i"``).
    """

    n_actions: int = 2
    horizon: int = 50
    recapture_cooldown: bool = True
    cancellation_enabled: bool = True
    pawn_same_square_fizzle_scope: FizzleScope = "both_pawns"
    annihilation_reading: AnnihilationReading = "B"
    intermezzo_reading: IntermezzoReading = "ii"
