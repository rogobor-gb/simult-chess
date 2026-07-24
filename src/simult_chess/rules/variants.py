"""Named `RuleSet` variants (Phase 14, Gate 11b freeze).

The Phase 14 freeze (`docs/DEVELOPMENT_roadmap_v2.md`, maintainer rulings
C1-C3) fixes every rule-bearing field of :class:`~simult_chess.rules.ruleset.RuleSet`
at its Phase 11b campaign baseline. This module is the other half of that
decision: **every rejected A/B arm value stays playable**, by name, without
forking. A variant is a `RuleSet` (or a swapped stage implementation, see
`rules/registry.py`) -- never a copy of the engine.

Registration policy: a value belongs here iff it is a rule-bearing level that
the campaign actually ran and the freeze declined. Levers with no registered
stage implementation (``annihilation_reading="timed"``, spec §13.2, declined
for v1) are deliberately absent -- naming them would promise a game that does
not resolve.

Stdlib only, like the rest of `rules/` (dev brief §0.6).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from simult_chess.rules.ruleset import RuleSet

#: The frozen provisional v1.1 defaults (Gate 14). Identical to ``RuleSet()``;
#: named so that a caller can say which rules it means rather than implying it.
FROZEN_V1_1 = RuleSet()

BASELINE_NAME = "frozen_v1_1"


@dataclass(frozen=True, slots=True)
class NamedVariant:
    """A rule set reachable by name, with the evidence that put it here.

    Parameters
    ----------
    name : str
        Registry key; the value accepted by ``--variant`` on the CLIs.
    ruleset : RuleSet
        The rules themselves. Fingerprint via ``ruleset.fingerprint()``.
    summary : str
        What the variant changes, in rules terms.
    evidence : str
        Why the freeze declined it (or, for the baseline, adopted it), traced
        to `reports/campaign_v1.md`.
    """

    name: str
    ruleset: RuleSet
    summary: str
    evidence: str


_VARIANTS: tuple[NamedVariant, ...] = (
    NamedVariant(
        name=BASELINE_NAME,
        ruleset=FROZEN_V1_1,
        summary="The frozen provisional v1.1 defaults; what an unqualified "
        "RuleSet() means.",
        evidence="Control arm: draw rate 0.382 [0.362, 0.402] over 2222 "
        "matrix_1ply self-play games.",
    ),
    NamedVariant(
        name="irrevocable_defense",
        ruleset=replace(FROZEN_V1_1, cancellation_enabled=False),
        summary="Cancel is inadmissible (L6 rejects it); a reservation, once "
        "declared, cannot be withdrawn (spec §9's named alternative).",
        evidence="Draw rate +0.009 vs control, inside the ±0.019 MDE -- but a "
        "confirmed null *by construction*: no campaign agent can declare a "
        "Cancel, so the arm never exercised the rule. Declined on C2 with "
        "that caveat, not on evidence of inertness.",
    ),
    NamedVariant(
        name="attacker_sequenced_intermezzo",
        ruleset=replace(FROZEN_V1_1, intermezzo_reading="i"),
        summary="Reading (i): captures resolve in the attacker's declared "
        "order, so striking a defender before its protege defuses the "
        "reservation (spec §13.4). Deliberately order-dependent -- inv M4 "
        "branches.",
        evidence="Draw rate +0.027 vs control, just past the ±0.019 MDE, with "
        "no accompanying shift in the draw-cause breakdown. Declined (C2).",
    ),
    NamedVariant(
        name="any_same_square_fizzle",
        ruleset=replace(FROZEN_V1_1, pawn_same_square_fizzle_scope="any_same_square"),
        summary="Same-square convergence fizzles for *any* two movers, not "
        "only two pawns; mixed pawn/non-pawn convergence stops being "
        "(V)-annihilation.",
        evidence="The largest effect any arm produced: draw rate 0.382 -> "
        "0.521 (+0.140). Declined by ruling C1 on mechanism -- "
        "mutual-king-loss draws collapse 0.636 -> 0.155 while "
        "horizon-attributed draws rise 0.311 -> 0.719, i.e. it does not "
        "balance the game, it stops it resolving.",
    ),
    NamedVariant(
        name="no_recapture_cooldown",
        ruleset=replace(FROZEN_V1_1, recapture_cooldown=False),
        summary="A recapturing token does not enter C', so it may act again "
        "next phase (spec §7's 'faster play' candidate).",
        evidence="Draw rate +0.020 vs control, at the ±0.019 MDE. Declined "
        "(C2).",
    ),
    NamedVariant(
        name="horizon_30",
        ruleset=replace(FROZEN_V1_1, horizon=30),
        summary="No-progress draw horizon H = 30 instead of 50: material is "
        "less convertible, games are cut shorter.",
        evidence="Draw rate +0.028 vs control (±0.027 MDE at n=2500), but the "
        "composition merely retrades mutual-king-loss draws (0.636 -> 0.457) "
        "for horizon draws (0.311 -> 0.533). Declined (C2).",
    ),
    NamedVariant(
        name="horizon_80",
        ruleset=replace(FROZEN_V1_1, horizon=80),
        summary="No-progress draw horizon H = 80 instead of 50: more room to "
        "convert an advantage before the draw fires.",
        evidence="Draw rate +0.017 vs control, inside the ±0.027 MDE; the "
        "retrade runs the other way (horizon draws 0.311 -> 0.177). Declined "
        "(C2).",
    ),
)

VARIANTS: dict[str, NamedVariant] = {variant.name: variant for variant in _VARIANTS}


def variant_names() -> tuple[str, ...]:
    """Every registered variant name, registration order (baseline first)."""
    return tuple(variant.name for variant in _VARIANTS)


def get_variant(name: str) -> RuleSet:
    """The `RuleSet` registered under `name`.

    Parameters
    ----------
    name : str
        A key of :data:`VARIANTS`; see :func:`variant_names`.

    Returns
    -------
    RuleSet
        The variant's rules. Its ``fingerprint()`` differs from the frozen
        baseline's for every non-baseline variant, which is what makes a game
        record state unambiguously which game was played.

    Raises
    ------
    KeyError
        If `name` is unregistered, listing what is available.
    """
    try:
        return VARIANTS[name].ruleset
    except KeyError:
        detail = (
            f"unknown variant {name!r}; registered: {', '.join(variant_names())}"
        )
        raise KeyError(detail) from None


def describe_variants() -> str:
    """A one-line-per-variant listing, for ``--variant list`` on the CLIs."""
    return "\n".join(
        f"{variant.name}  [{variant.ruleset.fingerprint()[:12]}]  {variant.summary}"
        for variant in _VARIANTS
    )
