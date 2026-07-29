from __future__ import annotations

import hashlib
from dataclasses import Field, dataclass, field, fields
from types import MappingProxyType
from typing import Any, Literal

FizzleScope = Literal["both_pawns", "any_same_square"]
AnnihilationReading = Literal["B", "timed"]
IntermezzoReading = Literal["ii", "i"]

#: Metadata marker for a field whose value *is* a rule: it enters
#: :func:`ruleset_fingerprint`, so changing it changes the fingerprint and
#: therefore invalidates every game record bound to the old one (Phase 14).
RULE_BEARING = MappingProxyType({"rule_bearing": True})

#: Metadata marker for a field that parameterizes an *implementation* rather
#: than the game (solver tolerances, cosmetic settings, ...). Excluded from
#: the fingerprint, so two rule sets differing only in such a field specify
#: the same game and their records interoperate. No v1.1 field carries this
#: marker; it exists so that adding one is a deliberate, reviewed act.
NOT_RULE_BEARING = MappingProxyType({"rule_bearing": False})

#: Domain-separation prefix of the canonical form. Bumping it re-keys every
#: fingerprint in existence; do not touch it to accommodate a field change.
FINGERPRINT_DOMAIN = "simult-chess/RuleSet/v1"


@dataclass(frozen=True, slots=True)
class RuleSet:
    """Immutable parameterization of the transition operator.

    Every field whose truth couples to an invariant is flagged ``[K]`` in
    ``INVARIANTS.md §8``. Changing a field for a variant requires editing the
    coupled invariant(s) in lockstep; the checker reads this object, never a
    literal.

    **Frozen provisional v1.1** (Gate 14; maintainer rulings C1-C3 of
    2026-07-24, ``docs/DEVELOPMENT_roadmap_v2.md`` §B). Every field is at the
    value the Phase 11b campaign (``reports/campaign_v1.md``) was run under,
    so the freeze is a no-op on values: what changes is their status, from
    ``[C]`` to ``[FROZEN v1.1]``. Each rejected A/B arm value stays playable
    as a named variant in :mod:`simult_chess.rules.variants` -- a variant is a
    `RuleSet` or a swapped stage, never a fork. The freeze is **provisional**
    (ruling C3, extending A5): it asserts a versioned default, not equilibrium
    balance, and :meth:`fingerprint` is what makes a later revision detectable
    rather than silent -- exactly what happened when ``king_capture_cooldown``
    was added post-freeze (Phase 17b, 2026-07-28): the fingerprint moved from
    ``bf2bb9dab0f0…`` to ``24932504dccb…``, a real rules change made visible
    rather than a silent drift.

    Every field must declare :data:`RULE_BEARING` or :data:`NOT_RULE_BEARING`
    metadata; a field declaring neither raises from :func:`ruleset_fingerprint`
    rather than defaulting into -- or silently out of -- the digest.

    Parameters
    ----------
    n_actions : int
        Actions per decision phase, :math:`N`. v1: ``2``. Couples: ``L1``.
    horizon : int
        No-progress draw horizon, :math:`H`. v1: ``50``. Couples: ``WF7``,
        ``T4``. Frozen at baseline (C2): the campaign's ``H`` arms moved the
        draw rate by +0.028 (``H=30``) and +0.017 (``H=80``) against a ±0.019
        minimum detectable effect, and both merely retrade horizon draws
        against mutual-king-loss draws. Variants: ``horizon_30``,
        ``horizon_80``.
    recapture_cooldown : bool
        Whether a recapturer enters :math:`C'`. v1: ``True``. Couples:
        ``R13``. Frozen at baseline (C2): the off-arm's +0.020 draw rate sits
        at the MDE. Variant: ``no_recapture_cooldown``.
    cancellation_enabled : bool
        Whether ``Cancel`` is admissible. v1: ``True`` (spec §9, resolved
        ``[C, retained]`` by the 2026-07-14 ruling A1). Frozen at baseline
        (C2), but note the off-arm is a **confirmed null by construction**:
        no agent in the campaign roster can declare a ``Cancel`` at all, so
        its null result is not evidence the rule is inert. Couples: ``R17``,
        ``L6``. Variant: ``irrevocable_defense``.
    pawn_same_square_fizzle_scope : FizzleScope
        Scope of the same-square fizzle. v1: ``"both_pawns"`` (spec §13,
        confirmed ``[C]`` by ruling A2). Frozen at that value by **ruling
        C1**, against the largest effect any arm produced: ``any_same_square``
        raised the draw rate 0.382 -> 0.521, but the mechanism is degenerate
        -- mutual-king-loss draws collapse 0.636 -> 0.155 while
        horizon-attributed draws rise 0.311 -> 0.719, i.e. games stop
        resolving and grind into the ``H`` wall. Couples: ``R2``. Variant:
        ``any_same_square_fizzle``.
    annihilation_reading : AnnihilationReading
        Mid-path collision semantics. v1: ``"B"`` (declaration-priority
        pairing). Couples: ``R4``. ``"timed"`` is declined for v1 (spec
        §13.2) and has no registered Stage A implementation, so -- unlike
        every A/B arm value -- it is deliberately *not* a named variant.
    intermezzo_reading : IntermezzoReading
        Defensive-precedence semantics. v1: ``"ii"`` (unconditional). ``"i"``
        (attacker-sequenced, spec §13.4, implemented Phase 11a) is frozen out
        of the default (C2) at +0.027 draw rate, just past the ±0.019 MDE and
        with no accompanying shift in the draw-cause breakdown. Couples:
        ``R7``, ``M4`` (flips ``M4`` from *true* to *order-dependent* under
        ``"i"``). Variant: ``attacker_sequenced_intermezzo``.
    king_capture_cooldown : bool
        Whether a king that captures (directly, or by firing a reservation as
        a recapturing defender) enters :math:`C'` like any other piece. v1:
        ``True`` (spec §7, ruling of Phase 17b, 2026-07-28: maintainer
        playtest of the Phase 16 window surfaced that an always-immune king
        could evade an attacker and then capture it back at zero risk --
        "double movement power" -- since fleeing costs nothing and, unlike
        every other piece, neither did striking back). A king's *non*-
        capturing moves (including evasion) remain always exempt -- the
        mobility guarantee is untouched; only the offensive follow-up now
        costs a cooldown turn, symmetric with how every other piece already
        pays for acting. A king that is its colour's only remaining live
        token is exempt regardless (spec §7's original "must always be able
        to move" guarantee must never leave a player with zero legal
        actions). Couples: ``WF3``, ``R13``. Variant:
        ``unconditional_king_immunity`` (the pre-17b behaviour).

    See Also
    --------
    fingerprint : stable identity of these values, bound into game records.
    simult_chess.rules.variants : the rejected arm values, still playable.
    """

    n_actions: int = field(default=2, metadata=RULE_BEARING)
    horizon: int = field(default=50, metadata=RULE_BEARING)
    recapture_cooldown: bool = field(default=True, metadata=RULE_BEARING)
    cancellation_enabled: bool = field(default=True, metadata=RULE_BEARING)
    pawn_same_square_fizzle_scope: FizzleScope = field(
        default="both_pawns", metadata=RULE_BEARING
    )
    annihilation_reading: AnnihilationReading = field(
        default="B", metadata=RULE_BEARING
    )
    intermezzo_reading: IntermezzoReading = field(default="ii", metadata=RULE_BEARING)
    king_capture_cooldown: bool = field(default=True, metadata=RULE_BEARING)

    def canonical_form(self) -> str:
        """The exact text :meth:`fingerprint` digests.

        See :func:`canonical_ruleset_form`.
        """
        return canonical_ruleset_form(self)

    def fingerprint(self) -> str:
        """Stable hex SHA-256 identity of this rule set's rule-bearing fields.

        Two rule sets fingerprint alike iff they specify the same game. The
        digest is a pure function of the ``(field name, value)`` pairs sorted
        by name, so it survives process restarts (unlike :func:`hash`, which
        is salted per-process for str keys), field reordering, and the
        addition of non-rule-bearing fields.

        Returns
        -------
        str
            64 lowercase hex characters.

        Examples
        --------
        >>> RuleSet().fingerprint()[:12]  # v1.1 defaults + the 17b king ruling
        '24932504dccb'
        """
        return ruleset_fingerprint(self)


def canonical_ruleset_form(ruleset: Any) -> str:
    """Canonical text form of a `RuleSet`-shaped dataclass, for hashing.

    The domain prefix on the first line, then one ``name=repr(value)`` line
    per rule-bearing field in ascending name order. Deliberately
    human-readable: the freeze block of ``reports/campaign_v1.md`` can be
    checked against it by eye, not only by digest.

    Parameters
    ----------
    ruleset : Any
        Any dataclass instance whose fields honour the :data:`RULE_BEARING` /
        :data:`NOT_RULE_BEARING` metadata contract -- `RuleSet` itself, or a
        future extension of it.

    Returns
    -------
    str
        Newline-terminated canonical form.

    Raises
    ------
    TypeError
        If a field declares neither marker. The fingerprint refuses to guess
        whether an undeclared field is part of the game.
    """
    lines = [FINGERPRINT_DOMAIN]
    owner = type(ruleset).__name__
    for spec in sorted(fields(ruleset), key=lambda f: f.name):
        if _is_rule_bearing(spec, owner):
            lines.append(f"{spec.name}={getattr(ruleset, spec.name)!r}")
    return "\n".join(lines) + "\n"


def ruleset_fingerprint(ruleset: Any) -> str:
    """Hex SHA-256 of :func:`canonical_ruleset_form`.

    The free-function form of :meth:`RuleSet.fingerprint`, which it
    implements; taking the instance as an argument is what lets the stability
    tests fingerprint a deliberately reordered clone of the dataclass.
    """
    return hashlib.sha256(canonical_ruleset_form(ruleset).encode("utf-8")).hexdigest()


def _is_rule_bearing(spec: Field[Any], owner: str) -> bool:
    """Read a field's fingerprint-membership marker, or refuse to guess."""
    marker = spec.metadata.get("rule_bearing")
    if marker is None:
        detail = (
            f"{owner}.{spec.name} declares neither RULE_BEARING nor "
            "NOT_RULE_BEARING metadata; the fingerprint cannot infer whether "
            "it is part of the game (Phase 14 freeze)"
        )
        raise TypeError(detail)
    return bool(marker)
