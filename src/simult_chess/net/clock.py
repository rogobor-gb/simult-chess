"""Concurrent-bank time control with a race bonus (Phase 15b).

A `TimeControl` is **session metadata**, not part of `RuleSet` or `State`: Φ
stays a pure function of :math:`(s, \\pi_\\mathrm W, \\pi_\\mathrm B)` with no
wall-clock (inv M1), and the clock ledger is carried alongside the state and
hashed separately (`net.session`).

Unlike alternating chess, both players are on the clock for the *same* phase
(clocks run concurrently). Per phase :math:`k` each player :math:`\\omega` has a
remaining bank :math:`B_\\omega^k` and spends a thinking time
:math:`t_\\omega^k \\ge 0`:

.. math::
    B_\\omega^{k+1} = B_\\omega^{k} - t_\\omega^{k} + \\iota + \\beta_\\omega^{k},

with :math:`\\iota` a Fischer increment paid to both and :math:`\\beta_\\omega^k`
the **race bonus** — a resource reward for committing first. A player whose
bank would reach :math:`\\le 0` on committing loses on time; because L1/L2 make
a null program illegal, forfeit-on-time needs no default program — the game
simply ends (this is why the control is preferred over a byoyomi).

The race-bonus rule is a **registry** (maintainer ruling C4, 2026-07-25), so
the rule is a parameter, not a rewrite: the default is capped-difference; also
registered are winner-take-all (documented-degenerate at the proposed
calibration), a dead-zone variant, and none.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal

BonusRule = Literal["capped_difference", "winner_take_all", "dead_zone", "none"]

#: Registered bonus-rule names, for CLI validation and error messages.
BONUS_RULES: Final = ("capped_difference", "winner_take_all", "dead_zone", "none")


@dataclass(frozen=True, slots=True)
class TimeControl:
    """A concurrent-bank control, written ``bank|ι|b`` (e.g. ``3|0|2``).

    Standard chess ``3|2`` already means "3 min + 2 s Fischer increment", so the
    notation here is deliberately three-field — ``minutes|increment|bonus`` — to
    avoid the collision (roadmap §15b).

    Parameters
    ----------
    initial_bank : float
        Per-player starting bank :math:`B_0`, in **seconds**.
    increment : float
        Fischer increment :math:`\\iota` paid to both each phase, in seconds.
    bonus : float
        Race-bonus size :math:`b`, in seconds.
    bonus_rule : BonusRule
        Which registered rule turns a speed gap into a bonus.
    dead_zone : float
        For ``dead_zone``: award no bonus when the two thinking times are within
        this many seconds of each other (ruling C4's flagged refinement — the
        bonus then rewards only a *decisive* speed gap).
    """

    initial_bank: float
    increment: float = 0.0
    bonus: float = 0.0
    bonus_rule: BonusRule = "capped_difference"
    dead_zone: float = 0.0

    @staticmethod
    def parse(spec: str) -> TimeControl:
        """Parse ``bank|ι|b`` (minutes|seconds|seconds), optionally ``…/rule``.

        Examples
        --------
        >>> TimeControl.parse("3|0|2").initial_bank
        180.0
        >>> TimeControl.parse("3|2|2/dead_zone").bonus_rule
        'dead_zone'
        """
        body, _, rule = spec.partition("/")
        parts = body.split("|")
        if len(parts) != 3:
            raise ValueError(
                f"time control must be 'minutes|increment|bonus', got {spec!r}"
            )
        try:
            minutes, increment, bonus = (float(p) for p in parts)
        except ValueError as exc:
            raise ValueError(f"non-numeric field in time control {spec!r}") from exc
        bonus_rule: BonusRule = "capped_difference"
        if rule:
            if rule not in BONUS_RULES:
                raise ValueError(
                    f"unknown bonus rule {rule!r}; choose from {BONUS_RULES}"
                )
            bonus_rule = rule  # type: ignore[assignment]
        return TimeControl(
            initial_bank=minutes * 60.0,
            increment=increment,
            bonus=bonus,
            bonus_rule=bonus_rule,
        )

    def format(self) -> str:
        """The ``bank|ι|b`` short form (bank rounded to whole minutes)."""
        text = f"{self.initial_bank / 60.0:g}|{self.increment:g}|{self.bonus:g}"
        if self.bonus_rule != "capped_difference":
            text += f"/{self.bonus_rule}"
        return text


# ---------------------------------------------------------------------------
# Race-bonus registry (ruling C4: the rule is a parameter)
# ---------------------------------------------------------------------------


def _capped_difference(t_self: float, t_other: float, tc: TimeControl) -> float:
    """Refund only what you saved versus the opponent, capped at ``b`` (default).

    Continuous at ties and self-defeating if both race (the gap is ~0), so it
    keeps the first-decider flavour without the winner-take-all preemption
    degeneracy (roadmap §15b analysis).
    """
    if t_self < t_other:
        return min(tc.bonus, t_other - t_self)
    return 0.0


def _winner_take_all(t_self: float, t_other: float, tc: TimeControl) -> float:
    """The full bonus to whoever is strictly faster. Documented-degenerate."""
    return tc.bonus if t_self < t_other else 0.0


def _dead_zone(t_self: float, t_other: float, tc: TimeControl) -> float:
    """Capped-difference, but nothing until the gap exceeds ``dead_zone`` (C4)."""
    if abs(t_self - t_other) < tc.dead_zone:
        return 0.0
    return _capped_difference(t_self, t_other, tc)


def _no_bonus(t_self: float, t_other: float, tc: TimeControl) -> float:
    return 0.0


_BONUS_RULES: dict[BonusRule, Callable[[float, float, TimeControl], float]] = {
    "capped_difference": _capped_difference,
    "winner_take_all": _winner_take_all,
    "dead_zone": _dead_zone,
    "none": _no_bonus,
}


def race_bonus(t_self: float, t_other: float, tc: TimeControl) -> float:
    """The bonus awarded to a player who thought ``t_self`` (opponent ``t_other``)."""
    return _BONUS_RULES[tc.bonus_rule](t_self, t_other, tc)


# ---------------------------------------------------------------------------
# Banks, ledger, and the per-phase update
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Banks:
    """Both players' remaining banks, in seconds."""

    white: float
    black: float


@dataclass(frozen=True, slots=True)
class ClockEntry:
    """One phase's clock ledger row — the object the divergence check hashes."""

    phase_index: int
    think_white: float
    think_black: float
    bonus_white: float
    bonus_black: float
    bank_white_after: float
    bank_black_after: float


@dataclass(frozen=True, slots=True)
class FlagFall:
    """A time forfeit: which side(s) ran their bank to ``<= 0`` this phase."""

    white_flagged: bool
    black_flagged: bool


def initial_banks(tc: TimeControl) -> Banks:
    """Both banks at :math:`B_0`."""
    return Banks(white=tc.initial_bank, black=tc.initial_bank)


def apply_phase(
    banks: Banks,
    tc: TimeControl,
    *,
    think_white: float,
    think_black: float,
    phase_index: int,
) -> tuple[Banks, ClockEntry, FlagFall | None]:
    """Advance both banks by one phase; detect a flag-fall.

    A flag-fall (a bank reaching ``<= 0`` on committing) ends the match, so the
    updated banks are still returned for the ledger but the third element names
    who forfeited. Bonus and increment are applied only on a surviving phase —
    a forfeit ends the game before either is earned. Pure and symmetric, so both
    peers derive identical banks from identical revealed times (ruling C5).
    """
    white_flagged = banks.white - think_white <= 0.0
    black_flagged = banks.black - think_black <= 0.0
    if white_flagged or black_flagged:
        after = Banks(
            white=banks.white - think_white,
            black=banks.black - think_black,
        )
        entry = ClockEntry(
            phase_index=phase_index,
            think_white=think_white,
            think_black=think_black,
            bonus_white=0.0,
            bonus_black=0.0,
            bank_white_after=after.white,
            bank_black_after=after.black,
        )
        return after, entry, FlagFall(white_flagged, black_flagged)

    bonus_white = race_bonus(think_white, think_black, tc)
    bonus_black = race_bonus(think_black, think_white, tc)
    after = Banks(
        white=banks.white - think_white + tc.increment + bonus_white,
        black=banks.black - think_black + tc.increment + bonus_black,
    )
    entry = ClockEntry(
        phase_index=phase_index,
        think_white=think_white,
        think_black=think_black,
        bonus_white=bonus_white,
        bonus_black=bonus_black,
        bank_white_after=after.white,
        bank_black_after=after.black,
    )
    return after, entry, None


def _quantize(seconds: float) -> int:
    """Round a time to whole microseconds for a stable cross-machine hash."""
    return round(seconds * 1_000_000)


def format_entry(entry: ClockEntry) -> str:
    """A compact one-field string for the `.scn` record's clock column."""
    return (
        f"tW={entry.think_white:g},tB={entry.think_black:g},"
        f"bW={entry.bonus_white:g},bB={entry.bonus_black:g},"
        f"BW={entry.bank_white_after:g},BB={entry.bank_black_after:g}"
    )


def entry_hash(entry: ClockEntry) -> str:
    """Hex SHA-256 over a clock entry, for the per-phase divergence check.

    Times are quantized to microseconds so two peers that measured the same
    revealed values hash identically despite float formatting.
    """
    payload = "|".join(
        str(x)
        for x in (
            entry.phase_index,
            _quantize(entry.think_white),
            _quantize(entry.think_black),
            _quantize(entry.bonus_white),
            _quantize(entry.bonus_black),
            _quantize(entry.bank_white_after),
            _quantize(entry.bank_black_after),
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()
