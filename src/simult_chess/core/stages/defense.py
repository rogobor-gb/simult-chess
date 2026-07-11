"""Stage B — defense-precedence capture and recapture cascade.

Spec §6.4; INVARIANTS.md R7-R12. The intermezzo: when a stationary victim
holding a valid reservation is captured, its oldest valid reservation fires
— the defender recaptures, categorically pre-empting any same-phase capture
of that defender, regardless of the attacker's declaration order. A fired
defender may itself be defended, so recaptures chain on a single contested
square ("battery") until one side exhausts valid reservations there.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from simult_chess.core import geometry
from simult_chess.core.moves import DeclaredMove
from simult_chess.core.types import Reservation, Square, State, Token
from simult_chess.rules.ruleset import RuleSet


@dataclass(frozen=True, slots=True)
class RecaptureFired:
    """One fired recapture in the cascade (inv §9 trace: "fired: [...]")."""

    defender: Token
    captured: Token
    square: Square
    reservation: Reservation


@dataclass(frozen=True, slots=True)
class DefenseResult:
    """The output of Stage B: who was captured, which reservations fired."""

    captured: tuple[tuple[Token, Square], ...]
    fired: tuple[RecaptureFired, ...]
    occupancy: dict[Token, Square]

    @property
    def captured_tokens(self) -> frozenset[Token]:
        """Every token removed from the game during Stage B."""
        return frozenset(token for token, _square in self.captured)

    def survives(self, token: Token) -> bool:
        """Whether `token` was not captured in Stage B."""
        return token not in self.captured_tokens


def _build_reservation_indices(
    reservations_white: tuple[Reservation, ...],
    reservations_black: tuple[Reservation, ...],
) -> tuple[dict[Token, list[Reservation]], dict[Token, list[Reservation]]]:
    by_protege: dict[Token, list[Reservation]] = {}
    by_defender: dict[Token, list[Reservation]] = {}
    for reservation in (*reservations_white, *reservations_black):
        by_protege.setdefault(reservation.protege, []).append(reservation)
        by_defender.setdefault(reservation.defender, []).append(reservation)
    for reservations in by_protege.values():
        reservations.sort(key=lambda r: r.age)
    return by_protege, by_defender


def _mutual_cycle_blacklist(
    pending: dict[Square, tuple[Token, Token]],
    victim_square: dict[Token, Square],
    by_defender: dict[Token, list[Reservation]],
) -> frozenset[int]:
    """R11 — detect mutual-defense 2-cycles (P defends Q, Q defends P, both attacked).

    The spec's own tie-break is base semantics on the cycle: neither
    reservation fires. This is the *only* cycle shape the precedence
    relation can exhibit (Lemma 6.4a's proof), so a direct pairwise scan
    suffices — no general graph search is needed.
    """
    blacklisted: set[int] = set()
    for _square, (_attacker, victim) in pending.items():
        for r1 in by_defender.get(victim, []):
            other_square = victim_square.get(r1.protege)
            if other_square is None:
                continue
            for r2 in by_defender.get(r1.protege, []):
                if r2.protege is victim:
                    blacklisted.add(id(r1))
                    blacklisted.add(id(r2))
    return frozenset(blacklisted)


def resolve_defense(
    executing: tuple[DeclaredMove, ...],
    survivors: tuple[DeclaredMove, ...],
    state: State,
    reservations_white: tuple[Reservation, ...],
    reservations_black: tuple[Reservation, ...],
    ruleset: RuleSet,
    *,
    tie_break: Sequence[Square] | None = None,
) -> DefenseResult:
    """Resolve Stage B: pending captures, then the precedence-DAG cascade (spec §6.4).

    Parameters
    ----------
    executing : tuple[DeclaredMove, ...]
        :math:`M^\\ast`, Stage F's output (every move that wasn't fizzled),
        needed to tell a genuinely *stationary* token apart from one that
        vacated its origin and was then annihilated in Stage A — per spec
        §3, a token vacates "whether or not it later dies en route," so an
        annihilated mover's origin is never a valid capture target (the
        vacated-square theorem, R6, generalized past pawns).
    survivors : tuple[DeclaredMove, ...]
        The moves surviving Stage A (a subset of `executing`).
    state : State
        The declaration-time state (board, reservations already in effect,
        cooldown).
    reservations_white, reservations_black : tuple[Reservation, ...]
        The reservations *in effect this phase* — the caller's
        responsibility to union `state.reservations_*` with any freshly
        declared `Reserve` actions (a reservation declared this phase can
        fire this same phase, spec §4.3's "aggressive dual" example).
    ruleset : RuleSet
        Unused by the v1 "ii" (unconditional) reading; present for the
        stage-strategy signature and the declined attacker-sequenced
        variant (spec §13.4).
    tie_break : Sequence[Square] | None
        An explicit outer processing order over contested squares. The
        cross-battery dependency ordering is enforced internally regardless
        (a battery whose victim might depart to defend elsewhere is always
        resolved after that other battery), so the result is invariant to
        this order (inv M2c) — mirroring the attacker's own declaration
        order invariance (inv M4).
    """
    moved_tokens = {move.token for move in survivors}
    vacated_ids = {move.token.id for move in executing}

    occupancy: dict[Square, Token] = {
        square: token
        for token, square in state.board.items()
        if token.id not in vacated_ids
    }
    for move in survivors:
        occupancy[move.trajectory.destination] = move.token

    declared_occupant = geometry.occupant_lookup(state.board)
    by_protege, by_defender = _build_reservation_indices(
        reservations_white, reservations_black
    )

    pending: dict[Square, tuple[Token, Token]] = {}
    for move in survivors:
        destination = move.trajectory.destination
        victim = declared_occupant(destination)
        if (
            victim is not None
            and victim.color != move.color
            and victim.id not in vacated_ids
        ):
            pending[destination] = (move.token, victim)

    victim_square: dict[Token, Square] = {
        victim: square for square, (_attacker, victim) in pending.items()
    }

    blacklisted = _mutual_cycle_blacklist(pending, victim_square, by_defender)

    survivor_ids = {token.id for token in moved_tokens}
    annihilated_ids = vacated_ids - survivor_ids
    alive: set[Token] = {
        token for token in state.board if token.id not in annihilated_ids
    }
    fired_defenders: set[Token] = set()
    captured_log: list[tuple[Token, Square]] = []
    fired_log: list[RecaptureFired] = []
    status: dict[Square, str] = {}

    def victim_present(victim: Token) -> bool:
        """Whether the original victim of a battery is still at its square.

        A victim escapes capture iff it fires as a defender for one of its
        *own* reservations elsewhere first — resolving that other battery
        now (if not already resolved) is the precedence-DAG coupling
        (Lemma 6.4a) that makes the d4/e3 worked example order-independent.
        """
        for reservation in by_defender.get(victim, []):
            if id(reservation) in blacklisted:
                continue
            other_square = victim_square.get(reservation.protege)
            if (
                other_square is not None
                and other_square in pending
                and status.get(other_square) != "resolved"
            ):
                resolve_square(other_square)
        return victim in alive and victim not in fired_defenders

    def defender_available(defender: Token) -> bool:
        """Whether a candidate defender can still fire.

        Deliberately no cross-battery lookahead: if `defender` also happens
        to be under direct attack elsewhere, whichever battery is processed
        first wins it — the same kind of order-dependent tie-break as R9's
        shared-defender case, which the spec does not resolve further.
        """
        return defender in alive and defender not in fired_defenders

    def find_valid_defender(
        victim: Token, square: Square
    ) -> tuple[Reservation, Square] | None:
        for reservation in by_protege.get(victim, []):
            if id(reservation) in blacklisted:
                continue
            defender = reservation.defender
            if defender in fired_defenders or defender in moved_tokens:
                continue
            if defender in state.cooldown:
                continue
            if not defender_available(defender):
                continue
            defender_square = state.board[defender]
            pattern = geometry.capturing_pattern_trajectory_at(
                defender.typ, defender.color, defender_square, square, occupancy.get
            )
            if pattern is None:
                continue
            return reservation, defender_square
        return None

    def resolve_square(square: Square) -> None:
        if square not in pending:
            return  # tolerate tie_break entries that aren't contested squares
        if status.get(square) == "resolved":
            return
        status[square] = "resolving"

        holder, victim = pending[square]
        if not victim_present(victim):
            status[square] = "resolved"
            return

        alive.discard(victim)
        captured_log.append((victim, square))
        current_victim = victim

        while True:
            found = find_valid_defender(current_victim, square)
            if found is None:
                break
            reservation, defender_square = found
            defender = reservation.defender

            # Only clear the defender's origin if no *separate* battery is
            # contesting it: a contested square's occupancy entry belongs to
            # its own attacker's provisional arrival, and popping it here
            # would erase that attacker rather than the defender (who never
            # held that entry to begin with once overwritten).
            if defender_square not in pending:
                occupancy.pop(defender_square, None)
            occupancy[square] = defender
            fired_defenders.add(defender)
            alive.discard(holder)
            captured_log.append((holder, square))
            fired_log.append(
                RecaptureFired(
                    defender=defender,
                    captured=holder,
                    square=square,
                    reservation=reservation,
                )
            )
            current_victim = holder
            holder = defender

        status[square] = "resolved"

    order = list(tie_break) if tie_break is not None else list(pending.keys())
    for square in order:
        resolve_square(square)
    for square in pending:
        resolve_square(square)

    final_occupancy = {token: square for square, token in occupancy.items()}
    return DefenseResult(
        captured=tuple(captured_log), fired=tuple(fired_log), occupancy=final_occupancy
    )
