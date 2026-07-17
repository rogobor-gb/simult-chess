"""Stage B, Reading (i) — attacker-sequenced intermezzo (spec §13.4).

The "leaner", order-*dependent* alternative to Reading (ii)'s categorical
defensive precedence (`defense.py`): pending captures are grouped into
rounds by the attacking mover's own declared index and resolved in
increasing round order, so a reservation's validity is checked against the
board as left by every strictly-earlier round. Attacking the *defender*
before its protégé strips the reservation before it can fire; attacking the
protégé first reproduces Reading (ii)'s outcome. Same-round events from
opposite colors (a genuine index tie between the two players) are resolved
via Reading (ii)'s own defender-lookahead, restricted to just that round, so
the tie itself stays color-symmetric.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from simult_chess.core import geometry
from simult_chess.core.moves import DeclaredMove
from simult_chess.core.stages.defense import (
    DefenseResult,
    RecaptureFired,
    build_reservation_indices,
    mutual_cycle_blacklist,
)
from simult_chess.core.types import Reservation, Square, State, Token
from simult_chess.rules.ruleset import RuleSet


@dataclass(frozen=True, slots=True)
class _Battery:
    holder: Token
    victim: Token
    round_index: int


def resolve_defense_seq(
    executing: tuple[DeclaredMove, ...],
    survivors: tuple[DeclaredMove, ...],
    state: State,
    reservations_white: tuple[Reservation, ...],
    reservations_black: tuple[Reservation, ...],
    ruleset: RuleSet,
    *,
    tie_break: Sequence[Square] | None = None,
) -> DefenseResult:
    """Resolve Stage B under Reading (i) (spec §13.4).

    Same parameters as `defense.resolve_defense` (the `DefenseResolver`
    Protocol, `rules/registry.py`). `tie_break`, if given, only reorders
    squares *within* a round (a same-index cross-color tie) — cross-round
    order is fixed by the attackers' own declared indices, so it is never
    up for grabs (mirroring how `defense.py`'s own `tie_break` only ever
    reorders within what the precedence DAG already leaves free).
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
    by_protege, by_defender = build_reservation_indices(
        reservations_white, reservations_black
    )

    pending: dict[Square, _Battery] = {}
    for move in survivors:
        destination = move.trajectory.destination
        victim = declared_occupant(destination)
        if (
            victim is not None
            and victim.color != move.color
            and victim.id not in vacated_ids
        ):
            pending[destination] = _Battery(
                holder=move.token, victim=victim, round_index=move.index
            )

    victim_square: dict[Token, Square] = {
        battery.victim: square for square, battery in pending.items()
    }
    blacklisted = mutual_cycle_blacklist(
        {sq: (b.holder, b.victim) for sq, b in pending.items()},
        victim_square,
        by_defender,
    )

    survivor_ids = {token.id for token in moved_tokens}
    annihilated_ids = vacated_ids - survivor_ids
    alive: set[Token] = {
        token for token in state.board if token.id not in annihilated_ids
    }
    fired_defenders: set[Token] = set()
    captured_log: list[tuple[Token, Square]] = []
    fired_log: list[RecaptureFired] = []
    status: dict[Square, str] = {}

    def defender_available(defender: Token) -> bool:
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

    def victim_present_in_round(
        victim: Token, round_squares: frozenset[Square]
    ) -> bool:
        """Reading (ii)'s own cross-battery lookahead (spec §6.4), but only
        ever consulted for a same-round (same declared-index) tie, and only
        ever looking at *other squares in that same round* -- a victim can
        never escape via an earlier or later round under Reading (i)."""
        for reservation in by_defender.get(victim, []):
            if id(reservation) in blacklisted:
                continue
            other_square = victim_square.get(reservation.protege)
            if (
                other_square is not None
                and other_square in round_squares
                and status.get(other_square) != "resolved"
            ):
                resolve_square(other_square, round_squares)
        return victim in alive and victim not in fired_defenders

    def resolve_square(square: Square, round_squares: frozenset[Square]) -> None:
        if status.get(square) == "resolved":
            return
        status[square] = "resolving"

        battery = pending[square]
        holder, victim = battery.holder, battery.victim
        if len(round_squares) > 1 and not victim_present_in_round(
            victim, round_squares
        ):
            status[square] = "resolved"
            return
        # A victim already gone -- dead, or having fired as a defender in a
        # *strictly earlier* round and thus relocated -- is simply absent;
        # the attacker arrives at a now-empty square, no capture (the same
        # vacated-square principle as R6, spec §6.3 Remark). Only earlier
        # rounds can have resolved by this point, so no lookahead is needed
        # outside `victim_present_in_round`'s own (same-round-only) case.
        if victim not in alive or victim in fired_defenders:
            status[square] = "resolved"
            return

        alive.discard(victim)
        captured_log.append((victim, square))

        while True:
            found = find_valid_defender(victim, square)
            if found is None:
                break
            reservation, defender_square = found
            defender = reservation.defender

            defender_resolved = status.get(defender_square) == "resolved"
            if defender_square not in pending or defender_resolved:
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
            victim = holder
            holder = defender

        status[square] = "resolved"

    # Group into rounds by the attacker's own declared index (spec §13.4);
    # process rounds in increasing order. A `tie_break` only reorders the
    # squares *within* a round.
    squares_by_round: dict[int, list[Square]] = {}
    for square, battery in pending.items():
        squares_by_round.setdefault(battery.round_index, []).append(square)

    if tie_break is not None:
        priority = {square: position for position, square in enumerate(tie_break)}
        for squares in squares_by_round.values():
            squares.sort(key=lambda sq: priority.get(sq, len(priority)))

    for round_index in sorted(squares_by_round):
        round_squares = frozenset(squares_by_round[round_index])
        for square in squares_by_round[round_index]:
            resolve_square(square, round_squares)

    final_occupancy = {token: square for square, token in occupancy.items()}
    return DefenseResult(
        captured=tuple(captured_log), fired=tuple(fired_log), occupancy=final_occupancy
    )
