"""Stage C/D — arrivals, promotion, cooldown, reservations, bookkeeping.

Spec §6.5-6.7; INVARIANTS.md R13-R18, T1-T4. Stage A/B already produce a
correct post-capture occupancy (`DefenseResult.occupancy`); this module's
job is everything *around* that: turning survivors into promoted pieces,
computing the next cooldown set, pruning/aging reservations, updating
bookkeeping, and detecting terminal outcomes.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Literal

from simult_chess.core import geometry, legality
from simult_chess.core.moves import DeclaredMove
from simult_chess.core.stages.annihilate import AnnihilationResult
from simult_chess.core.stages.defense import DefenseResult
from simult_chess.core.types import (
    Bookkeeping,
    CastlingRights,
    Color,
    PieceType,
    Reservation,
    Square,
    State,
    Token,
)
from simult_chess.rules.ruleset import RuleSet

Outcome = Literal["ongoing", "white_wins", "black_wins", "draw"]


def apply_promotions(
    board: Mapping[Token, Square], promotion_choices: Mapping[int, PieceType]
) -> dict[Token, Square]:
    """Replace each promoted pawn with its post-promotion Token (spec §6.5).

    Parameters
    ----------
    board : Mapping[Token, Square]
        The post-Stage-B occupancy (`DefenseResult.occupancy`); already
        excludes anything captured in Stages A/B.
    promotion_choices : Mapping[int, PieceType]
        Token id -> declared promotion type, collected from every `Move`
        action that declared one (L6 already restricts this to pawns
        reaching the last rank).
    """
    final: dict[Token, Square] = {}
    for token, square in board.items():
        new_type = promotion_choices.get(token.id)
        if new_type is not None:
            final[Token(id=token.id, color=token.color, typ=new_type)] = square
        else:
            final[token] = square
    return final


def compute_displaced_tokens(
    survivors: tuple[DeclaredMove, ...],
    defense_result: DefenseResult,
    final_board: Mapping[Token, Square],
) -> frozenset[Token]:
    """Every token displaced this phase (spec §6.7), in post-promotion form."""
    captured_ids = {token.id for token in defense_result.captured_tokens}
    displaced_ids = {m.token.id for m in survivors if m.token.id not in captured_ids}
    displaced_ids |= {fired.defender.id for fired in defense_result.fired}
    return frozenset(token for token in final_board if token.id in displaced_ids)


def king_capture_candidates(
    state: State,
    survivors: tuple[DeclaredMove, ...],
    defense_result: DefenseResult,
) -> frozenset[int]:
    """King ids whose own action captured an enemy this phase.

    Ruling 17b (2026-07-28), spec §7. A king is a candidate iff *it* captured
    — directly (its declared destination held an enemy at declaration, and
    that enemy is among ``defense_result.captured_tokens``) or by firing a
    reservation as a recapturing defender (``defense_result.fired``). Fleeing
    is never a candidate: a king that simply moved to an empty square costs
    nothing, matching the original "must always be able to move" mobility
    guarantee.

    These are *candidates* only — :func:`compute_cooldown` applies the
    "never leave a colour with zero uncooled pieces" safety valve, which
    needs the full live roster and the other-piece cooldown outcome to
    evaluate, neither of which this function has visibility into.
    """
    declared_occupant = geometry.occupant_lookup(state.board)
    captured_tokens = defense_result.captured_tokens
    direct = {
        move.token.id
        for move in survivors
        if move.token.typ == "k"
        and (victim := declared_occupant(move.trajectory.destination)) is not None
        and victim in captured_tokens
    }
    fired = {
        recapture.defender.id
        for recapture in defense_result.fired
        if recapture.defender.typ == "k"
    }
    return frozenset(direct | fired)


def compute_cooldown(
    displaced_tokens: frozenset[Token],
    recapturer_ids: frozenset[int],
    king_capture_candidate_ids: frozenset[int],
    final_board: Mapping[Token, Square],
    ruleset: RuleSet,
) -> frozenset[Token]:
    """R13 — displaced tokens minus pawns (always) and kings (ordinarily).

    Pawns are always exempt. A non-pawn, non-king displaced token is cooled,
    with recapturers further gated by ``ruleset.recapture_cooldown``.

    A king in ``king_capture_candidate_ids`` (:func:`king_capture_candidates`,
    ruling 17b, 2026-07-28) is *also* cooled — unless doing so would leave its
    colour with **zero uncooled live tokens**, spec §7's "must always be able
    to move" guarantee: a lone king is the simplest instance, but the same
    rule equally covers a king whose only other piece just happens to already
    be cooled from an earlier move (verified against real self-play: a
    30-game sweep produced exactly this — a king that captured while its only
    other piece, a knight, was still cooled from a prior displacement).
    ``ruleset.king_capture_cooldown`` off makes ``king_capture_candidate_ids``
    irrelevant (the caller passes an empty set; nothing here re-checks the
    flag, since an empty candidate set already reproduces the old behaviour).
    """
    eligible = displaced_tokens
    if not ruleset.recapture_cooldown:
        eligible = frozenset(t for t in eligible if t.id not in recapturer_ids)
    non_king_cooldown = frozenset(t for t in eligible if t.typ not in ("p", "k"))

    if not king_capture_candidate_ids:
        return non_king_cooldown

    cooled_ids_by_color: dict[Color, set[int]] = {}
    for token in non_king_cooldown:
        cooled_ids_by_color.setdefault(token.color, set()).add(token.id)

    king_cooldown: set[Token] = set()
    for king in final_board:
        if king.typ != "k" or king.id not in king_capture_candidate_ids:
            continue
        live_ids = {t.id for t in final_board if t.color == king.color}
        cooled_ids = cooled_ids_by_color.get(king.color, set()) | {king.id}
        if cooled_ids != live_ids:  # someone of this colour stays uncooled
            king_cooldown.add(king)

    return non_king_cooldown | frozenset(king_cooldown)


def relax_king_cooldown_if_stranding(
    cooldown: frozenset[Token],
    final_board: Mapping[Token, Square],
    final_reservations_white: tuple[Reservation, ...],
    final_reservations_black: tuple[Reservation, ...],
    castling_rights: CastlingRights,
    ruleset: RuleSet,
) -> frozenset[Token]:
    """Undo a king's cooldown if it would strand its colour with zero legal
    actions (ruling 17b, 2026-07-28).

    :func:`compute_cooldown`'s own "zero uncooled tokens" filter is cheap but
    only *necessary*, not *sufficient*: an uncooled piece can still have no
    legal action of its own (blocked, no captures, no valid Reserve pairing)
    — a 30-game self-play sweep hit exactly this in practice. This asks the
    full, correct existence question, `legality.has_any_legal_program`,
    against the actual post-phase position, and relaxes (removes from
    cooldown) exactly the cooled kings whose colour would otherwise be
    stranded — falling back to the pre-17b behaviour for that king alone,
    which can never be a worse outcome than the engine already guaranteed
    before this ruling existed.
    """
    cooled_king_colors = {t.color for t in cooldown if t.typ == "k"}
    if not cooled_king_colors:
        return cooldown

    hypothetical_state = State(
        board=final_board,
        cooldown=cooldown,
        reservations_white=final_reservations_white,
        reservations_black=final_reservations_black,
        bookkeeping=Bookkeeping(
            castling_rights=castling_rights,
            repetition_ledger={},
            no_progress_counter=0,
            phase_index=0,
        ),
    )
    stranded_colors = {
        color
        for color in cooled_king_colors
        if not legality.has_any_legal_program(hypothetical_state, color, ruleset)
    }
    if not stranded_colors:
        return cooldown
    return frozenset(
        t for t in cooldown if not (t.typ == "k" and t.color in stranded_colors)
    )


def update_castling_rights(
    rights: CastlingRights, state: State, displaced_ids: frozenset[int]
) -> CastlingRights:
    """Revoke rights whose king/rook moved this phase (spec §6.6, WF7)."""
    occupant = geometry.occupant_lookup(state.board)

    def moved(square: Square) -> bool:
        token = occupant(square)
        return token is not None and token.id in displaced_ids

    white_king_moved = moved(Square(4, 0))
    black_king_moved = moved(Square(4, 7))
    return CastlingRights(
        white_kingside=(
            rights.white_kingside and not white_king_moved and not moved(Square(7, 0))
        ),
        white_queenside=(
            rights.white_queenside and not white_king_moved and not moved(Square(0, 0))
        ),
        black_kingside=(
            rights.black_kingside and not black_king_moved and not moved(Square(7, 7))
        ),
        black_queenside=(
            rights.black_queenside and not black_king_moved and not moved(Square(0, 7))
        ),
    )


def update_no_progress_counter(
    previous: int, *, any_capture: bool, any_pawn_displacement: bool
) -> int:
    """T4 — reset on capture/pawn displacement; else increment (spec §6.7, §10)."""
    if any_capture or any_pawn_displacement:
        return 0
    return previous + 1


def update_repetition_ledger(
    ledger: Mapping[Hashable, int], position_key: Hashable
) -> dict[Hashable, int]:
    """T3 — bump the occurrence count of the new public position (spec §10)."""
    updated = dict(ledger)
    updated[position_key] = updated.get(position_key, 0) + 1
    return updated


def update_reservations(
    reservations_in_effect: tuple[Reservation, ...],
    current_phase_index: int,
    displaced_ids: frozenset[int],
    dead_ids: frozenset[int],
    cancelled: frozenset[Reservation],
    ruleset: RuleSet,
) -> tuple[Reservation, ...]:
    """R17 — invalidate/cancel reservations at closure (spec §6.7, §9).

    A reservation is dropped iff: it was cancelled (and cancellation is
    enabled); its defender is dead or displaced this phase (a fired
    defender has displaced, so this also covers R9's self-invalidation);
    its protégé is dead; or — for a *pre-existing* reservation only — its
    protégé displaced this phase. A reservation declared *this* phase
    (`age[0] == current_phase_index`) is exempt from that last rule: the
    protégé's own move this phase is what the "aggressive dual" pattern
    (spec §4.3) defends, not a disqualifying displacement.
    """
    kept: list[Reservation] = []
    for reservation in reservations_in_effect:
        if ruleset.cancellation_enabled and reservation in cancelled:
            continue
        if (
            reservation.defender.id in dead_ids
            or reservation.defender.id in displaced_ids
        ):
            continue
        if reservation.protege.id in dead_ids:
            continue
        is_new = reservation.age[0] == current_phase_index
        if not is_new and reservation.protege.id in displaced_ids:
            continue
        kept.append(reservation)
    return tuple(kept)


def refresh_reservation_tokens(
    reservations: tuple[Reservation, ...], final_board: Mapping[Token, Square]
) -> tuple[Reservation, ...]:
    """Re-point each kept reservation's defender/protégé at their *current*
    Token snapshot (WF6).

    A `Token` is a frozen, by-value snapshot (spec §1.1): promotion mutates
    `typ` by minting a new snapshot with the same `id`. A reservation
    surviving `update_reservations` still holds the *pre-promotion*
    snapshot of any defender/protégé promoted this phase, which no longer
    value-equals the live token of the same id — silently violating WF6's
    referential integrity. Re-deriving from `final_board` by id keeps the
    reservation pointed at whichever snapshot is actually live.
    """
    by_id = {token.id: token for token in final_board}
    return tuple(
        Reservation(
            defender=by_id.get(r.defender.id, r.defender),
            protege=by_id.get(r.protege.id, r.protege),
            age=r.age,
        )
        for r in reservations
    )


def detect_terminal(board: Mapping[Token, Square]) -> Outcome:
    """T1 — king-capture terminal / synchronous draw (spec §10)."""
    has_white_king = any(t.typ == "k" and t.color is Color.WHITE for t in board)
    has_black_king = any(t.typ == "k" and t.color is Color.BLACK for t in board)
    if not has_white_king and not has_black_king:
        return "draw"
    if not has_white_king:
        return "black_wins"
    if not has_black_king:
        return "white_wins"
    return "ongoing"


def annihilated_tokens(result: AnnihilationResult) -> frozenset[Token]:
    """Token identities removed by Stage A, for no-progress/materiality bookkeeping."""
    return frozenset(move.token for move in result.annihilated)
