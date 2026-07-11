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

from simult_chess.core import geometry
from simult_chess.core.moves import DeclaredMove
from simult_chess.core.stages.annihilate import AnnihilationResult
from simult_chess.core.stages.defense import DefenseResult
from simult_chess.core.types import (
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


def compute_cooldown(
    displaced_tokens: frozenset[Token],
    recapturer_ids: frozenset[int],
    ruleset: RuleSet,
) -> frozenset[Token]:
    """R13 — displaced tokens minus pawns/kings; recapturers gated by a RuleSet flag."""
    eligible = displaced_tokens
    if not ruleset.recapture_cooldown:
        eligible = frozenset(t for t in eligible if t.id not in recapturer_ids)
    return frozenset(t for t in eligible if t.typ not in ("p", "k"))


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
