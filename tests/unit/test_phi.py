from __future__ import annotations

import pytest
from conftest import build_state

from simult_chess.core.phi import phi
from simult_chess.core.types import (
    Castle,
    Color,
    Move,
    PieceType,
    Reservation,
    Reserve,
    Square,
    Token,
    Trajectory,
)
from simult_chess.invariants.checks import check_all_state
from simult_chess.rules.ruleset import RuleSet

RULESET = RuleSet()

WHITE_KING = Token(id=100, color=Color.WHITE, typ="k")
BLACK_KING = Token(id=200, color=Color.BLACK, typ="k")


def _move(
    token: Token, path: tuple[Square, ...], promotion: PieceType | None = None
) -> Move:
    return Move(token=token, trajectory=Trajectory(path=path), promotion=promotion)


def test_basic_pawn_pushes_no_capture() -> None:
    white_pawn = Token(id=1, color=Color.WHITE, typ="p")
    black_pawn = Token(id=2, color=Color.BLACK, typ="p")
    state = build_state(
        {
            WHITE_KING: Square(4, 0),
            BLACK_KING: Square(4, 7),
            white_pawn: Square(4, 1),
            black_pawn: Square(4, 6),
        }
    )
    program_white = (_move(white_pawn, (Square(4, 1), Square(4, 2), Square(4, 3))),)
    program_black = (_move(black_pawn, (Square(4, 6), Square(4, 5), Square(4, 4))),)

    result = phi(state, program_white, program_black, RULESET)

    assert result.outcome == "ongoing"
    assert result.state.board[white_pawn] == Square(4, 3)
    assert result.state.board[black_pawn] == Square(4, 4)
    assert result.state.bookkeeping.no_progress_counter == 0
    assert result.state.bookkeeping.phase_index == 1
    assert white_pawn not in result.state.cooldown  # pawns exempt


def test_undefended_capture_removes_victim_and_cools_capturer() -> None:
    rook = Token(id=1, color=Color.WHITE, typ="r")
    knight = Token(id=2, color=Color.BLACK, typ="n")
    filler_pawn = Token(id=3, color=Color.BLACK, typ="p")
    state = build_state(
        {
            WHITE_KING: Square(4, 0),
            BLACK_KING: Square(4, 7),
            rook: Square(0, 0),
            knight: Square(0, 4),
            filler_pawn: Square(7, 6),
        }
    )
    rook_path = (Square(0, 0), Square(0, 1), Square(0, 2), Square(0, 3), Square(0, 4))
    program_white = (_move(rook, rook_path),)
    program_black = (_move(filler_pawn, (Square(7, 6), Square(7, 5))),)

    result = phi(state, program_white, program_black, RULESET)

    assert knight.id not in {t.id for t in result.state.board}
    assert result.state.board[rook] == Square(0, 4)
    assert rook in result.state.cooldown
    assert result.state.bookkeeping.no_progress_counter == 0


def test_worked_example_d4_e3_defense_end_to_end() -> None:
    d4_pawn = Token(id=1, color=Color.WHITE, typ="p")
    e3_pawn = Token(id=2, color=Color.WHITE, typ="p")
    rook1 = Token(id=3, color=Color.BLACK, typ="r")
    rook2 = Token(id=4, color=Color.BLACK, typ="r")
    filler_pawn = Token(id=5, color=Color.WHITE, typ="p")
    state = build_state(
        {
            WHITE_KING: Square(4, 0),
            BLACK_KING: Square(4, 7),
            d4_pawn: Square(3, 3),
            e3_pawn: Square(4, 2),
            rook1: Square(3, 7),
            rook2: Square(4, 6),
            filler_pawn: Square(0, 1),
        },
        reservations_white=(
            Reservation(defender=e3_pawn, protege=d4_pawn, age=(0, 0)),
        ),
    )
    program_white = (_move(filler_pawn, (Square(0, 1), Square(0, 2))),)
    rook1_path = (Square(3, 7), Square(3, 6), Square(3, 5), Square(3, 4), Square(3, 3))
    rook2_path = (Square(4, 6), Square(4, 5), Square(4, 4), Square(4, 3), Square(4, 2))
    program_black = (_move(rook1, rook1_path), _move(rook2, rook2_path))

    result = phi(state, program_white, program_black, RULESET)

    board_by_id = {t.id: t for t in result.state.board}
    assert d4_pawn.id not in board_by_id
    assert rook1.id not in board_by_id
    assert result.state.board[board_by_id[e3_pawn.id]] == Square(3, 3)
    assert result.state.board[board_by_id[rook2.id]] == Square(4, 2)
    assert result.state.reservations_white == ()  # fired -> defender displaced


def test_promotion_produces_new_type_and_enters_cooldown() -> None:
    pawn = Token(id=1, color=Color.WHITE, typ="p")
    filler = Token(id=2, color=Color.BLACK, typ="p")
    state = build_state(
        {
            WHITE_KING: Square(4, 0),
            BLACK_KING: Square(4, 7),
            pawn: Square(0, 6),
            filler: Square(7, 6),
        }
    )
    program_white = (_move(pawn, (Square(0, 6), Square(0, 7)), promotion="q"),)
    program_black = (_move(filler, (Square(7, 6), Square(7, 5))),)

    result = phi(state, program_white, program_black, RULESET)

    promoted = next(t for t in result.state.board if t.id == pawn.id)
    assert promoted.typ == "q"
    assert result.state.board[promoted] == Square(0, 7)
    assert promoted in result.state.cooldown


def test_castling_moves_both_pieces_and_revokes_rights() -> None:
    rook = Token(id=1, color=Color.WHITE, typ="r")
    filler = Token(id=2, color=Color.BLACK, typ="p")
    state = build_state(
        {
            WHITE_KING: Square(4, 0),
            BLACK_KING: Square(4, 7),
            rook: Square(7, 0),
            filler: Square(0, 6),
        }
    )
    program_white = (Castle(side="king"),)
    program_black = (_move(filler, (Square(0, 6), Square(0, 5))),)

    result = phi(state, program_white, program_black, RULESET)

    king_token = next(t for t in result.state.board if t.id == WHITE_KING.id)
    rook_token = next(t for t in result.state.board if t.id == rook.id)
    assert result.state.board[king_token] == Square(6, 0)
    assert result.state.board[rook_token] == Square(5, 0)
    assert king_token not in result.state.cooldown
    assert rook_token in result.state.cooldown
    assert not result.state.bookkeeping.castling_rights.white_kingside
    assert not result.state.bookkeeping.castling_rights.white_queenside


def test_king_capture_ends_the_game() -> None:
    rook = Token(id=1, color=Color.WHITE, typ="r")
    filler = Token(id=2, color=Color.BLACK, typ="p")
    state = build_state(
        {
            WHITE_KING: Square(4, 0),
            rook: Square(0, 7),
            BLACK_KING: Square(4, 7),
            filler: Square(0, 6),
        }
    )
    rook_path = (Square(0, 7), Square(1, 7), Square(2, 7), Square(3, 7), Square(4, 7))
    program_white = (_move(rook, rook_path),)
    program_black = (_move(filler, (Square(0, 6), Square(0, 5))),)

    result = phi(state, program_white, program_black, RULESET)

    assert result.outcome == "white_wins"  # white's rook captured black's king


def test_synchronous_double_king_capture_is_a_draw() -> None:
    white_rook = Token(id=1, color=Color.WHITE, typ="r")
    black_rook = Token(id=2, color=Color.BLACK, typ="r")
    state = build_state(
        {
            WHITE_KING: Square(4, 0),
            BLACK_KING: Square(4, 7),
            white_rook: Square(0, 7),
            black_rook: Square(0, 0),
        }
    )
    white_path = (Square(0, 7), Square(1, 7), Square(2, 7), Square(3, 7), Square(4, 7))
    black_path = (Square(0, 0), Square(1, 0), Square(2, 0), Square(3, 0), Square(4, 0))
    program_white = (_move(white_rook, white_path),)
    program_black = (_move(black_rook, black_path),)

    result = phi(state, program_white, program_black, RULESET)

    assert result.outcome == "draw"


def test_no_progress_horizon_triggers_draw() -> None:
    small_horizon = RuleSet(horizon=2)
    state = build_state({WHITE_KING: Square(4, 0), BLACK_KING: Square(4, 7)})

    step1 = phi(
        state,
        (_move(WHITE_KING, (Square(4, 0), Square(3, 0))),),
        (_move(BLACK_KING, (Square(4, 7), Square(3, 7))),),
        small_horizon,
    )
    assert step1.state.bookkeeping.no_progress_counter == 1
    assert step1.outcome == "ongoing"

    step2 = phi(
        step1.state,
        (_move(WHITE_KING, (Square(3, 0), Square(4, 0))),),
        (_move(BLACK_KING, (Square(3, 7), Square(4, 7))),),
        small_horizon,
    )
    assert step2.state.bookkeeping.no_progress_counter == 2
    assert step2.outcome == "draw"


def test_threefold_repetition_triggers_draw() -> None:
    state = build_state({WHITE_KING: Square(4, 0), BLACK_KING: Square(4, 7)})

    forward_white = (_move(WHITE_KING, (Square(4, 0), Square(3, 0))),)
    forward_black = (_move(BLACK_KING, (Square(4, 7), Square(3, 7))),)
    backward_white = (_move(WHITE_KING, (Square(3, 0), Square(4, 0))),)
    backward_black = (_move(BLACK_KING, (Square(3, 7), Square(4, 7))),)

    outcomes = []
    forward = True
    for _ in range(5):
        white_program = forward_white if forward else backward_white
        black_program = forward_black if forward else backward_black
        result = phi(state, white_program, black_program, RULESET)
        outcomes.append(result.outcome)
        state = result.state
        forward = not forward

    assert outcomes[:4] == ["ongoing", "ongoing", "ongoing", "ongoing"]
    assert outcomes[4] == "draw"


def test_illegal_program_raises_value_error() -> None:
    state = build_state({WHITE_KING: Square(4, 0), BLACK_KING: Square(4, 7)})
    # white has a legal king move available, so an empty program violates L2
    with pytest.raises(ValueError):
        phi(state, (), (_move(BLACK_KING, (Square(4, 7), Square(3, 7))),), RULESET)


def test_aggressive_dual_reservation_persists_and_fires_next_phase() -> None:
    # Phase 1: the rook moves to f4 and, in the same program, a pawn on e3
    # declares a reservation defending it *there* (spec §4.3's "aggressive
    # dual"). Phase 2: a black rook attacks f4; the pawn must still recapture.
    king = Token(id=1, color=Color.WHITE, typ="k")
    rook = Token(id=2, color=Color.WHITE, typ="r")
    pawn = Token(id=3, color=Color.WHITE, typ="p")
    black_filler = Token(id=4, color=Color.BLACK, typ="p")
    state = build_state(
        {
            king: Square(0, 0),
            rook: Square(4, 3),  # e4
            pawn: Square(4, 2),  # e3
            BLACK_KING: Square(4, 7),
            black_filler: Square(0, 6),
        }
    )
    program_white = (
        _move(rook, (Square(4, 3), Square(5, 3))),  # Re4-f4
        Reserve(defender=pawn, protege=rook),
    )
    program_black = (_move(black_filler, (Square(0, 6), Square(0, 5))),)

    phase1 = phi(state, program_white, program_black, RULESET)

    assert len(phase1.state.reservations_white) == 1
    reservation = phase1.state.reservations_white[0]
    assert reservation.protege == rook
    assert reservation.defender == pawn
    rook_after_phase1 = next(t for t in phase1.state.board if t.id == rook.id)
    assert phase1.state.board[rook_after_phase1] == Square(5, 3)

    black_rook = Token(id=5, color=Color.BLACK, typ="r")
    state2 = build_state(
        dict(phase1.state.board) | {black_rook: Square(5, 7)},
        reservations_white=phase1.state.reservations_white,
    )
    king_survivor = next(t for t in state2.board if t.id == king.id)
    program_white2 = (
        _move(king_survivor, (Square(0, 0), Square(0, 1))),
    )
    black_rook_path = (
        Square(5, 7), Square(5, 6), Square(5, 5), Square(5, 4), Square(5, 3),
    )
    program_black2 = (_move(black_rook, black_rook_path),)

    phase2 = phi(state2, program_white2, program_black2, RULESET)

    board_ids2 = {t.id: t for t in phase2.state.board}
    assert rook.id not in board_ids2  # rook captured
    assert black_rook.id not in board_ids2  # recaptured by the pawn
    assert phase2.state.board[board_ids2[pawn.id]] == Square(5, 3)
    assert phase2.state.reservations_white == ()  # defender fired -> displaced


def test_reservation_survives_promotion_with_a_refreshed_token_snapshot() -> None:
    # A pawn promotes and, in the same program, a king declares a reservation
    # defending it there (aggressive dual + promotion combined). The surviving
    # reservation must reference the *promoted* token, not a stale pre-promotion
    # snapshot -- otherwise WF6 referential integrity silently breaks (found by
    # the Phase 6 self-play sweep).
    black_pawn = Token(id=1, color=Color.BLACK, typ="p")
    black_king = Token(id=2, color=Color.BLACK, typ="k")
    white_king = Token(id=3, color=Color.WHITE, typ="k")
    state = build_state(
        {
            black_pawn: Square(0, 1),  # a2
            black_king: Square(1, 1),  # b2, adjacent to a1
            white_king: Square(7, 7),
        }
    )
    program_black = (
        _move(black_pawn, (Square(0, 1), Square(0, 0)), promotion="q"),
        Reserve(defender=black_king, protege=black_pawn),
    )
    program_white = (_move(white_king, (Square(7, 7), Square(7, 6))),)

    result = phi(state, program_white, program_black, RULESET)

    assert check_all_state(result.state, RULESET) == []
    assert len(result.state.reservations_black) == 1
    reservation = result.state.reservations_black[0]
    assert reservation.protege.id == black_pawn.id
    assert reservation.protege.typ == "q"
    assert reservation.protege in result.state.board


def test_promotion_does_not_apply_to_a_fizzled_capture() -> None:
    # A pawn's promoting diagonal capture can fizzle (F1: the target itself
    # executes a move this phase) -- the pawn then stays on its origin,
    # unpromoted, uncooled. Found by the Phase 6 self-play sweep: promotion
    # was being applied even when the declared move never executed.
    pawn = Token(id=1, color=Color.WHITE, typ="p")
    knight = Token(id=2, color=Color.BLACK, typ="n")
    state = build_state(
        {
            WHITE_KING: Square(4, 0),
            BLACK_KING: Square(4, 7),
            pawn: Square(0, 6),  # a7
            knight: Square(1, 7),  # b8
        }
    )
    program_white = (_move(pawn, (Square(0, 6), Square(1, 7)), promotion="q"),)
    knight_move = Move(
        token=knight,
        trajectory=Trajectory(path=(Square(1, 7), Square(3, 6)), is_jump=True),
    )
    program_black = (knight_move,)  # vacates b8

    result = phi(state, program_white, program_black, RULESET)

    board_by_id = {t.id: t for t in result.state.board}
    pawn_after = board_by_id[pawn.id]
    assert pawn_after.typ == "p"  # not promoted
    assert result.state.board[pawn_after] == Square(0, 6)  # never left a7
    assert pawn_after not in result.state.cooldown
