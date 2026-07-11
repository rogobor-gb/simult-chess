from __future__ import annotations

from conftest import build_state

from simult_chess.core.moves import DeclaredMove
from simult_chess.core.stages import closure
from simult_chess.core.stages.annihilate import AnnihilationEvent, AnnihilationResult
from simult_chess.core.stages.defense import DefenseResult, RecaptureFired
from simult_chess.core.types import (
    CastlingRights,
    Color,
    Reservation,
    Square,
    Token,
    Trajectory,
)
from simult_chess.rules.ruleset import RuleSet

RULESET = RuleSet()


def _dm(
    token: Token, path: tuple[Square, ...], color: Color, index: int = 1
) -> DeclaredMove:
    return DeclaredMove(
        token=token,
        trajectory=Trajectory(path=path),
        color=color,
        index=index,
        kind="move",
    )


def test_apply_promotions_replaces_only_chosen_tokens() -> None:
    pawn = Token(id=1, color=Color.WHITE, typ="p")
    knight = Token(id=2, color=Color.BLACK, typ="n")
    board = {pawn: Square(0, 7), knight: Square(1, 1)}
    final = closure.apply_promotions(board, {1: "q"})
    promoted = next(t for t in final if t.id == 1)
    assert promoted.typ == "q"
    assert final[promoted] == Square(0, 7)
    unchanged = next(t for t in final if t.id == 2)
    assert unchanged.typ == "n"


def test_compute_displaced_tokens_excludes_captured_includes_recapturer() -> None:
    mover = Token(id=1, color=Color.WHITE, typ="r")
    captured_attacker = Token(id=2, color=Color.BLACK, typ="r")
    defender = Token(id=3, color=Color.WHITE, typ="b")
    stationary = Token(id=4, color=Color.WHITE, typ="p")
    survivors = (
        _dm(mover, (Square(0, 0), Square(0, 4)), Color.WHITE),
        _dm(captured_attacker, (Square(7, 7), Square(0, 4)), Color.BLACK),
    )
    defense_result = DefenseResult(
        captured=((captured_attacker, Square(0, 4)),),
        fired=(
            RecaptureFired(
                defender=defender,
                captured=captured_attacker,
                square=Square(0, 4),
                reservation=Reservation(defender=defender, protege=mover, age=(0, 0)),
            ),
        ),
        occupancy={
            mover: Square(0, 4),
            defender: Square(0, 4),
            stationary: Square(0, 1),
        },
    )
    displaced = closure.compute_displaced_tokens(
        survivors, defense_result, defense_result.occupancy
    )
    assert mover in displaced
    assert defender in displaced
    assert captured_attacker not in displaced
    assert stationary not in displaced


def test_compute_cooldown_excludes_pawns_and_kings() -> None:
    rook = Token(id=1, color=Color.WHITE, typ="r")
    pawn = Token(id=2, color=Color.WHITE, typ="p")
    king = Token(id=3, color=Color.WHITE, typ="k")
    displaced = frozenset({rook, pawn, king})
    cooldown = closure.compute_cooldown(displaced, frozenset(), RULESET)
    assert cooldown == {rook}


def test_compute_cooldown_gates_recapturers_when_disabled() -> None:
    rook = Token(id=1, color=Color.WHITE, typ="r")
    bishop = Token(id=2, color=Color.WHITE, typ="b")
    ruleset = RuleSet(recapture_cooldown=False)
    displaced = frozenset({rook, bishop})
    cooldown = closure.compute_cooldown(displaced, frozenset({bishop.id}), ruleset)
    assert cooldown == {rook}


def test_update_castling_rights_king_move_revokes_both_sides() -> None:
    king = Token(id=1, color=Color.WHITE, typ="k")
    rook = Token(id=2, color=Color.WHITE, typ="r")
    state = build_state({king: Square(4, 0), rook: Square(7, 0)})
    rights = closure.update_castling_rights(CastlingRights(), state, frozenset({1}))
    assert not rights.white_kingside
    assert not rights.white_queenside
    assert rights.black_kingside and rights.black_queenside


def test_update_castling_rights_rook_move_revokes_only_its_own_side() -> None:
    king = Token(id=1, color=Color.WHITE, typ="k")
    rook = Token(id=2, color=Color.WHITE, typ="r")
    state = build_state({king: Square(4, 0), rook: Square(7, 0)})
    rights = closure.update_castling_rights(CastlingRights(), state, frozenset({2}))
    assert not rights.white_kingside
    assert rights.white_queenside


def test_update_castling_rights_is_monotone_non_increasing() -> None:
    king = Token(id=1, color=Color.WHITE, typ="k")
    state = build_state({king: Square(4, 0)})
    already_lost = CastlingRights(white_kingside=False)
    rights = closure.update_castling_rights(already_lost, state, frozenset())
    assert not rights.white_kingside


def test_update_no_progress_counter_resets_and_increments() -> None:
    reset_on_capture = closure.update_no_progress_counter(
        5, any_capture=True, any_pawn_displacement=False
    )
    reset_on_pawn_move = closure.update_no_progress_counter(
        5, any_capture=False, any_pawn_displacement=True
    )
    incremented = closure.update_no_progress_counter(
        5, any_capture=False, any_pawn_displacement=False
    )
    assert reset_on_capture == 0
    assert reset_on_pawn_move == 0
    assert incremented == 6


def test_update_repetition_ledger_increments_count() -> None:
    ledger: dict[object, int] = {"a": 2}
    updated = closure.update_repetition_ledger(ledger, "a")
    assert updated["a"] == 3
    assert ledger == {"a": 2}  # original untouched (pure)


def test_update_reservations_drops_cancelled_dead_and_displaced() -> None:
    d1 = Token(id=1, color=Color.WHITE, typ="r")
    q1 = Token(id=2, color=Color.WHITE, typ="p")
    d2 = Token(id=3, color=Color.WHITE, typ="b")
    q2 = Token(id=4, color=Color.WHITE, typ="p")
    d3 = Token(id=5, color=Color.WHITE, typ="n")
    q3 = Token(id=6, color=Color.WHITE, typ="p")

    cancelled_res = Reservation(defender=d1, protege=q1, age=(0, 0))
    dead_defender_res = Reservation(defender=d2, protege=q2, age=(0, 1))
    displaced_protege_res = Reservation(defender=d3, protege=q3, age=(0, 2))
    survives_res = Reservation(defender=d3, protege=q1, age=(0, 3))

    kept = closure.update_reservations(
        (cancelled_res, dead_defender_res, displaced_protege_res, survives_res),
        current_phase_index=1,
        displaced_ids=frozenset({q3.id}),
        dead_ids=frozenset({d2.id}),
        cancelled=frozenset({cancelled_res}),
        ruleset=RULESET,
    )
    assert kept == (survives_res,)


def test_update_reservations_exempts_new_reservation_from_displacement() -> None:
    # the "aggressive dual" pattern: protege (a king) moved THIS phase, and
    # the reservation was declared THIS SAME phase -- must not be dropped.
    defender = Token(id=1, color=Color.WHITE, typ="p")
    king = Token(id=2, color=Color.WHITE, typ="k")
    new_reservation = Reservation(defender=defender, protege=king, age=(3, 0))

    kept = closure.update_reservations(
        (new_reservation,),
        current_phase_index=3,
        displaced_ids=frozenset({king.id}),
        dead_ids=frozenset(),
        cancelled=frozenset(),
        ruleset=RULESET,
    )
    assert kept == (new_reservation,)


def test_update_reservations_does_not_exempt_old_reservation() -> None:
    defender = Token(id=1, color=Color.WHITE, typ="p")
    king = Token(id=2, color=Color.WHITE, typ="k")
    old_reservation = Reservation(defender=defender, protege=king, age=(0, 0))

    kept = closure.update_reservations(
        (old_reservation,),
        current_phase_index=3,
        displaced_ids=frozenset({king.id}),
        dead_ids=frozenset(),
        cancelled=frozenset(),
        ruleset=RULESET,
    )
    assert kept == ()


def test_detect_terminal_outcomes() -> None:
    white_king = Token(id=1, color=Color.WHITE, typ="k")
    black_king = Token(id=2, color=Color.BLACK, typ="k")
    both = {white_king: Square(4, 0), black_king: Square(4, 7)}
    assert closure.detect_terminal(both) == "ongoing"
    assert closure.detect_terminal({white_king: Square(4, 0)}) == "white_wins"
    assert closure.detect_terminal({black_king: Square(4, 7)}) == "black_wins"
    assert closure.detect_terminal({}) == "draw"


def test_annihilated_tokens_extracts_token_identities() -> None:
    white_token = Token(id=1, color=Color.WHITE, typ="p")
    black_token = Token(id=2, color=Color.BLACK, typ="p")
    white_move = _dm(white_token, (Square(0, 0), Square(0, 1)), Color.WHITE)
    black_move = _dm(black_token, (Square(0, 1), Square(0, 0)), Color.BLACK)
    event = AnnihilationEvent(white_move=white_move, black_move=black_move, rank=(1, 1))
    result = AnnihilationResult(events=(event,))
    tokens = closure.annihilated_tokens(result)
    assert tokens == {white_token, black_token}
