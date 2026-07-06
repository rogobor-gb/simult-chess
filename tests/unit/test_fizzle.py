from __future__ import annotations

from conftest import build_state

from simult_chess.core.moves import DeclaredMove
from simult_chess.core.stages.fizzle import resolve_fizzles
from simult_chess.core.types import Color, Square, Token, Trajectory
from simult_chess.rules.ruleset import RuleSet

BOTH_PAWNS = RuleSet()
ANY_SQUARE = RuleSet(pawn_same_square_fizzle_scope="any_same_square")


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


def test_f2_opposing_pawn_pushes_onto_shared_square_both_fizzle() -> None:
    white_pawn = Token(id=1, color=Color.WHITE, typ="p")
    black_pawn = Token(id=2, color=Color.BLACK, typ="p")
    state = build_state({white_pawn: Square(3, 2), black_pawn: Square(3, 4)})
    white_move = _dm(white_pawn, (Square(3, 2), Square(3, 3)), Color.WHITE)
    black_move = _dm(black_pawn, (Square(3, 4), Square(3, 3)), Color.BLACK)
    result = resolve_fizzles((white_move, black_move), state, BOTH_PAWNS)
    assert result.fizzled == {white_move, black_move}
    assert not result.executes(white_move)
    assert not result.executes(black_move)


def test_f2_mixed_convergence_not_fizzled_under_both_pawns_scope() -> None:
    white_pawn = Token(id=1, color=Color.WHITE, typ="p")
    black_knight = Token(id=2, color=Color.BLACK, typ="n")
    state = build_state({white_pawn: Square(3, 2), black_knight: Square(1, 4)})
    white_move = _dm(white_pawn, (Square(3, 2), Square(3, 3)), Color.WHITE)
    black_move = _dm(black_knight, (Square(1, 4), Square(3, 3)), Color.BLACK)
    result = resolve_fizzles((white_move, black_move), state, BOTH_PAWNS)
    assert result.fizzled == frozenset()


def test_f2_mixed_convergence_fizzles_under_any_same_square_scope() -> None:
    white_pawn = Token(id=1, color=Color.WHITE, typ="p")
    black_knight = Token(id=2, color=Color.BLACK, typ="n")
    state = build_state({white_pawn: Square(3, 2), black_knight: Square(1, 4)})
    white_move = _dm(white_pawn, (Square(3, 2), Square(3, 3)), Color.WHITE)
    black_move = _dm(black_knight, (Square(1, 4), Square(3, 3)), Color.BLACK)
    result = resolve_fizzles((white_move, black_move), state, ANY_SQUARE)
    assert result.fizzled == {white_move, black_move}


def test_f1_capture_fizzles_when_target_executes() -> None:
    attacker = Token(id=1, color=Color.WHITE, typ="p")
    target = Token(id=2, color=Color.BLACK, typ="n")
    state = build_state({attacker: Square(0, 3), target: Square(1, 4)})
    attack_move = _dm(attacker, (Square(0, 3), Square(1, 4)), Color.WHITE)
    target_move = _dm(target, (Square(1, 4), Square(3, 5)), Color.BLACK)
    result = resolve_fizzles((attack_move, target_move), state, BOTH_PAWNS)
    assert result.fizzled == {attack_move}
    assert result.executes(target_move)


def test_f1_capture_stands_when_target_is_stationary() -> None:
    attacker = Token(id=1, color=Color.WHITE, typ="p")
    target = Token(id=2, color=Color.BLACK, typ="n")
    state = build_state({attacker: Square(0, 3), target: Square(1, 4)})
    attack_move = _dm(attacker, (Square(0, 3), Square(1, 4)), Color.WHITE)
    result = resolve_fizzles((attack_move,), state, BOTH_PAWNS)
    assert result.fizzled == frozenset()


def test_f1_capture_stands_when_targets_own_move_was_f2_fizzled() -> None:
    attacker = Token(id=1, color=Color.WHITE, typ="p")
    target = Token(id=2, color=Color.BLACK, typ="p")
    third = Token(id=3, color=Color.WHITE, typ="p")
    state = build_state(
        {attacker: Square(3, 3), target: Square(4, 4), third: Square(4, 2)}
    )
    attack_move = _dm(attacker, (Square(3, 3), Square(4, 4)), Color.WHITE)
    target_push = _dm(target, (Square(4, 4), Square(4, 3)), Color.BLACK)
    third_push = _dm(third, (Square(4, 2), Square(4, 3)), Color.WHITE)
    result = resolve_fizzles((attack_move, target_push, third_push), state, BOTH_PAWNS)
    # target_push and third_push converge on (4,3): F2 fizzles both.
    assert target_push in result.fizzled
    assert third_push in result.fizzled
    # target's own move fizzled -> target is stationary -> attacker's capture stands.
    assert attack_move not in result.fizzled


def test_f1_mutual_diagonal_capture_is_a_2cycle_neither_fizzles() -> None:
    white_pawn = Token(id=1, color=Color.WHITE, typ="p")
    black_pawn = Token(id=2, color=Color.BLACK, typ="p")
    state = build_state({white_pawn: Square(4, 3), black_pawn: Square(5, 4)})
    white_move = _dm(white_pawn, (Square(4, 3), Square(5, 4)), Color.WHITE)
    black_move = _dm(black_pawn, (Square(5, 4), Square(4, 3)), Color.BLACK)
    result = resolve_fizzles((white_move, black_move), state, BOTH_PAWNS)
    # Lemma 6.2: this 2-cycle is an (E) edge-conflict, exported to Stage A.
    assert result.fizzled == frozenset()


def test_f1_chain_result_is_independent_of_tie_break_order() -> None:
    attacker = Token(id=1, color=Color.WHITE, typ="p")
    middle = Token(id=2, color=Color.BLACK, typ="p")
    end = Token(id=3, color=Color.WHITE, typ="n")
    state = build_state(
        {attacker: Square(0, 1), middle: Square(1, 2), end: Square(2, 3)}
    )
    attack_move = _dm(attacker, (Square(0, 1), Square(1, 2)), Color.WHITE)
    middle_move = _dm(middle, (Square(1, 2), Square(2, 3)), Color.BLACK)

    result_default = resolve_fizzles((attack_move, middle_move), state, BOTH_PAWNS)
    result_shuffled = resolve_fizzles(
        (attack_move, middle_move),
        state,
        BOTH_PAWNS,
        tie_break=(middle_move, attack_move),
    )
    assert result_default.fizzled == result_shuffled.fizzled == {attack_move}
    assert middle_move not in result_default.fizzled
