from __future__ import annotations

from simult_chess.core.moves import DeclaredMove
from simult_chess.core.stages.annihilate import resolve_annihilation
from simult_chess.core.types import Color, Square, Token, Trajectory
from simult_chess.rules.ruleset import RuleSet

RULESET = RuleSet()


def _dm(
    token: Token, path: tuple[Square, ...], color: Color, index: int
) -> DeclaredMove:
    return DeclaredMove(
        token=token,
        trajectory=Trajectory(path=path),
        color=color,
        index=index,
        kind="move",
    )


def _line(origin: Square, destination: Square) -> tuple[Square, ...]:
    """Build a full straight-line path (every intermediate square included)."""
    step_file = (destination.file > origin.file) - (destination.file < origin.file)
    step_rank = (destination.rank > origin.rank) - (destination.rank < origin.rank)
    squares = [origin]
    file, rank = origin.file, origin.rank
    while (file, rank) != (destination.file, destination.rank):
        file, rank = file + step_file, rank + step_rank
        squares.append(Square(file, rank))
    return tuple(squares)


def test_r3_head_on_swap_annihilates_though_disjoint_swept_sets() -> None:
    white_pawn = Token(id=1, color=Color.WHITE, typ="p")
    black_pawn = Token(id=2, color=Color.BLACK, typ="p")
    white_move = _dm(white_pawn, (Square(4, 3), Square(4, 4)), Color.WHITE, 1)
    black_move = _dm(black_pawn, (Square(4, 4), Square(4, 3)), Color.BLACK, 1)
    result = resolve_annihilation((white_move, black_move), RULESET)
    assert result.annihilated == {white_move, black_move}


def test_r6_vacated_square_theorem_rook_passes_through_unharmed() -> None:
    rook = Token(id=1, color=Color.WHITE, typ="r")
    knight = Token(id=2, color=Color.BLACK, typ="n")
    rook_move = _dm(
        rook, (Square(0, 4), Square(4, 4), Square(7, 4)), Color.WHITE, 1
    )
    knight_move = _dm(knight, (Square(4, 4), Square(2, 5)), Color.BLACK, 1)
    result = resolve_annihilation((rook_move, knight_move), RULESET)
    assert result.annihilated == frozenset()
    assert result.survives(rook_move)
    assert result.survives(knight_move)


def test_worked_case_i_rook_crossing_two_enemies_stationary_knight_survives() -> None:
    # White rook a5-h5; Black rook (B1) d8-d1 crosses at d5; Black bishop
    # (B2) c2-h7 crosses at f5; a stationary Black knight sits on h5.
    white_rook = Token(id=1, color=Color.WHITE, typ="r")
    black_rook = Token(id=2, color=Color.BLACK, typ="r")
    black_bishop = Token(id=3, color=Color.BLACK, typ="b")

    w1 = _dm(white_rook, _line(Square(0, 4), Square(7, 4)), Color.WHITE, 1)
    d_file = tuple(Square(3, r) for r in range(7, -1, -1))
    b1 = _dm(black_rook, d_file, Color.BLACK, 1)
    diagonal = (
        Square(2, 1), Square(3, 2), Square(4, 3),
        Square(5, 4), Square(6, 5), Square(7, 6),
    )
    b2 = _dm(black_bishop, diagonal, Color.BLACK, 2)

    result = resolve_annihilation((w1, b1, b2), RULESET)

    assert result.annihilated == {w1, b1}
    assert result.survives(b2)
    assert len(result.events) == 1
    assert result.events[0].rank == (1, 1)


def test_worked_case_ii_four_rook_cycle_trades_cleanly_skips_cross_edges() -> None:
    # Two White rooks slide along ranks 3 and 6; two Black rooks slide down
    # files d and e, each crossing both ranks: a complete bipartite conflict.
    w_rook1 = Token(id=1, color=Color.WHITE, typ="r")
    w_rook2 = Token(id=2, color=Color.WHITE, typ="r")
    b_rook1 = Token(id=3, color=Color.BLACK, typ="r")
    b_rook2 = Token(id=4, color=Color.BLACK, typ="r")

    w1 = _dm(w_rook1, _line(Square(0, 2), Square(7, 2)), Color.WHITE, 1)
    w2 = _dm(w_rook2, _line(Square(0, 5), Square(7, 5)), Color.WHITE, 2)
    b1 = _dm(b_rook1, tuple(Square(3, r) for r in range(7, -1, -1)), Color.BLACK, 1)
    b2 = _dm(b_rook2, tuple(Square(4, r) for r in range(7, -1, -1)), Color.BLACK, 2)

    result = resolve_annihilation((w1, w2, b1, b2), RULESET)

    assert result.annihilated == {w1, w2, b1, b2}
    fired_pairs = {(e.white_move, e.black_move) for e in result.events}
    assert fired_pairs == {(w1, b1), (w2, b2)}  # cross pairs (w1,b2)/(w2,b1) skipped


def test_equal_rank_edges_commute_under_tie_break() -> None:
    # W1 conflicts only with B2 (rank (2,1)); W2 conflicts only with B1 (rank
    # (2,1) too). These two edges are vertex-disjoint (Lemma 6.3a), so their
    # relative processing order must not change the outcome (inv M2b).
    w_rook1 = Token(id=1, color=Color.WHITE, typ="r")
    w_rook2 = Token(id=2, color=Color.WHITE, typ="r")
    b_rook1 = Token(id=3, color=Color.BLACK, typ="r")
    b_rook2 = Token(id=4, color=Color.BLACK, typ="r")

    w1 = _dm(w_rook1, (Square(0, 0), Square(3, 3)), Color.WHITE, 1)
    w2 = _dm(w_rook2, (Square(0, 1), Square(5, 5)), Color.WHITE, 2)
    b1 = _dm(b_rook1, (Square(7, 0), Square(5, 5)), Color.BLACK, 1)
    b2 = _dm(b_rook2, (Square(7, 1), Square(3, 3)), Color.BLACK, 2)

    default_result = resolve_annihilation((w1, w2, b1, b2), RULESET)
    reversed_edges = [(e.white_move, e.black_move) for e in default_result.events][::-1]
    shuffled_result = resolve_annihilation(
        (w1, w2, b1, b2), RULESET, tie_break=reversed_edges
    )

    assert default_result.annihilated == shuffled_result.annihilated == {w1, w2, b1, b2}
