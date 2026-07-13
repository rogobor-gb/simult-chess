from __future__ import annotations

from conftest import build_state

from simult_chess.core import legality
from simult_chess.core.types import (
    Cancel,
    Castle,
    Color,
    Move,
    Reservation,
    Reserve,
    Square,
    Token,
    Trajectory,
)
from simult_chess.rules.ruleset import RuleSet

RULESET = RuleSet()


def _move(token: Token, path: tuple[Square, ...]) -> Move:
    return Move(token=token, trajectory=Trajectory(path=path))


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


def test_l1_budget_rejects_empty_and_oversized_programs() -> None:
    pawn = Token(id=1, color=Color.WHITE, typ="p")
    move = _move(pawn, (Square(4, 1), Square(4, 2)))
    assert legality.check_l1_budget((), RULESET) != []
    assert legality.check_l1_budget((move, move, move), RULESET) != []
    assert legality.check_l1_budget((move,), RULESET) == []


def test_l2_passes_when_a_move_is_declared() -> None:
    pawn = Token(id=1, color=Color.WHITE, typ="p")
    state = build_state({pawn: Square(4, 1)})
    program = (_move(pawn, (Square(4, 1), Square(4, 2))),)
    assert legality.check_l2_mandatory_displacement(state, program, Color.WHITE) == []


def test_l2_rejects_reservation_only_program_when_displacement_exists() -> None:
    defender = Token(id=1, color=Color.WHITE, typ="r")
    protege = Token(id=2, color=Color.WHITE, typ="p")
    state = build_state({defender: Square(0, 0), protege: Square(0, 3)})
    program = (Reserve(defender=defender, protege=protege),)
    violations = legality.check_l2_mandatory_displacement(state, program, Color.WHITE)
    assert len(violations) == 1


def test_l2_degenerate_exception_allows_reservation_only_program(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    defender = Token(id=1, color=Color.WHITE, typ="r")
    protege = Token(id=2, color=Color.WHITE, typ="p")
    state = build_state({defender: Square(0, 0), protege: Square(0, 3)})
    program = (Reserve(defender=defender, protege=protege),)
    monkeypatch.setattr(legality, "has_any_legal_displacement", lambda s, c: False)
    assert legality.check_l2_mandatory_displacement(state, program, Color.WHITE) == []


def test_l3_rejects_token_acting_twice() -> None:
    pawn = Token(id=1, color=Color.WHITE, typ="p")
    knight = Token(id=2, color=Color.WHITE, typ="n")
    state = build_state({pawn: Square(4, 1), knight: Square(1, 0)})
    program = (
        _move(pawn, (Square(4, 1), Square(4, 2))),
        Reserve(defender=pawn, protege=knight),
    )
    violations = legality.check_l3_distinct_actors(state, program, Color.WHITE)
    assert len(violations) == 1


def test_l3_allows_move_while_another_token_defends_it() -> None:
    mover = Token(id=1, color=Color.WHITE, typ="p")
    defender = Token(id=2, color=Color.WHITE, typ="r")
    state = build_state({mover: Square(4, 1), defender: Square(4, 0)})
    program = (
        _move(mover, (Square(4, 1), Square(4, 2))),
        Reserve(defender=defender, protege=mover),
    )
    assert legality.check_l3_distinct_actors(state, program, Color.WHITE) == []


def test_l3_rejects_castle_and_separate_move_of_the_castling_rook() -> None:
    # v1.1 ruling A3: Castle's actor set is {king, flank rook}, not king alone.
    king = Token(id=1, color=Color.WHITE, typ="k")
    kingside_rook = Token(id=2, color=Color.WHITE, typ="r")
    queenside_rook = Token(id=3, color=Color.WHITE, typ="r")
    state = build_state(
        {
            king: Square(4, 0),
            kingside_rook: Square(7, 0),
            queenside_rook: Square(0, 0),
        }
    )
    program = (
        Castle(side="king"),
        _move(kingside_rook, (Square(7, 0), Square(7, 3))),
    )
    violations = legality.check_l3_distinct_actors(state, program, Color.WHITE)
    assert len(violations) == 1


def test_l3_allows_castle_and_move_of_the_other_rook() -> None:
    king = Token(id=1, color=Color.WHITE, typ="k")
    kingside_rook = Token(id=2, color=Color.WHITE, typ="r")
    queenside_rook = Token(id=3, color=Color.WHITE, typ="r")
    state = build_state(
        {
            king: Square(4, 0),
            kingside_rook: Square(7, 0),
            queenside_rook: Square(0, 0),
        }
    )
    program = (
        Castle(side="king"),
        _move(queenside_rook, (Square(0, 0), Square(0, 3))),
    )
    assert legality.check_l3_distinct_actors(state, program, Color.WHITE) == []


def test_l3_rejects_queenside_castle_and_separate_move_of_that_rook() -> None:
    king = Token(id=1, color=Color.WHITE, typ="k")
    kingside_rook = Token(id=2, color=Color.WHITE, typ="r")
    queenside_rook = Token(id=3, color=Color.WHITE, typ="r")
    state = build_state(
        {
            king: Square(4, 0),
            kingside_rook: Square(7, 0),
            queenside_rook: Square(0, 0),
        }
    )
    program = (
        Castle(side="queen"),
        _move(queenside_rook, (Square(0, 0), Square(0, 3))),
    )
    violations = legality.check_l3_distinct_actors(state, program, Color.WHITE)
    assert len(violations) == 1


def test_l3_allows_queenside_castle_and_move_of_the_other_rook() -> None:
    king = Token(id=1, color=Color.WHITE, typ="k")
    kingside_rook = Token(id=2, color=Color.WHITE, typ="r")
    queenside_rook = Token(id=3, color=Color.WHITE, typ="r")
    state = build_state(
        {
            king: Square(4, 0),
            kingside_rook: Square(7, 0),
            queenside_rook: Square(0, 0),
        }
    )
    program = (
        Castle(side="queen"),
        _move(kingside_rook, (Square(7, 0), Square(7, 3))),
    )
    assert legality.check_l3_distinct_actors(state, program, Color.WHITE) == []


def test_l3_rejects_castle_and_separate_move_of_the_castling_rook_chi_mirror() -> None:
    # χ-mirrored image (INVARIANTS.md M3) of the kingside case: Black, rank 7.
    king = Token(id=1, color=Color.BLACK, typ="k")
    kingside_rook = Token(id=2, color=Color.BLACK, typ="r")
    queenside_rook = Token(id=3, color=Color.BLACK, typ="r")
    state = build_state(
        {
            king: Square(4, 7),
            kingside_rook: Square(7, 7),
            queenside_rook: Square(0, 7),
        }
    )
    program = (
        Castle(side="king"),
        _move(kingside_rook, (Square(7, 7), Square(7, 4))),
    )
    violations = legality.check_l3_distinct_actors(state, program, Color.BLACK)
    assert len(violations) == 1


def test_l3_allows_castle_and_move_of_the_other_rook_chi_mirror() -> None:
    king = Token(id=1, color=Color.BLACK, typ="k")
    kingside_rook = Token(id=2, color=Color.BLACK, typ="r")
    queenside_rook = Token(id=3, color=Color.BLACK, typ="r")
    state = build_state(
        {
            king: Square(4, 7),
            kingside_rook: Square(7, 7),
            queenside_rook: Square(0, 7),
        }
    )
    program = (
        Castle(side="king"),
        _move(queenside_rook, (Square(0, 7), Square(0, 4))),
    )
    assert legality.check_l3_distinct_actors(state, program, Color.BLACK) == []


def test_l4_rejects_cooled_actor() -> None:
    pawn = Token(id=1, color=Color.WHITE, typ="p")
    knight = Token(id=2, color=Color.WHITE, typ="n")
    state = build_state(
        {pawn: Square(4, 1), knight: Square(1, 0)}, cooldown=frozenset({knight})
    )
    program = (Reserve(defender=knight, protege=pawn),)
    violations = legality.check_l4_cooldown_respected(state, program, Color.WHITE)
    assert len(violations) == 1


def test_l4_allows_cooled_protege() -> None:
    pawn = Token(id=1, color=Color.WHITE, typ="p")
    defender = Token(id=2, color=Color.WHITE, typ="r")
    state = build_state(
        {pawn: Square(4, 1), defender: Square(4, 0)}, cooldown=frozenset({pawn})
    )
    program = (Reserve(defender=defender, protege=pawn),)
    assert legality.check_l4_cooldown_respected(state, program, Color.WHITE) == []


def test_l5_rejects_own_moves_that_conflict() -> None:
    rook_a = Token(id=1, color=Color.WHITE, typ="r")
    rook_b = Token(id=2, color=Color.WHITE, typ="r")
    state = build_state({rook_a: Square(0, 4), rook_b: Square(3, 0)})
    program = (
        _move(rook_a, _line(Square(0, 4), Square(7, 4))),
        _move(rook_b, _line(Square(3, 0), Square(3, 7))),
    )
    violations = legality.check_l5_own_consistency(state, program, Color.WHITE)
    assert len(violations) == 1


def test_l5_allows_non_conflicting_own_moves() -> None:
    rook = Token(id=1, color=Color.WHITE, typ="r")
    knight = Token(id=2, color=Color.WHITE, typ="n")
    state = build_state({rook: Square(0, 0), knight: Square(1, 7)})
    program = (
        _move(rook, _line(Square(0, 0), Square(0, 4))),
        _move(knight, (Square(1, 7), Square(3, 6))),
    )
    assert legality.check_l5_own_consistency(state, program, Color.WHITE) == []


def test_l6_rejects_geometrically_illegal_move() -> None:
    pawn = Token(id=1, color=Color.WHITE, typ="p")
    state = build_state({pawn: Square(4, 1)})
    bogus = _move(pawn, (Square(4, 1), Square(4, 5)))  # pawn can't leap 4 squares
    violations = legality.check_l6_geometric_legality(state, (bogus,), Color.WHITE)
    assert len(violations) == 1


def test_l6_accepts_admissible_reservation() -> None:
    defender = Token(id=1, color=Color.WHITE, typ="r")
    protege = Token(id=2, color=Color.WHITE, typ="p")
    state = build_state({defender: Square(0, 0), protege: Square(0, 3)})
    program = (Reserve(defender=defender, protege=protege),)
    assert legality.check_l6_geometric_legality(state, program, Color.WHITE) == []


def test_l6_rejects_inadmissible_reservation_pattern() -> None:
    defender = Token(id=1, color=Color.WHITE, typ="n")
    protege = Token(id=2, color=Color.WHITE, typ="p")
    # d1-a4-like offset is not a knight move
    state = build_state({defender: Square(0, 0), protege: Square(0, 3)})
    program = (Reserve(defender=defender, protege=protege),)
    violations = legality.check_l6_geometric_legality(state, program, Color.WHITE)
    assert len(violations) == 1


def test_l6_rejects_illegal_castle() -> None:
    king = Token(id=1, color=Color.WHITE, typ="k")
    state = build_state({king: Square(4, 0)})  # no rook present
    program = (Castle(side="king"),)
    violations = legality.check_l6_geometric_legality(state, program, Color.WHITE)
    assert len(violations) == 1


def test_l6_rejects_cancel_of_unknown_reservation() -> None:
    defender = Token(id=1, color=Color.WHITE, typ="r")
    protege = Token(id=2, color=Color.WHITE, typ="p")
    state = build_state({defender: Square(0, 0), protege: Square(0, 3)})
    phantom = Reservation(defender=defender, protege=protege, age=(0, 0))
    program = (Cancel(reservation=phantom),)
    violations = legality.check_l6_geometric_legality(state, program, Color.WHITE)
    assert len(violations) == 1


def test_l6_pawn_reaching_last_rank_requires_promotion() -> None:
    pawn = Token(id=1, color=Color.WHITE, typ="p")
    state = build_state({pawn: Square(0, 6)})
    move = Move(token=pawn, trajectory=Trajectory(path=(Square(0, 6), Square(0, 7))))
    violations = legality.check_l6_geometric_legality(state, (move,), Color.WHITE)
    assert len(violations) == 1


def test_l6_accepts_pawn_promotion_on_last_rank() -> None:
    pawn = Token(id=1, color=Color.WHITE, typ="p")
    state = build_state({pawn: Square(0, 6)})
    move = Move(
        token=pawn,
        trajectory=Trajectory(path=(Square(0, 6), Square(0, 7))),
        promotion="q",
    )
    assert legality.check_l6_geometric_legality(state, (move,), Color.WHITE) == []


def test_l6_rejects_promotion_declared_without_reaching_last_rank() -> None:
    pawn = Token(id=1, color=Color.WHITE, typ="p")
    state = build_state({pawn: Square(0, 1)})
    move = Move(
        token=pawn,
        trajectory=Trajectory(path=(Square(0, 1), Square(0, 2))),
        promotion="q",
    )
    violations = legality.check_l6_geometric_legality(state, (move,), Color.WHITE)
    assert len(violations) == 1


def test_l6_reservation_admissibility_uses_proteges_declared_destination() -> None:
    # The "aggressive dual" pattern (spec §4.3): the king moves to f4 and a
    # pawn on e3 defends it *there*, in the same program. Admissibility must
    # use the king's declared destination, not its pre-move square.
    king = Token(id=1, color=Color.WHITE, typ="k")
    pawn = Token(id=2, color=Color.WHITE, typ="p")
    state = build_state({king: Square(4, 3), pawn: Square(4, 2)})  # e4, e3
    program = (
        _move(king, (Square(4, 3), Square(5, 3))),  # Kf4
        Reserve(defender=pawn, protege=king),
    )
    assert legality.check_l6_geometric_legality(state, program, Color.WHITE) == []


def test_check_legal_program_accepts_a_simple_legal_program() -> None:
    pawn = Token(id=1, color=Color.WHITE, typ="p")
    state = build_state({pawn: Square(4, 1)})
    program = (_move(pawn, (Square(4, 1), Square(4, 2), Square(4, 3))),)
    assert legality.is_legal_program(state, program, Color.WHITE, RULESET)
