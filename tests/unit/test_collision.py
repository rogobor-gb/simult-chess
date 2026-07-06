from __future__ import annotations

from conftest import build_state

from simult_chess.core.collision import (
    annihilation_rank,
    conflicts,
    edge_conflict,
    mirror_program,
    mirror_state,
    vertex_conflict,
)
from simult_chess.core.types import (
    CastlingRights,
    Color,
    Move,
    Reservation,
    Reserve,
    Square,
    State,
    Token,
    Trajectory,
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


def test_vertex_conflict_on_perpendicular_crossing() -> None:
    # rook a5-h5 vs rook d1-d8: paths cross at d5
    horizontal = Trajectory(path=_line(Square(0, 4), Square(7, 4)))
    vertical = Trajectory(path=_line(Square(3, 0), Square(3, 7)))
    assert vertex_conflict(horizontal, vertical)
    assert conflicts(horizontal, vertical)


def test_edge_conflict_head_on_swap_not_a_vertex_conflict() -> None:
    # e4-e5 vs e5-e4: swept sets are disjoint (each ends on the other's origin)
    white_push = Trajectory(path=(Square(4, 3), Square(4, 4)))
    black_push = Trajectory(path=(Square(4, 4), Square(4, 3)))
    assert not vertex_conflict(white_push, black_push)
    assert edge_conflict(white_push, black_push)
    assert conflicts(white_push, black_push)


def test_no_conflict_for_independent_trajectories() -> None:
    t1 = Trajectory(path=(Square(0, 0), Square(0, 1)))
    t2 = Trajectory(path=(Square(7, 7), Square(7, 6)))
    assert not conflicts(t1, t2)


def test_vacated_square_theorem_knight_leaving_does_not_block_rook() -> None:
    # a rook sliding a5-h5 passes unharmed through a square a knight vacated
    rook_slide = Trajectory(path=(Square(0, 4), Square(4, 4), Square(7, 4)))
    knight_departure = Trajectory(path=(Square(4, 4), Square(2, 5)), is_jump=True)
    # knight's swept set excludes its origin, so the rook passes unharmed
    assert not vertex_conflict(rook_slide, knight_departure)


def test_annihilation_rank_is_max_then_min() -> None:
    assert annihilation_rank(1, 1) == (1, 1)
    assert annihilation_rank(2, 1) == (2, 1)
    assert annihilation_rank(1, 2) == (2, 1)


def test_annihilation_rank_symmetric_under_swap() -> None:
    assert annihilation_rank(1, 2) == annihilation_rank(2, 1)


def _sample_state() -> State:
    white_king = Token(id=1, color=Color.WHITE, typ="k")
    black_king = Token(id=2, color=Color.BLACK, typ="k")
    white_rook = Token(id=3, color=Color.WHITE, typ="r")
    black_knight = Token(id=4, color=Color.BLACK, typ="n")
    return build_state(
        {
            white_king: Square(4, 0),
            black_king: Square(4, 7),
            white_rook: Square(0, 0),
            black_knight: Square(1, 7),
        },
        cooldown=frozenset({black_knight}),
        reservations_white=(
            Reservation(defender=white_rook, protege=white_king, age=(0, 0)),
        ),
        castling_rights=CastlingRights(white_kingside=False, black_queenside=False),
        no_progress_counter=3,
        phase_index=5,
    )


def test_mirror_state_is_an_involution() -> None:
    state = _sample_state()
    mirrored_twice = mirror_state(mirror_state(state))
    assert mirrored_twice == state


def test_mirror_state_flips_color_and_rank() -> None:
    white_king = Token(id=1, color=Color.WHITE, typ="k")
    state = build_state({white_king: Square(4, 0)})
    mirrored = mirror_state(state)
    (token,) = mirrored.board.keys()
    assert token.color is Color.BLACK
    assert token.id == 1
    assert mirrored.board[token] == Square(4, 7)


def test_mirror_state_swaps_reservation_lists() -> None:
    defender = Token(id=1, color=Color.WHITE, typ="r")
    protege = Token(id=2, color=Color.WHITE, typ="p")
    state = build_state(
        {defender: Square(0, 0), protege: Square(0, 1)},
        reservations_white=(
            Reservation(defender=defender, protege=protege, age=(0, 0)),
        ),
    )
    mirrored = mirror_state(state)
    assert mirrored.reservations_white == ()
    assert len(mirrored.reservations_black) == 1
    assert mirrored.reservations_black[0].defender.color is Color.BLACK


def test_mirror_program_is_an_involution() -> None:
    token = Token(id=1, color=Color.WHITE, typ="p")
    trajectory = Trajectory(path=(Square(4, 1), Square(4, 3)))
    program = (Move(token=token, trajectory=trajectory),)
    assert mirror_program(mirror_program(program)) == program


def test_mirror_program_relabels_reserve_action() -> None:
    defender = Token(id=1, color=Color.WHITE, typ="r")
    protege = Token(id=2, color=Color.WHITE, typ="p")
    program = (Reserve(defender=defender, protege=protege),)
    mirrored = mirror_program(program)
    action = mirrored[0]
    assert isinstance(action, Reserve)
    assert action.defender.color is Color.BLACK
    assert action.protege.color is Color.BLACK
