from __future__ import annotations

import pytest

from simult_chess.core.types import Color, Square, Token, Trajectory


def test_square_valid_range() -> None:
    sq = Square(4, 0)
    assert (sq.file, sq.rank) == (4, 0)
    assert repr(sq) == "e1"


@pytest.mark.parametrize("file,rank", [(-1, 0), (8, 0), (0, -1), (0, 8)])
def test_square_out_of_range_rejected(file: int, rank: int) -> None:
    with pytest.raises(ValueError):
        Square(file, rank)


def test_color_opponent_is_involution() -> None:
    assert Color.WHITE.opponent is Color.BLACK
    assert Color.BLACK.opponent is Color.WHITE
    assert Color.WHITE.opponent.opponent is Color.WHITE


def test_trajectory_swept_excludes_origin() -> None:
    path = (Square(0, 0), Square(0, 1), Square(0, 2))
    traj = Trajectory(path=path)
    assert traj.origin == Square(0, 0)
    assert traj.destination == Square(0, 2)
    assert traj.swept == frozenset({Square(0, 1), Square(0, 2)})
    expected_edges = {(Square(0, 0), Square(0, 1)), (Square(0, 1), Square(0, 2))}
    assert traj.edges == frozenset(expected_edges)


def test_knight_jump_has_empty_edges_but_nonempty_swept() -> None:
    traj = Trajectory(path=(Square(1, 0), Square(2, 2)), is_jump=True)
    assert traj.swept == frozenset({Square(2, 2)})
    assert traj.edges == frozenset()


def test_jump_trajectory_must_be_single_step() -> None:
    with pytest.raises(ValueError):
        Trajectory(path=(Square(0, 0), Square(1, 1), Square(2, 2)), is_jump=True)


def test_trajectory_requires_at_least_one_step() -> None:
    with pytest.raises(ValueError):
        Trajectory(path=(Square(0, 0),))


def test_token_identity_via_id_not_object_identity() -> None:
    a = Token(id=1, color=Color.WHITE, typ="p")
    b = Token(id=1, color=Color.WHITE, typ="p")
    c = Token(id=2, color=Color.WHITE, typ="p")
    assert a == b
    assert hash(a) == hash(b)
    assert a != c
