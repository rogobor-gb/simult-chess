from __future__ import annotations

from conftest import build_state

from simult_chess.core import geometry
from simult_chess.core.types import CastlingRights, Color, Square, Token


def test_rook_slides_on_open_board() -> None:
    rook = Token(id=1, color=Color.WHITE, typ="r")
    state = build_state({rook: Square(3, 3)})
    trajectories = geometry.pseudo_legal_trajectories(state, rook)
    destinations = {t.destination for t in trajectories}
    # full rank + file minus origin: 7 (file) + 7 (rank) = 14 destinations
    assert len(destinations) == 14
    assert Square(3, 0) in destinations
    assert Square(7, 3) in destinations


def test_rook_stops_before_own_piece() -> None:
    rook = Token(id=1, color=Color.WHITE, typ="r")
    blocker = Token(id=2, color=Color.WHITE, typ="p")
    state = build_state({rook: Square(3, 3), blocker: Square(3, 5)})
    destinations = {
        t.destination for t in geometry.pseudo_legal_trajectories(state, rook)
    }
    assert Square(3, 4) in destinations
    assert Square(3, 5) not in destinations
    assert Square(3, 6) not in destinations


def test_rook_captures_enemy_but_no_further() -> None:
    rook = Token(id=1, color=Color.WHITE, typ="r")
    enemy = Token(id=2, color=Color.BLACK, typ="p")
    state = build_state({rook: Square(3, 3), enemy: Square(3, 5)})
    destinations = {
        t.destination for t in geometry.pseudo_legal_trajectories(state, rook)
    }
    assert Square(3, 4) in destinations
    assert Square(3, 5) in destinations
    assert Square(3, 6) not in destinations


def test_bishop_diagonal_on_open_board() -> None:
    bishop = Token(id=1, color=Color.WHITE, typ="b")
    state = build_state({bishop: Square(3, 3)})
    destinations = {
        t.destination for t in geometry.pseudo_legal_trajectories(state, bishop)
    }
    assert Square(0, 0) in destinations
    assert Square(6, 6) in destinations
    assert Square(3, 4) not in destinations  # not a diagonal


def test_knight_jumps_over_occupied_squares() -> None:
    knight = Token(id=1, color=Color.WHITE, typ="n")
    blocker = Token(id=2, color=Color.WHITE, typ="p")
    state = build_state({knight: Square(1, 0), blocker: Square(1, 1)})
    trajectories = geometry.pseudo_legal_trajectories(state, knight)
    destinations = {t.destination for t in trajectories}
    assert Square(2, 2) in destinations  # b1-c3-style jump, unblocked by b2
    assert all(t.is_jump for t in trajectories)
    assert all(t.edges == frozenset() for t in trajectories)


def test_knight_cannot_land_on_own_piece_but_can_capture_enemy() -> None:
    knight = Token(id=1, color=Color.WHITE, typ="n")
    own = Token(id=2, color=Color.WHITE, typ="p")
    enemy = Token(id=3, color=Color.BLACK, typ="p")
    state = build_state({knight: Square(1, 0), own: Square(3, 1), enemy: Square(2, 2)})
    destinations = {
        t.destination for t in geometry.pseudo_legal_trajectories(state, knight)
    }
    assert Square(3, 1) not in destinations
    assert Square(2, 2) in destinations


def test_king_steps_one_square_and_captures() -> None:
    king = Token(id=1, color=Color.WHITE, typ="k")
    own = Token(id=2, color=Color.WHITE, typ="p")
    enemy = Token(id=3, color=Color.BLACK, typ="p")
    state = build_state({king: Square(4, 4), own: Square(4, 5), enemy: Square(5, 5)})
    destinations = {
        t.destination for t in geometry.pseudo_legal_trajectories(state, king)
    }
    assert Square(4, 5) not in destinations
    assert Square(5, 5) in destinations
    assert len(destinations) == 7  # 8 neighbors, minus the one own-occupied square


def test_pawn_single_and_double_push_from_start_rank() -> None:
    pawn = Token(id=1, color=Color.WHITE, typ="p")
    state = build_state({pawn: Square(4, 1)})
    destinations = {
        t.destination for t in geometry.pseudo_legal_trajectories(state, pawn)
    }
    assert destinations == {Square(4, 2), Square(4, 3)}


def test_pawn_double_push_blocked_by_occupied_intermediate() -> None:
    pawn = Token(id=1, color=Color.WHITE, typ="p")
    blocker = Token(id=2, color=Color.BLACK, typ="p")
    state = build_state({pawn: Square(4, 1), blocker: Square(4, 2)})
    destinations = {
        t.destination for t in geometry.pseudo_legal_trajectories(state, pawn)
    }
    assert destinations == set()  # push blocked; no enemy on either diagonal


def test_pawn_diagonal_capture_requires_enemy_present() -> None:
    pawn = Token(id=1, color=Color.WHITE, typ="p")
    state = build_state({pawn: Square(4, 4)})
    destinations = {
        t.destination for t in geometry.pseudo_legal_trajectories(state, pawn)
    }
    assert Square(3, 5) not in destinations
    assert Square(5, 5) not in destinations

    enemy = Token(id=2, color=Color.BLACK, typ="p")
    state_with_enemy = build_state({pawn: Square(4, 4), enemy: Square(5, 5)})
    trajectories = geometry.pseudo_legal_trajectories(state_with_enemy, pawn)
    destinations_with_enemy = {t.destination for t in trajectories}
    assert Square(5, 5) in destinations_with_enemy


def test_black_pawn_pushes_toward_rank_zero() -> None:
    pawn = Token(id=1, color=Color.BLACK, typ="p")
    state = build_state({pawn: Square(4, 6)})
    destinations = {
        t.destination for t in geometry.pseudo_legal_trajectories(state, pawn)
    }
    assert destinations == {Square(4, 5), Square(4, 4)}


def test_capturing_pattern_ignores_friendly_occupant_at_target() -> None:
    defender = Token(id=1, color=Color.WHITE, typ="r")
    protege = Token(id=2, color=Color.WHITE, typ="p")
    state = build_state({defender: Square(0, 0), protege: Square(0, 3)})
    pattern = geometry.capturing_pattern_trajectory(state, defender, Square(0, 3))
    assert pattern is not None
    assert pattern.destination == Square(0, 3)


def test_capturing_pattern_requires_clear_interior_for_sliders() -> None:
    defender = Token(id=1, color=Color.WHITE, typ="r")
    protege = Token(id=2, color=Color.WHITE, typ="p")
    blocker = Token(id=3, color=Color.BLACK, typ="n")
    state = build_state(
        {defender: Square(0, 0), protege: Square(0, 3), blocker: Square(0, 1)}
    )
    assert geometry.capturing_pattern_trajectory(state, defender, Square(0, 3)) is None


def test_capturing_pattern_pawn_requires_diagonal_not_push() -> None:
    defender = Token(id=1, color=Color.WHITE, typ="p")
    protege = Token(id=2, color=Color.WHITE, typ="p")
    state = build_state({defender: Square(4, 4), protege: Square(4, 5)})
    assert geometry.capturing_pattern_trajectory(state, defender, Square(4, 5)) is None
    state2 = build_state({defender: Square(4, 4), protege: Square(5, 5)})
    pattern = geometry.capturing_pattern_trajectory(state2, defender, Square(5, 5))
    assert pattern is not None


def test_castle_move_legal_white_kingside() -> None:
    king = Token(id=1, color=Color.WHITE, typ="k")
    rook = Token(id=2, color=Color.WHITE, typ="r")
    state = build_state({king: Square(4, 0), rook: Square(7, 0)})
    move = geometry.castle_move(state, Color.WHITE, "king")
    assert move is not None
    assert move.king_trajectory.destination == Square(6, 0)
    assert move.rook_trajectory.destination == Square(5, 0)


def test_castle_move_illegal_without_right() -> None:
    king = Token(id=1, color=Color.WHITE, typ="k")
    rook = Token(id=2, color=Color.WHITE, typ="r")
    state = build_state(
        {king: Square(4, 0), rook: Square(7, 0)},
        castling_rights=CastlingRights(white_kingside=False),
    )
    assert geometry.castle_move(state, Color.WHITE, "king") is None


def test_castle_move_illegal_when_path_blocked() -> None:
    king = Token(id=1, color=Color.WHITE, typ="k")
    rook = Token(id=2, color=Color.WHITE, typ="r")
    blocker = Token(id=3, color=Color.WHITE, typ="b")
    state = build_state({king: Square(4, 0), rook: Square(7, 0), blocker: Square(5, 0)})
    assert geometry.castle_move(state, Color.WHITE, "king") is None


def test_castle_move_illegal_when_rook_not_home() -> None:
    king = Token(id=1, color=Color.WHITE, typ="k")
    state = build_state({king: Square(4, 0)})
    assert geometry.castle_move(state, Color.WHITE, "king") is None
