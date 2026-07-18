from __future__ import annotations

import random

from conftest import build_state

from simult_chess.agents.candidates import (
    cancel_candidates,
    move_candidates,
    reserve_candidates,
)
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
from simult_chess.learn.action_grid import (
    CANCEL_OFFSET,
    CASTLE_OFFSET,
    MOVE_BLOCK,
    MOVE_TYPES,
    NO_SECOND_INDEX,
    RESERVE_OFFSET,
    SLOT_SIZE,
    encode_action,
    square_index,
)
from simult_chess.referee.setup import standard_starting_state
from simult_chess.rules.ruleset import RuleSet

RULESET = RuleSet()


def test_block_layout_sums_to_the_documented_slot_size() -> None:
    assert MOVE_TYPES == 76
    assert MOVE_BLOCK == 64 * 76 == 4864
    assert CASTLE_OFFSET == 4864
    assert RESERVE_OFFSET == 4866
    assert CANCEL_OFFSET == 8962
    assert SLOT_SIZE == 9026
    assert NO_SECOND_INDEX == SLOT_SIZE


def _dummy_state() -> object:
    king = Token(id=1, color=Color.WHITE, typ="k")
    return build_state({king: Square(4, 0)})


def test_encode_a_sliding_move() -> None:
    # Rook a1 -> a4: from-square 0, direction N (index 0), distance 3.
    rook = Token(id=1, color=Color.WHITE, typ="r")
    trajectory = Trajectory(
        path=(Square(0, 0), Square(0, 1), Square(0, 2), Square(0, 3))
    )
    move = Move(token=rook, trajectory=trajectory)
    assert encode_action(move, _dummy_state()) == 0 * MOVE_TYPES + (0 * 7 + 2)


def test_encode_a_knight_move() -> None:
    # Knight b1 -> c3: from-square 1, knight delta (1, 2) -> knight index 0.
    knight = Token(id=1, color=Color.WHITE, typ="n")
    trajectory = Trajectory(path=(Square(1, 0), Square(2, 2)), is_jump=True)
    move = Move(token=knight, trajectory=trajectory)
    assert encode_action(move, _dummy_state()) == 1 * MOVE_TYPES + (56 + 0)


def test_encode_a_promotion_move() -> None:
    # White pawn a7 -> a8 promoting to queen: from-square 48, promo dir 1
    # (dfile 0), promo index 3 (q) -> move_type 64 + 1*4 + 3 = 71.
    pawn = Token(id=1, color=Color.WHITE, typ="p")
    trajectory = Trajectory(path=(Square(0, 6), Square(0, 7)))
    move = Move(token=pawn, trajectory=trajectory, promotion="q")
    assert encode_action(move, _dummy_state()) == 48 * MOVE_TYPES + 71


def test_encode_castle_entries() -> None:
    assert encode_action(Castle(side="king"), _dummy_state()) == CASTLE_OFFSET
    assert encode_action(Castle(side="queen"), _dummy_state()) == CASTLE_OFFSET + 1


def test_encode_reserve_and_cancel() -> None:
    # Defender e3 (file 4, rank 2), protege d4 (file 3, rank 3).
    defender = Token(id=1, color=Color.WHITE, typ="p")
    protege = Token(id=2, color=Color.WHITE, typ="p")
    reservation = Reservation(defender=defender, protege=protege, age=(0, 0))
    state = build_state(
        {defender: Square(4, 2), protege: Square(3, 3)},
        reservations_white=(reservation,),
    )
    d, p = square_index(4, 2), square_index(3, 3)
    assert encode_action(Reserve(defender=defender, protege=protege), state) == (
        RESERVE_OFFSET + d * 64 + p
    )
    assert encode_action(Cancel(reservation=reservation), state) == CANCEL_OFFSET + p


def test_cancels_sharing_a_protege_square_collapse_to_one_entry() -> None:
    # D4 (oldest-valid tie-break, spec §6.4): several reservations defending the
    # same protege collapse to the single protege-square-keyed Cancel entry --
    # the head cannot distinguish them, matching R-multi-in's firing order.
    protege = Token(id=1, color=Color.WHITE, typ="n")
    rook_def = Token(id=2, color=Color.WHITE, typ="r")
    queen_def = Token(id=3, color=Color.WHITE, typ="q")
    reservation_a = Reservation(defender=rook_def, protege=protege, age=(0, 0))
    reservation_b = Reservation(defender=queen_def, protege=protege, age=(1, 0))
    state = build_state(
        {protege: Square(3, 3), rook_def: Square(3, 0), queen_def: Square(0, 3)},
        reservations_white=(reservation_a, reservation_b),
    )
    assert encode_action(Cancel(reservation=reservation_a), state) == encode_action(
        Cancel(reservation=reservation_b), state
    )


def _block_of(index: int) -> str:
    if index < CASTLE_OFFSET:
        return "move"
    if index < RESERVE_OFFSET:
        return "castle"
    if index < CANCEL_OFFSET:
        return "reserve"
    return "cancel"


def _individually_admissible(state: object, color: Color) -> list[object]:
    rng = random.Random(0)
    return [
        *move_candidates(state, color, rng),  # type: ignore[arg-type]
        *reserve_candidates(state, color),  # type: ignore[arg-type]
        *cancel_candidates(state, color),  # type: ignore[arg-type]
    ]


def test_encode_is_injective_and_block_correct_over_random_play() -> None:
    # Every distinct individually-admissible action of a state must land on a
    # distinct in-range index in the block matching its kind. Sweep states
    # reached by random self-play so promotions, reservations and captures
    # actually occur.
    from simult_chess.agents.random_legal import random_legal_program
    from simult_chess.core.phi import phi

    rng = random.Random(7)
    states = [standard_starting_state()]
    state = states[0]
    for _ in range(60):
        pw = random_legal_program(state, Color.WHITE, RULESET, rng)
        pb = random_legal_program(state, Color.BLACK, RULESET, rng)
        result = phi(state, pw, pb, RULESET)
        if result.outcome != "ongoing":
            break
        state = result.state
        states.append(state)

    expected_block = {
        Move: "move",
        Castle: "castle",
        Reserve: "reserve",
        Cancel: "cancel",
    }
    for state in states:
        for color in (Color.WHITE, Color.BLACK):
            seen: dict[int, object] = {}
            for action in _individually_admissible(state, color):
                index = encode_action(action, state)  # type: ignore[arg-type]
                assert 0 <= index < SLOT_SIZE
                assert _block_of(index) == expected_block[type(action)]
                previous = seen.get(index)
                if previous is not None and previous != action:
                    # The one permitted collision (D4): two Cancels whose
                    # reservations share a protege square collapse to the single
                    # protege-square-keyed entry (decoded to the oldest).
                    both_cancel = isinstance(previous, Cancel) and isinstance(
                        action, Cancel
                    )
                    same_protege_square = both_cancel and (
                        state.board[previous.reservation.protege]
                        == state.board[action.reservation.protege]
                    )
                    if not same_protege_square:
                        raise AssertionError(
                            f"encode collision: {previous!r} and {action!r} -> {index}"
                        )
                seen[index] = action
