from __future__ import annotations

from conftest import build_state

from simult_chess.core.moves import DeclaredMove
from simult_chess.core.stages.defense import resolve_defense
from simult_chess.core.types import Color, Reservation, Square, Token, Trajectory
from simult_chess.rules.ruleset import RuleSet

RULESET = RuleSet()

D4 = Square(3, 3)
E3 = Square(4, 2)


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


def _line(origin: Square, destination: Square) -> tuple[Square, ...]:
    step_file = (destination.file > origin.file) - (destination.file < origin.file)
    step_rank = (destination.rank > origin.rank) - (destination.rank < origin.rank)
    squares = [origin]
    file, rank = origin.file, origin.rank
    while (file, rank) != (destination.file, destination.rank):
        file, rank = file + step_file, rank + step_rank
        squares.append(Square(file, rank))
    return tuple(squares)


def test_r7_worked_example_d4_e3_defense_holds() -> None:
    # White pawns d4, e3; e3 defends d4. Black plays Rxd4 and Rxe3.
    d4_pawn = Token(id=1, color=Color.WHITE, typ="p")
    e3_pawn = Token(id=2, color=Color.WHITE, typ="p")
    rook1 = Token(id=3, color=Color.BLACK, typ="r")  # attacks d4
    rook2 = Token(id=4, color=Color.BLACK, typ="r")  # attacks e3

    state = build_state(
        {
            d4_pawn: D4,
            e3_pawn: E3,
            rook1: Square(3, 7),
            rook2: Square(4, 7),
        }
    )
    reservations_white = (Reservation(defender=e3_pawn, protege=d4_pawn, age=(0, 0)),)

    survivors = (
        _dm(rook1, _line(Square(3, 7), D4), Color.BLACK, index=1),
        _dm(rook2, _line(Square(4, 7), E3), Color.BLACK, index=2),
    )

    result = resolve_defense(
        survivors, survivors, state, reservations_white, (), RULESET
    )

    assert result.captured_tokens == {d4_pawn, rook1}
    assert result.survives(e3_pawn)
    assert result.survives(rook2)
    assert len(result.fired) == 1
    assert result.fired[0].defender == e3_pawn
    assert result.fired[0].captured == rook1
    assert result.occupancy[e3_pawn] == D4
    assert result.occupancy[rook2] == E3  # occupies the now-empty square, no capture


def test_m4_worked_example_is_independent_of_attacker_declaration_order() -> None:
    d4_pawn = Token(id=1, color=Color.WHITE, typ="p")
    e3_pawn = Token(id=2, color=Color.WHITE, typ="p")
    rook1 = Token(id=3, color=Color.BLACK, typ="r")
    rook2 = Token(id=4, color=Color.BLACK, typ="r")

    state = build_state(
        {d4_pawn: D4, e3_pawn: E3, rook1: Square(3, 7), rook2: Square(4, 7)}
    )
    reservations_white = (Reservation(defender=e3_pawn, protege=d4_pawn, age=(0, 0)),)
    survivors = (
        _dm(rook1, _line(Square(3, 7), D4), Color.BLACK, index=1),
        _dm(rook2, _line(Square(4, 7), E3), Color.BLACK, index=2),
    )

    result_ab = resolve_defense(
        survivors, survivors, state, reservations_white, (), RULESET, tie_break=(D4, E3)
    )
    result_ba = resolve_defense(
        survivors, survivors, state, reservations_white, (), RULESET, tie_break=(E3, D4)
    )

    assert result_ab.captured_tokens == result_ba.captured_tokens == {d4_pawn, rook1}
    assert result_ab.occupancy == result_ba.occupancy


def test_r8_oldest_valid_reservation_fires_not_the_newer_one() -> None:
    protege = Token(id=1, color=Color.WHITE, typ="p")
    older_defender = Token(id=2, color=Color.WHITE, typ="r")  # d1, older reservation
    newer_defender = Token(id=3, color=Color.WHITE, typ="b")  # a1, newer reservation
    attacker = Token(id=4, color=Color.BLACK, typ="r")

    state = build_state(
        {
            protege: D4,
            older_defender: Square(3, 0),
            newer_defender: Square(0, 0),
            attacker: Square(3, 7),
        }
    )
    reservations_white = (
        Reservation(defender=older_defender, protege=protege, age=(0, 0)),
        Reservation(defender=newer_defender, protege=protege, age=(0, 1)),
    )
    survivors = (_dm(attacker, _line(Square(3, 7), D4), Color.BLACK),)

    result = resolve_defense(
        survivors, survivors, state, reservations_white, (), RULESET
    )

    assert len(result.fired) == 1
    assert result.fired[0].defender == older_defender
    assert result.survives(newer_defender)
    assert result.occupancy[newer_defender] == Square(0, 0)  # never moved


def test_r9_defender_fires_at_most_once_for_two_attacked_proteges() -> None:
    defender = Token(id=1, color=Color.WHITE, typ="r")  # on d4, defends both
    protege1 = Token(id=2, color=Color.WHITE, typ="p")  # d2
    protege2 = Token(id=3, color=Color.WHITE, typ="p")  # b4
    attacker1 = Token(id=4, color=Color.BLACK, typ="n")
    attacker2 = Token(id=5, color=Color.BLACK, typ="n")

    d2 = Square(3, 1)
    b4 = Square(1, 3)
    state = build_state(
        {
            defender: D4,
            protege1: d2,
            protege2: b4,
            attacker1: Square(5, 2),  # knight jump to d2
            attacker2: Square(3, 4),  # knight jump to b4
        }
    )
    reservations_white = (
        Reservation(defender=defender, protege=protege1, age=(0, 0)),
        Reservation(defender=defender, protege=protege2, age=(0, 1)),
    )
    survivors = (
        _dm(attacker1, (Square(5, 2), d2), Color.BLACK, index=1),
        _dm(attacker2, (Square(3, 4), b4), Color.BLACK, index=2),
    )

    result = resolve_defense(
        survivors, survivors, state, reservations_white, (), RULESET, tie_break=(d2, b4)
    )

    assert len(result.fired) == 1
    assert result.fired[0].defender == defender
    assert result.fired[0].square == d2
    # protege1 is avenged; protege2 has no recapture left for it
    assert not result.survives(protege1)
    assert not result.survives(attacker1)
    assert not result.survives(protege2)
    assert result.survives(attacker2)


def test_r10_mover_as_defender_is_automatically_invalid() -> None:
    protege = Token(id=1, color=Color.WHITE, typ="p")
    defender = Token(id=2, color=Color.WHITE, typ="r")  # declares its own move
    attacker = Token(id=3, color=Color.BLACK, typ="r")

    state = build_state({protege: D4, defender: Square(0, 0), attacker: Square(3, 7)})
    reservations_white = (Reservation(defender=defender, protege=protege, age=(0, 0)),)
    survivors = (
        _dm(attacker, _line(Square(3, 7), D4), Color.BLACK, index=1),
        _dm(defender, (Square(0, 0), Square(0, 4)), Color.WHITE, index=1),
    )

    result = resolve_defense(
        survivors, survivors, state, reservations_white, (), RULESET
    )

    assert result.fired == ()
    assert not result.survives(protege)
    assert result.survives(attacker)
    assert result.survives(defender)
    assert result.occupancy[defender] == Square(0, 4)


def test_r11_mutual_defense_cycle_resolves_to_base_semantics() -> None:
    # P and Q are both rooks on rank 4, each defending the other; both attacked.
    p = Token(id=1, color=Color.WHITE, typ="r")
    q = Token(id=2, color=Color.WHITE, typ="r")
    attacker_p = Token(id=3, color=Color.BLACK, typ="r")
    attacker_q = Token(id=4, color=Color.BLACK, typ="r")

    p_square = Square(0, 3)
    q_square = Square(7, 3)
    state = build_state(
        {
            p: p_square,
            q: q_square,
            attacker_p: Square(0, 7),
            attacker_q: Square(7, 0),
        }
    )
    reservations_white = (
        Reservation(defender=p, protege=q, age=(0, 0)),
        Reservation(defender=q, protege=p, age=(0, 1)),
    )
    survivors = (
        _dm(attacker_p, _line(Square(0, 7), p_square), Color.BLACK, index=1),
        _dm(attacker_q, _line(Square(7, 0), q_square), Color.BLACK, index=2),
    )

    result = resolve_defense(
        survivors, survivors, state, reservations_white, (), RULESET
    )

    assert result.fired == ()
    assert result.captured_tokens == {p, q}
    assert result.survives(attacker_p)
    assert result.survives(attacker_q)


def test_r12_multi_level_chain_terminates_with_correct_final_holder() -> None:
    # d4 pawn defended by e3 pawn; the attacking rook is itself defended by a
    # bishop. Chain: rook takes d4 -> e3-pawn recaptures (kills rook) ->
    # bishop recaptures (kills e3-pawn, now sitting on d4) -> bishop stands.
    d4_pawn = Token(id=1, color=Color.WHITE, typ="p")
    e3_pawn = Token(id=2, color=Color.WHITE, typ="p")
    rook = Token(id=3, color=Color.BLACK, typ="r")
    bishop = Token(id=4, color=Color.BLACK, typ="b")

    a7 = Square(0, 6)
    state = build_state(
        {d4_pawn: D4, e3_pawn: E3, rook: Square(3, 7), bishop: a7}
    )
    reservations_white = (Reservation(defender=e3_pawn, protege=d4_pawn, age=(0, 0)),)
    reservations_black = (Reservation(defender=bishop, protege=rook, age=(0, 0)),)
    survivors = (_dm(rook, _line(Square(3, 7), D4), Color.BLACK),)

    result = resolve_defense(
        survivors, survivors, state, reservations_white, reservations_black, RULESET
    )

    assert result.captured_tokens == {d4_pawn, rook, e3_pawn}
    assert len(result.fired) == 2
    assert result.fired[0].defender == e3_pawn
    assert result.fired[0].captured == rook
    assert result.fired[1].defender == bishop
    assert result.fired[1].captured == e3_pawn
    assert result.occupancy[bishop] == D4


def test_undefended_capture_stands_with_no_recapture() -> None:
    protege = Token(id=1, color=Color.WHITE, typ="p")
    attacker = Token(id=2, color=Color.BLACK, typ="r")
    state = build_state({protege: D4, attacker: Square(3, 7)})
    survivors = (_dm(attacker, _line(Square(3, 7), D4), Color.BLACK),)

    result = resolve_defense(survivors, survivors, state, (), (), RULESET)

    assert result.fired == ()
    assert result.captured_tokens == {protege}
    assert result.survives(attacker)
    assert result.occupancy[attacker] == D4


def test_r6_non_pawn_capture_of_a_vacated_square_is_not_a_capture() -> None:
    # A knight leaves d4 the same phase a rook's declared trajectory ends
    # there (d8-d4): the knight vacated, so the rook just arrives peacefully
    # -- this must not be recorded as capturing the knight (R6, generalized
    # past the pawn-only F1 fizzle).
    knight = Token(id=1, color=Color.WHITE, typ="n")
    rook = Token(id=2, color=Color.BLACK, typ="r")
    state = build_state({knight: D4, rook: Square(3, 7)})
    knight_move = _dm(knight, (D4, Square(1, 4)), Color.WHITE, index=1)
    rook_move = _dm(rook, _line(Square(3, 7), D4), Color.BLACK, index=1)
    executing = (knight_move, rook_move)

    result = resolve_defense(executing, executing, state, (), (), RULESET)

    assert result.fired == ()
    assert result.captured_tokens == frozenset()
    assert result.occupancy[knight] == Square(1, 4)
    assert result.occupancy[rook] == D4
