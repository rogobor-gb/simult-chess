from __future__ import annotations

from conftest import build_state

from simult_chess.core.moves import DeclaredMove
from simult_chess.core.stages.defense_seq import resolve_defense_seq
from simult_chess.core.types import Color, Reservation, Square, Token, Trajectory
from simult_chess.rules.ruleset import RuleSet

RULESET = RuleSet(intermezzo_reading="i")

D4 = Square(3, 3)
E3 = Square(4, 2)


def _dm(
    token: Token, path: tuple[Square, ...], color: Color, index: int
) -> DeclaredMove:
    return DeclaredMove(
        token=token, trajectory=Trajectory(path=path), color=color, index=index,
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


_WorkedExample = tuple[Token, Token, Token, Token, tuple[Reservation, ...]]


def _worked_example_state() -> _WorkedExample:
    d4_pawn = Token(id=1, color=Color.WHITE, typ="p")
    e3_pawn = Token(id=2, color=Color.WHITE, typ="p")
    rook1 = Token(id=3, color=Color.BLACK, typ="r")  # attacks d4 (the protege)
    rook2 = Token(id=4, color=Color.BLACK, typ="r")  # attacks e3 (the defender)
    reservations_white = (Reservation(defender=e3_pawn, protege=d4_pawn, age=(0, 0)),)
    return d4_pawn, e3_pawn, rook1, rook2, reservations_white


def test_attack_protege_first_reproduces_reading_ii_outcome() -> None:
    # Spec §13.4: declaring the protege-attack (d4) as round 1 and the
    # defender-attack (e3) as round 2 -- the reservation is still valid when
    # d4 falls, so e3-pawn recaptures before its own attacker's turn comes.
    # Also a regression guard: e3-pawn relocates to d4 in round 1 (joining
    # fired_defenders) and must not be double-counted as still capturable
    # at its old square when round 2 resolves -- the exact-equality check
    # on captured_tokens below (not just survives(e3_pawn)) catches that.
    d4_pawn, e3_pawn, rook1, rook2, reservations_white = _worked_example_state()
    state = build_state(
        {d4_pawn: D4, e3_pawn: E3, rook1: Square(3, 7), rook2: Square(4, 7)}
    )
    survivors = (
        _dm(rook1, _line(Square(3, 7), D4), Color.BLACK, index=1),
        _dm(rook2, _line(Square(4, 7), E3), Color.BLACK, index=2),
    )

    result = resolve_defense_seq(
        survivors, survivors, state, reservations_white, (), RULESET
    )

    assert result.captured_tokens == {d4_pawn, rook1}
    assert result.survives(e3_pawn)
    assert result.survives(rook2)
    assert len(result.fired) == 1
    assert result.fired[0].defender == e3_pawn
    assert result.fired[0].captured == rook1
    assert result.occupancy[e3_pawn] == D4
    assert result.occupancy[rook2] == E3  # occupies the now-empty square


def test_attack_defender_first_defuses_the_reservation() -> None:
    # Spec §13.4's leaner half: declaring the defender-attack (e3) as round 1
    # kills e3-pawn before d4's reservation is ever checked, so when round 2
    # captures d4, its only reservation names an already-dead defender --
    # invalid, no recapture. Black loses nothing.
    d4_pawn, e3_pawn, rook1, rook2, reservations_white = _worked_example_state()
    state = build_state(
        {d4_pawn: D4, e3_pawn: E3, rook1: Square(3, 7), rook2: Square(4, 7)}
    )
    survivors = (
        _dm(rook2, _line(Square(4, 7), E3), Color.BLACK, index=1),
        _dm(rook1, _line(Square(3, 7), D4), Color.BLACK, index=2),
    )

    result = resolve_defense_seq(
        survivors, survivors, state, reservations_white, (), RULESET
    )

    assert result.captured_tokens == {d4_pawn, e3_pawn}
    assert result.survives(rook1)
    assert result.survives(rook2)
    assert result.fired == ()
    assert result.occupancy[rook1] == D4
    assert result.occupancy[rook2] == E3


def test_worked_example_orderings_diverge_under_reading_i() -> None:
    # The whole point of Reading (i): unlike Reading (ii) (inv M4, "ii"
    # branch), these two orderings of the *same* declared captures do not
    # agree -- that disagreement is the order-dependence spec section 13.4
    # documents and inv M4's "i" branch checks for.
    d4_pawn, e3_pawn, rook1, rook2, reservations_white = _worked_example_state()
    state = build_state(
        {d4_pawn: D4, e3_pawn: E3, rook1: Square(3, 7), rook2: Square(4, 7)}
    )
    protege_first = (
        _dm(rook1, _line(Square(3, 7), D4), Color.BLACK, index=1),
        _dm(rook2, _line(Square(4, 7), E3), Color.BLACK, index=2),
    )
    defender_first = (
        _dm(rook2, _line(Square(4, 7), E3), Color.BLACK, index=1),
        _dm(rook1, _line(Square(3, 7), D4), Color.BLACK, index=2),
    )

    result_a = resolve_defense_seq(
        protege_first, protege_first, state, reservations_white, (), RULESET
    )
    result_b = resolve_defense_seq(
        defender_first, defender_first, state, reservations_white, (), RULESET
    )

    assert result_a.captured_tokens != result_b.captured_tokens


def test_same_round_cross_color_tie_uses_symmetric_lookahead() -> None:
    # Both sides declare their capture as *their own* action 1 -- a genuine
    # cross-color index tie. Spec §13.4: resolved via Reading (ii)'s own
    # defender-lookahead restricted to just this round, so White's e3-pawn
    # still saves d4-pawn (fires before Black's own attacker's index would
    # otherwise matter) exactly as if it were the sole round.
    d4_pawn, e3_pawn, rook1, rook2, reservations_white = _worked_example_state()
    black_king = Token(id=200, color=Color.BLACK, typ="k")
    white_attacker = Token(id=5, color=Color.WHITE, typ="r")
    black_defended = Token(id=6, color=Color.BLACK, typ="p")
    state = build_state(
        {
            d4_pawn: D4,
            e3_pawn: E3,
            rook1: Square(3, 7),
            rook2: Square(4, 7),
            black_king: Square(7, 6),
            white_attacker: Square(0, 5),
            black_defended: Square(0, 6),
        }
    )
    survivors = (
        _dm(rook1, _line(Square(3, 7), D4), Color.BLACK, index=1),
        _dm(rook2, _line(Square(4, 7), E3), Color.BLACK, index=2),
        _dm(
            white_attacker,
            _line(Square(0, 5), Square(0, 6)),
            Color.WHITE,
            index=1,
        ),
    )

    result = resolve_defense_seq(
        survivors, survivors, state, reservations_white, (), RULESET
    )

    # The unrelated same-round White attack does not disturb the Black
    # d4/e3 battery's own outcome (still the "defense holds" case, since
    # d4 is attacker-declared before e3 within Black's own program).
    assert {d4_pawn, e3_pawn} - result.captured_tokens
    assert result.survives(e3_pawn)
    assert black_defended in result.captured_tokens
