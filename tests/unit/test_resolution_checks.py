from __future__ import annotations

from conftest import build_state

from simult_chess.core.moves import DeclaredMove
from simult_chess.core.phi import PhiResult, PhiTrace, phi
from simult_chess.core.stages.annihilate import AnnihilationEvent
from simult_chess.core.stages.defense import RecaptureFired
from simult_chess.core.stages.fizzle import FizzleOutcome
from simult_chess.core.types import (
    Castle,
    Color,
    Move,
    Reservation,
    Square,
    State,
    Token,
    Trajectory,
)
from simult_chess.invariants import resolution_checks as rc
from simult_chess.rules.ruleset import RuleSet

RULESET = RuleSet()
WHITE_KING = Token(id=100, color=Color.WHITE, typ="k")
BLACK_KING = Token(id=200, color=Color.BLACK, typ="k")


def _move(token: Token, path: tuple[Square, ...]) -> Move:
    return Move(token=token, trajectory=Trajectory(path=path))


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


def _empty_trace(**overrides: object) -> PhiTrace:
    defaults: dict[str, object] = {
        "fizzled": (),
        "executing": (),
        "annihilated": (),
        "survivors": (),
        "captured": (),
        "fired": (),
        "promoted": frozenset(),
        "cancelled": frozenset(),
    }
    defaults.update(overrides)
    return PhiTrace(**defaults)  # type: ignore[arg-type]


def _all_clean(state_pre: State, result: PhiResult) -> list[object]:
    return rc.check_all_trace(
        state_pre, result.state, result.trace, result.outcome, RULESET
    )


def test_check_all_trace_is_clean_for_correct_phi_calls() -> None:
    rook = Token(id=1, color=Color.WHITE, typ="r")
    knight = Token(id=2, color=Color.BLACK, typ="n")
    filler = Token(id=3, color=Color.BLACK, typ="p")
    state = build_state(
        {
            WHITE_KING: Square(4, 0),
            BLACK_KING: Square(4, 7),
            rook: Square(0, 0),
            knight: Square(0, 4),
            filler: Square(7, 6),
        }
    )
    path = (Square(0, 0), Square(0, 1), Square(0, 2), Square(0, 3), Square(0, 4))
    program_white = (_move(rook, path),)
    program_black = (_move(filler, (Square(7, 6), Square(7, 5))),)

    result = phi(state, program_white, program_black, RULESET)

    assert _all_clean(state, result) == []


def test_check_all_trace_is_clean_for_worked_example() -> None:
    d4_pawn = Token(id=1, color=Color.WHITE, typ="p")
    e3_pawn = Token(id=2, color=Color.WHITE, typ="p")
    rook1 = Token(id=3, color=Color.BLACK, typ="r")
    rook2 = Token(id=4, color=Color.BLACK, typ="r")
    filler = Token(id=5, color=Color.WHITE, typ="p")
    reservation = Reservation(defender=e3_pawn, protege=d4_pawn, age=(0, 0))
    state = build_state(
        {
            WHITE_KING: Square(4, 0),
            BLACK_KING: Square(4, 7),
            d4_pawn: Square(3, 3),
            e3_pawn: Square(4, 2),
            rook1: Square(3, 7),
            rook2: Square(4, 6),
            filler: Square(0, 1),
        },
        reservations_white=(reservation,),
    )
    rook1_path = (Square(3, 7), Square(3, 6), Square(3, 5), Square(3, 4), Square(3, 3))
    rook2_path = (Square(4, 6), Square(4, 5), Square(4, 4), Square(4, 3), Square(4, 2))
    program_white = (_move(filler, (Square(0, 1), Square(0, 2))),)
    program_black = (_move(rook1, rook1_path), _move(rook2, rook2_path))

    result = phi(state, program_white, program_black, RULESET)

    assert _all_clean(state, result) == []


def test_check_all_trace_is_clean_for_castling() -> None:
    rook = Token(id=1, color=Color.WHITE, typ="r")
    filler = Token(id=2, color=Color.BLACK, typ="p")
    state = build_state(
        {
            WHITE_KING: Square(4, 0),
            BLACK_KING: Square(4, 7),
            rook: Square(7, 0),
            filler: Square(0, 6),
        }
    )
    program_white = (Castle(side="king"),)
    program_black = (_move(filler, (Square(0, 6), Square(0, 5))),)

    result = phi(state, program_white, program_black, RULESET)

    assert _all_clean(state, result) == []


def test_check_all_trace_is_clean_for_promotion() -> None:
    pawn = Token(id=1, color=Color.WHITE, typ="p")
    filler = Token(id=2, color=Color.BLACK, typ="p")
    state = build_state(
        {
            WHITE_KING: Square(1, 0),
            BLACK_KING: Square(4, 7),
            pawn: Square(0, 6),
            filler: Square(7, 6),
        }
    )
    promo_trajectory = Trajectory(path=(Square(0, 6), Square(0, 7)))
    promo_move = Move(token=pawn, trajectory=promo_trajectory, promotion="q")
    program_white = (promo_move,)
    program_black = (_move(filler, (Square(7, 6), Square(7, 5))),)

    result = phi(state, program_white, program_black, RULESET)

    assert _all_clean(state, result) == []


def test_check_r5_catches_two_survivors_at_one_destination() -> None:
    white_rook = Token(id=1, color=Color.WHITE, typ="r")
    white_bishop = Token(id=2, color=Color.WHITE, typ="b")
    m1 = _dm(white_rook, (Square(0, 0), Square(0, 4)), Color.WHITE, 1)
    m2 = _dm(white_bishop, (Square(4, 0), Square(0, 4)), Color.WHITE, 2)
    trace = _empty_trace(survivors=(m1, m2))

    violations = rc.check_r5_one_survivor_per_destination(trace)

    assert len(violations) == 1
    assert violations[0].invariant_id == "R5"


def test_check_r9_catches_defender_firing_twice() -> None:
    defender = Token(id=1, color=Color.WHITE, typ="r")
    captured_a = Token(id=2, color=Color.BLACK, typ="n")
    captured_b = Token(id=3, color=Color.BLACK, typ="n")
    reservation = Reservation(defender=defender, protege=captured_a, age=(0, 0))
    fired = (
        RecaptureFired(
            defender=defender,
            captured=captured_a,
            square=Square(0, 0),
            reservation=reservation,
        ),
        RecaptureFired(
            defender=defender,
            captured=captured_b,
            square=Square(1, 1),
            reservation=reservation,
        ),
    )
    trace = _empty_trace(fired=fired)

    violations = rc.check_r9_defender_fires_once(trace)

    assert len(violations) == 1
    assert violations[0].invariant_id == "R9"


def test_check_r10_catches_fired_defender_that_also_moved() -> None:
    defender = Token(id=1, color=Color.WHITE, typ="r")
    victim = Token(id=2, color=Color.BLACK, typ="n")
    reservation = Reservation(defender=defender, protege=victim, age=(0, 0))
    fired = (
        RecaptureFired(
            defender=defender,
            captured=victim,
            square=Square(0, 0),
            reservation=reservation,
        ),
    )
    executing = (_dm(defender, (Square(5, 5), Square(5, 6)), Color.WHITE),)
    trace = _empty_trace(fired=fired, executing=executing)

    violations = rc.check_r10_mover_as_defender_forbidden(trace)

    assert len(violations) == 1
    assert violations[0].invariant_id == "R10"


def test_check_r18_catches_a_resurrected_token() -> None:
    live = Token(id=1, color=Color.WHITE, typ="k")
    ghost = Token(id=99, color=Color.BLACK, typ="q")
    state_pre = build_state({live: Square(0, 0)})
    state_post = build_state({live: Square(0, 1), ghost: Square(3, 3)})

    violations = rc.check_r18_token_conservation(state_pre, state_post)

    assert len(violations) == 1
    assert violations[0].invariant_id == "R18"


def test_check_t1_catches_outcome_mismatch_with_board() -> None:
    white_king = Token(id=1, color=Color.WHITE, typ="k")
    state_post = build_state({white_king: Square(4, 0)})  # black king absent

    violations = rc.check_t1_terminal(state_post, "draw")

    assert len(violations) == 1
    assert violations[0].invariant_id == "T1"
    assert rc.check_t1_terminal(state_post, "white_wins") == []


def test_check_t4_catches_missed_no_progress_draw() -> None:
    white_king = Token(id=1, color=Color.WHITE, typ="k")
    black_king = Token(id=2, color=Color.BLACK, typ="k")
    state_post = build_state(
        {white_king: Square(4, 0), black_king: Square(4, 7)}, no_progress_counter=50
    )
    ruleset = RuleSet(horizon=50)

    violations = rc.check_t4_no_progress(state_post, "ongoing", ruleset)

    assert len(violations) == 1
    assert violations[0].invariant_id == "T4"
    assert rc.check_t4_no_progress(state_post, "draw", ruleset) == []


def test_check_r3_catches_annihilated_pair_with_no_real_conflict() -> None:
    white_rook = Token(id=1, color=Color.WHITE, typ="r")
    black_rook = Token(id=2, color=Color.BLACK, typ="r")
    white_move = _dm(white_rook, (Square(0, 0), Square(0, 1)), Color.WHITE, 1)
    black_move = _dm(black_rook, (Square(7, 7), Square(7, 6)), Color.BLACK, 1)
    event = AnnihilationEvent(white_move=white_move, black_move=black_move, rank=(1, 1))
    trace = _empty_trace(annihilated=(event,))

    violations = rc.check_r3_edge_conflict(trace)

    assert len(violations) == 1
    assert violations[0].invariant_id == "R3"


def test_check_r1_catches_f1_fizzle_whose_target_never_moved() -> None:
    attacker = Token(id=1, color=Color.WHITE, typ="p")
    target = Token(id=2, color=Color.BLACK, typ="n")
    state_pre = build_state({attacker: Square(0, 3), target: Square(1, 4)})
    fizzled_move = _dm(attacker, (Square(0, 3), Square(1, 4)), Color.WHITE, 1)
    outcome = FizzleOutcome(move=fizzled_move, cause="F1")
    trace = _empty_trace(fizzled=(outcome,), executing=())  # target never executed

    violations = rc.check_r1_fizzle_f1(state_pre, trace)

    assert len(violations) == 1
    assert violations[0].invariant_id == "R1"
