"""Phase 16.1: check_partial_program runs L3–L6, skipping L1/L2."""

from __future__ import annotations

from simult_chess.core.legality import check_legal_program, check_partial_program
from simult_chess.core.types import Color, Move, Square, Trajectory
from simult_chess.referee.setup import standard_starting_state
from simult_chess.rules.ruleset import RuleSet

RULESET = RuleSet()


def _token_at(state: object, file: int, rank: int) -> object:
    square = Square(file=file, rank=rank)
    for token, sq in state.board.items():  # type: ignore[attr-defined]
        if sq == square:
            return token
    raise AssertionError(f"no token at {square}")


def _pawn_step(state: object, file: int, from_rank: int, to_rank: int) -> Move:
    token = _token_at(state, file, from_rank)
    step = 1 if to_rank > from_rank else -1
    ranks = range(from_rank, to_rank + step, step)
    path = tuple(Square(file, r) for r in ranks)
    return Move(token=token, trajectory=Trajectory(path=path, is_jump=False))  # type: ignore[arg-type]


def test_empty_program_is_partial_legal_but_not_fully_legal() -> None:
    state = standard_starting_state()
    # L1 (1..N budget) is skipped by the partial check, so the in-progress
    # empty program is fine; full L rejects it.
    assert check_partial_program(state, (), Color.WHITE, RULESET) == []
    assert any(
        v.invariant_id == "L1"
        for v in check_legal_program(state, (), Color.WHITE, RULESET)
    )


def test_a_single_legal_move_passes_both() -> None:
    state = standard_starting_state()
    program = (_pawn_step(state, 4, 1, 3),)  # e2-e3
    assert check_partial_program(state, program, Color.WHITE, RULESET) == []
    assert check_legal_program(state, program, Color.WHITE, RULESET) == []


def test_l3_double_use_is_illegal_under_both() -> None:
    state = standard_starting_state()
    e2e3 = _pawn_step(state, 4, 1, 3)
    program = (e2e3, e2e3)  # same token twice
    assert any(
        v.invariant_id == "L3"
        for v in check_partial_program(state, program, Color.WHITE, RULESET)
    )
    assert any(
        v.invariant_id == "L3"
        for v in check_legal_program(state, program, Color.WHITE, RULESET)
    )


def test_l6_geometric_illegality_is_caught_partially() -> None:
    state = standard_starting_state()
    # e2 pawn "moving" to e5 in one step is geometrically illegal (L6).
    token = _token_at(state, 4, 1)
    bad = Move(
        token=token,  # type: ignore[arg-type]
        trajectory=Trajectory(
            path=(Square(4, 1), Square(4, 4)), is_jump=False
        ),
    )
    assert any(
        v.invariant_id == "L6"
        for v in check_partial_program(state, (bad,), Color.WHITE, RULESET)
    )
