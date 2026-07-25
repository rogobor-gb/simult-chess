"""Phase 16.2: the affordance API — destinations, reservations, threats."""

from __future__ import annotations

from simult_chess.core.types import Color, Square
from simult_chess.referee.setup import standard_starting_state
from simult_chess.rules.ruleset import RuleSet
from simult_chess.ui.affordances import affordances, threat_overlay

RULESET = RuleSet()


def _token_at(state: object, file: int, rank: int) -> object:
    for token, sq in state.board.items():  # type: ignore[attr-defined]
        if sq == Square(file=file, rank=rank):
            return token
    raise AssertionError("no token there")


def test_pawn_and_knight_destinations_at_the_start() -> None:
    state = standard_starting_state()
    aff = affordances(state, Color.WHITE, RULESET)
    e_pawn = _token_at(state, 4, 1)
    assert aff.legal_destinations[e_pawn.id] == frozenset(  # type: ignore[attr-defined]
        {Square(4, 2), Square(4, 3)}
    )
    g_knight = _token_at(state, 6, 0)
    assert aff.legal_destinations[g_knight.id] == frozenset(  # type: ignore[attr-defined]
        {Square(5, 2), Square(7, 2)}
    )


def test_no_threats_and_no_castles_at_the_start() -> None:
    state = standard_starting_state()
    aff = affordances(state, Color.WHITE, RULESET)
    assert aff.threatened_tokens == frozenset()
    assert aff.legal_castles == frozenset()
    assert threat_overlay(state, Color.WHITE) == frozenset()


def test_a_partial_program_removes_a_used_token() -> None:
    from simult_chess.core.types import Move, Trajectory

    state = standard_starting_state()
    e_pawn = _token_at(state, 4, 1)
    e2e3 = Move(
        token=e_pawn,  # type: ignore[arg-type]
        trajectory=Trajectory(path=(Square(4, 1), Square(4, 2)), is_jump=False),
    )
    aff = affordances(state, Color.WHITE, RULESET, (e2e3,))
    # The e-pawn is now an actor (L3), so it offers no further destinations.
    assert e_pawn.id not in aff.legal_destinations  # type: ignore[attr-defined]
    # Other tokens still do.
    assert aff.legal_destinations
