"""v3: `project_slot1_marginal`/`project_slot2_conditional` (`row_sketch.
py`), the two halves of the roadmap's own factorisation language ("the
policy target x*_tau... factorises into a slot-1 marginal and a slot-2
conditional"). `project_slot1_marginal` previously had no dedicated unit
test of its own (only exercised indirectly via `test_selfplay_row_sketch.
py`'s integration tests); both get one together here, against a hand-
built pool with a known, hand-verifiable expected marginal/conditional --
pure aggregation logic, deliberately decoupled from any real search or
legality machinery (`project_slot1_marginal`/`project_slot2_conditional`
never touch either, so a minimal placeholder `State` is enough)."""

from __future__ import annotations

import numpy as np
import pytest

from simult_chess.core.types import (
    Bookkeeping,
    CastlingRights,
    Color,
    Move,
    Square,
    State,
    Token,
    Trajectory,
)
from simult_chess.learn.action_grid import NO_SECOND_INDEX, encode_action
from simult_chess.learn.row_sketch import (
    project_slot1_marginal,
    project_slot2_conditional,
)

_KING = Token(id=1, color=Color.WHITE, typ="k")
_KNIGHT = Token(id=2, color=Color.WHITE, typ="n")

_STATE = State(
    board={_KING: Square(0, 1), _KNIGHT: Square(3, 3)},
    cooldown=frozenset(),
    reservations_white=(),
    reservations_black=(),
    bookkeeping=Bookkeeping(
        castling_rights=CastlingRights(False, False, False, False),
        repetition_ledger={},
        no_progress_counter=0,
        phase_index=0,
    ),
)

_A1_A = Move(
    token=_KING,
    trajectory=Trajectory(path=(Square(0, 1), Square(1, 0)), is_jump=False),
    promotion=None,
)
_A1_B = Move(
    token=_KNIGHT,
    trajectory=Trajectory(path=(Square(3, 3), Square(4, 5)), is_jump=True),
    promotion=None,
)
_A2_X = Move(
    token=_KING,
    trajectory=Trajectory(path=(Square(1, 0), Square(2, 0)), is_jump=False),
    promotion=None,
)
_A2_Y = Move(
    token=_KNIGHT,
    trajectory=Trajectory(path=(Square(4, 5), Square(5, 3)), is_jump=True),
    promotion=None,
)

# Programs and a hand-picked (not search-derived) strategy, chosen to make
# the expected marginal/conditional easy to verify by hand:
#   i=0: (A1_A,)        mass 0.4  -> a1=A1_A, a2=NO_SECOND
#   i=1: (A1_A, A2_X)   mass 0.1  -> a1=A1_A, a2=A2_X
#   i=2: (A1_B, A2_Y)   mass 0.3  -> a1=A1_B, a2=A2_Y
#   i=3: (A1_B,)        mass 0.2  -> a1=A1_B, a2=NO_SECOND
_PROGRAMS = ((_A1_A,), (_A1_A, _A2_X), (_A1_B, _A2_Y), (_A1_B,))
_STRATEGY = np.array([0.4, 0.1, 0.3, 0.2])


def test_slot1_marginal_sums_mass_by_first_action() -> None:
    marginal = project_slot1_marginal(_STRATEGY, _PROGRAMS, _STATE)
    a1_a_index = encode_action(_A1_A, _STATE)
    a1_b_index = encode_action(_A1_B, _STATE)
    assert marginal[a1_a_index] == pytest.approx(0.5)
    assert marginal[a1_b_index] == pytest.approx(0.5)
    assert sum(marginal.values()) == pytest.approx(1.0)


def test_slot2_conditional_given_a1_a() -> None:
    a1_a_index = encode_action(_A1_A, _STATE)
    a2_x_index = encode_action(_A2_X, _STATE)
    conditional = project_slot2_conditional(_STRATEGY, _PROGRAMS, _STATE, a1_a_index)
    assert conditional[NO_SECOND_INDEX] == pytest.approx(0.8)
    assert conditional[a2_x_index] == pytest.approx(0.2)
    assert sum(conditional.values()) == pytest.approx(1.0)


def test_slot2_conditional_given_a1_b() -> None:
    a1_b_index = encode_action(_A1_B, _STATE)
    a2_y_index = encode_action(_A2_Y, _STATE)
    conditional = project_slot2_conditional(_STRATEGY, _PROGRAMS, _STATE, a1_b_index)
    assert conditional[a2_y_index] == pytest.approx(0.6)
    assert conditional[NO_SECOND_INDEX] == pytest.approx(0.4)
    assert sum(conditional.values()) == pytest.approx(1.0)


def test_slot2_conditional_falls_back_to_no_second_when_a1_has_no_mass() -> None:
    # Defensive path: an a1_index that no pool entry's first action matches.
    unrelated_index = encode_action(_A2_X, _STATE) + 1
    conditional = project_slot2_conditional(
        _STRATEGY, _PROGRAMS, _STATE, unrelated_index
    )
    assert conditional == {NO_SECOND_INDEX: 1.0}
