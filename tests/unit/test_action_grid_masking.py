"""Per-slot legality masking (Phase 13b, docs/LEARNING_DESIGN.md §3.3/§4.3).

The critical correctness contract: the codec's legal-program set is *sound*
(every program it yields is legal by `L(s,pi)`) and *complete* against the
Phase-12 exhaustive enumeration (it reaches at least every program that
`enumerate_legal_programs` does), plus the aggressive-dual programs that
enumeration structurally misses.
"""

from __future__ import annotations

import random

import pytest
from conftest import build_state

from simult_chess.core.legality import is_legal_program
from simult_chess.core.phi import phi
from simult_chess.core.types import (
    Cancel,
    Color,
    Move,
    Reservation,
    Reserve,
    Square,
    Token,
    Trajectory,
)
from simult_chess.learn.action_grid import (
    NO_SECOND_INDEX,
    encode_action,
    slot1_legal_actions,
    slot2_legal_actions,
)
from simult_chess.referee.setup import standard_starting_state
from simult_chess.rules.ruleset import RuleSet

RULESET = RuleSet()


def _self_play_states(n_plies: int, seed: int) -> list[object]:
    from simult_chess.agents.random_legal import random_legal_program

    rng = random.Random(seed)
    state = standard_starting_state()
    states = [state]
    for _ in range(n_plies):
        pw = random_legal_program(state, Color.WHITE, RULESET, rng)
        pb = random_legal_program(state, Color.BLACK, RULESET, rng)
        result = phi(state, pw, pb, RULESET)
        if result.outcome != "ongoing":
            break
        state = result.state
        states.append(state)
    return states


def _codec_signatures(state: object, color: Color) -> set[tuple[int, int]]:
    """Grid signatures ``(slot1_index, slot2_index)`` the codec can reach.
    Index-based, so the Cancel collapse (D4) is transparent."""
    signatures: set[tuple[int, int]] = set()
    slot1 = slot1_legal_actions(state, color, RULESET)  # type: ignore[arg-type]
    for index1, first in slot1.items():
        slot2, single = slot2_legal_actions(state, color, RULESET, first)  # type: ignore[arg-type]
        if single:
            signatures.add((index1, NO_SECOND_INDEX))
        for index2 in slot2:
            signatures.add((index1, index2))
    return signatures


def test_masking_is_sound_every_yielded_program_is_legal() -> None:
    for state in _self_play_states(25, seed=11):
        for color in (Color.WHITE, Color.BLACK):
            slot1 = slot1_legal_actions(state, color, RULESET)  # type: ignore[arg-type]
            for first in slot1.values():
                slot2, single = slot2_legal_actions(state, color, RULESET, first)  # type: ignore[arg-type]
                if single:
                    assert is_legal_program(state, (first,), color, RULESET)  # type: ignore[arg-type]
                for second in slot2.values():
                    assert is_legal_program(state, (first, second), color, RULESET)  # type: ignore[arg-type]


@pytest.mark.slow
def test_masking_is_complete_against_the_exhaustive_enumeration() -> None:
    pytest.importorskip("pyspiel")
    from simult_chess.interop.openspiel_adapter import enumerate_legal_programs

    for state in _self_play_states(40, seed=5):
        for color in (Color.WHITE, Color.BLACK):
            codec = _codec_signatures(state, color)
            for program in enumerate_legal_programs(state, color, RULESET):  # type: ignore[arg-type]
                index1 = encode_action(program[0], state)  # type: ignore[arg-type]
                index2 = (
                    encode_action(program[1], state)  # type: ignore[arg-type]
                    if len(program) == 2
                    else NO_SECOND_INDEX
                )
                assert (index1, index2) in codec, (
                    f"codec misses enumerated program {program!r} "
                    f"-> signature {(index1, index2)}"
                )


def test_masking_reaches_an_aggressive_dual_the_enumeration_misses() -> None:
    # Bishop c1 cannot guard the rook at a1 (same rank), but CAN guard it at a3
    # (diagonal c1-b2-a3). So Reserve(bishop, rook) is admissible only against
    # the rook's destination -- an aggressive dual (spec §6.2) absent from
    # reserve_candidates and hence from the exhaustive enumeration.
    white_king = Token(id=100, color=Color.WHITE, typ="k")
    black_king = Token(id=200, color=Color.BLACK, typ="k")
    rook = Token(id=1, color=Color.WHITE, typ="r")
    bishop = Token(id=2, color=Color.WHITE, typ="b")
    state = build_state(
        {
            white_king: Square(4, 0),
            black_king: Square(4, 7),
            rook: Square(0, 0),
            bishop: Square(2, 0),
        }
    )
    rook_move = Move(
        token=rook,
        trajectory=Trajectory(path=(Square(0, 0), Square(0, 1), Square(0, 2))),
    )
    dual_reserve = Reserve(defender=bishop, protege=rook)
    # It is a genuinely legal program...
    assert is_legal_program(state, (rook_move, dual_reserve), Color.WHITE, RULESET)

    # ...reachable through the codec: the reserve appears in slot-2 given the
    # rook move as slot-1.
    slot2, _ = slot2_legal_actions(state, Color.WHITE, RULESET, rook_move)
    assert encode_action(dual_reserve, state) in slot2
    assert slot2[encode_action(dual_reserve, state)] == dual_reserve

    # ...but the exhaustive enumeration misses it (reserve_candidates keys on
    # the current square, where the bishop cannot guard the rook).
    pytest.importorskip("pyspiel")
    from simult_chess.interop.openspiel_adapter import enumerate_legal_programs

    assert (rook_move, dual_reserve) not in enumerate_legal_programs(
        state, Color.WHITE, RULESET
    )


def test_slot1_cancel_collapse_decodes_to_the_oldest_reservation() -> None:
    # Two reservations defend the same protege (id=1). Their Cancels share a
    # grid index; slot-1 stores the OLDEST (R-multi-in oldest-valid, spec §6.4).
    white_king = Token(id=100, color=Color.WHITE, typ="k")
    black_king = Token(id=200, color=Color.BLACK, typ="k")
    protege = Token(id=1, color=Color.WHITE, typ="n")
    rook_def = Token(id=2, color=Color.WHITE, typ="r")
    queen_def = Token(id=3, color=Color.WHITE, typ="q")
    oldest = Reservation(defender=rook_def, protege=protege, age=(0, 0))
    newer = Reservation(defender=queen_def, protege=protege, age=(5, 0))
    state = build_state(
        {
            white_king: Square(4, 0),
            black_king: Square(4, 7),
            protege: Square(3, 3),
            rook_def: Square(3, 0),
            queen_def: Square(0, 3),
        },
        reservations_white=(newer, oldest),  # deliberately not age-ordered
    )
    slot1 = slot1_legal_actions(state, Color.WHITE, RULESET)
    cancel_index = encode_action(Cancel(reservation=oldest), state)
    assert cancel_index in slot1
    decoded = slot1[cancel_index]
    assert isinstance(decoded, Cancel)
    assert decoded.reservation is oldest


def test_single_action_legality_flag() -> None:
    # A Move alone is a legal single-action program; a Reserve alone is not,
    # while a legal displacement exists (L2).
    white_king = Token(id=100, color=Color.WHITE, typ="k")
    black_king = Token(id=200, color=Color.BLACK, typ="k")
    defender = Token(id=1, color=Color.WHITE, typ="q")
    protege = Token(id=2, color=Color.WHITE, typ="p")
    state = build_state(
        {
            white_king: Square(4, 0),
            black_king: Square(4, 7),
            defender: Square(3, 3),
            protege: Square(3, 4),
        }
    )
    king_move = Move(
        token=white_king, trajectory=Trajectory(path=(Square(4, 0), Square(5, 0)))
    )
    _, king_single = slot2_legal_actions(state, Color.WHITE, RULESET, king_move)
    assert king_single is True

    reserve = Reserve(defender=defender, protege=protege)
    assert reserve in slot1_legal_actions(state, Color.WHITE, RULESET).values()
    _, reserve_single = slot2_legal_actions(state, Color.WHITE, RULESET, reserve)
    assert reserve_single is False
