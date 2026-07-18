from __future__ import annotations

import pytest

pytest.importorskip("pyspiel")
pytest.importorskip("numpy")

import pyspiel  # noqa: E402

from simult_chess.core import legality  # noqa: E402
from simult_chess.core.types import (  # noqa: E402
    Bookkeeping,
    Cancel,
    CastlingRights,
    Color,
    Move,
    Reservation,
    Square,
    State,
    Token,
)
from simult_chess.interop import (
    openspiel_adapter,  # noqa: E402, F401 (registers the game)
)
from simult_chess.interop.openspiel_adapter import (
    enumerate_legal_programs,  # noqa: E402
)
from simult_chess.referee.setup import standard_starting_state  # noqa: E402
from simult_chess.rules.ruleset import RuleSet  # noqa: E402

RULESET = RuleSet()


def _state_with_a_white_reservation() -> tuple[State, Reservation]:
    """Minimal legal-ish position with one standing white reservation: a
    contact pawn defence (e3-pawn defends d4-pawn), the same shape as the M4
    worked example. Both kings present so `has_any_legal_displacement` is
    true, which is what makes a Cancel-only program L2-illegal (spec §4.4)."""
    white_king = Token(id=100, color=Color.WHITE, typ="k")
    black_king = Token(id=200, color=Color.BLACK, typ="k")
    d4_pawn = Token(id=1, color=Color.WHITE, typ="p")
    e3_pawn = Token(id=2, color=Color.WHITE, typ="p")
    reservation = Reservation(defender=e3_pawn, protege=d4_pawn, age=(0, 0))
    state = State(
        board={
            white_king: Square(0, 0),
            black_king: Square(7, 7),
            d4_pawn: Square(3, 3),
            e3_pawn: Square(4, 2),
        },
        cooldown=frozenset(),
        reservations_white=(reservation,),
        reservations_black=(),
        bookkeeping=Bookkeeping(
            castling_rights=CastlingRights(),
            repetition_ledger={},
            no_progress_counter=0,
            phase_index=0,
        ),
    )
    return state, reservation


def test_game_is_registered_under_its_short_name() -> None:
    assert "simult_chess" in pyspiel.registered_names()
    game = pyspiel.load_game("simult_chess")
    assert game.num_players() == 2


def test_enumerate_legal_programs_matches_legality_predicate() -> None:
    state = standard_starting_state()
    programs = enumerate_legal_programs(state, Color.WHITE, RULESET)
    assert programs
    for program in programs:
        assert legality.is_legal_program(state, program, Color.WHITE, RULESET)


def test_standard_start_emits_no_cancel_actions() -> None:
    # No reservations stand at the opening, so the (D3) Cancel extension adds
    # nothing there -- the enumeration is unchanged for the standard start.
    state = standard_starting_state()
    programs = enumerate_legal_programs(state, Color.WHITE, RULESET)
    assert not any(isinstance(a, Cancel) for program in programs for a in program)


def test_enumerate_emits_cancel_when_a_reservation_stands() -> None:
    # D3 (docs/LEARNING_DESIGN.md): the exhaustive enumeration now emits
    # Cancel, so the learned agent's action space is the full L(s,pi). A
    # Cancel names the standing reservation and (per L2) legalizes only
    # paired with a Move/Castle when a displacement exists.
    state, reservation = _state_with_a_white_reservation()
    programs = enumerate_legal_programs(state, Color.WHITE, RULESET)

    cancels = [p for p in programs if any(isinstance(a, Cancel) for a in p)]
    assert cancels, "no Cancel program emitted though a reservation stands"

    for program in cancels:
        assert legality.is_legal_program(state, program, Color.WHITE, RULESET)
        for action in program:
            if isinstance(action, Cancel):
                assert action.reservation == reservation

    # L2 forbids a Cancel-only program while a legal displacement exists, so
    # every emitted Cancel program pairs the Cancel with a Move/Castle.
    assert all(len(p) == 2 for p in cancels)
    assert any(
        isinstance(p[0], Move) or isinstance(p[1], Move) for p in cancels
    ), "expected at least one Move+Cancel program"


def test_initial_state_is_simultaneous_and_not_terminal() -> None:
    game = pyspiel.load_game("simult_chess")
    state = game.new_initial_state()
    assert state.current_player() == pyspiel.PlayerId.SIMULTANEOUS
    assert not state.is_terminal()


def test_legal_actions_count_matches_native_enumeration() -> None:
    game = pyspiel.load_game("simult_chess")
    state = game.new_initial_state()
    native_white = enumerate_legal_programs(state.state, Color.WHITE, RULESET)
    native_black = enumerate_legal_programs(state.state, Color.BLACK, RULESET)
    assert len(state.legal_actions(0)) == len(native_white)
    assert len(state.legal_actions(1)) == len(native_black)


def test_apply_actions_advances_and_matches_native_phi() -> None:
    from simult_chess.core.phi import phi

    game = pyspiel.load_game("simult_chess")
    state = game.new_initial_state()
    native_state = standard_starting_state()

    programs_white = enumerate_legal_programs(native_state, Color.WHITE, RULESET)
    programs_black = enumerate_legal_programs(native_state, Color.BLACK, RULESET)
    result = phi(native_state, programs_white[0], programs_black[0], RULESET)

    state.apply_actions([0, 0])

    assert state.state == result.state
    assert not state.is_terminal()


def test_observer_tensor_has_the_documented_shape() -> None:
    # 21 planes: 12 board + 1 cooldown + 4 reservation-actor + 4 reservation-
    # pairing (D5, docs/LEARNING_DESIGN.md §3.2).
    game = pyspiel.load_game("simult_chess")
    state = game.new_initial_state()
    observer = game.make_py_observer()
    observer.set_from(state, 0)
    assert observer.tensor.shape == (21 * 8 * 8 + 7,)
    assert observer.dict["planes"].shape == (21, 8, 8)
    assert observer.dict["scalars"].shape == (7,)
    # Standard start: all four castling rights true, no-progress 0, phase
    # parity 0, horizon = RuleSet default 50.
    assert list(observer.dict["scalars"]) == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 50.0]
    # 16 white + 16 black pieces on the board -> 32 marked squares total
    # across the 12 (color, type) planes.
    assert observer.dict["planes"][:12].sum() == 32
    # No reservations stand at the opening, so the 4 reservation-actor planes
    # (13..16) and the 4 pairing planes (17..20) are all zero.
    assert observer.dict["planes"][13:].sum() == 0.0


def test_observer_encodes_the_reservation_pairing_offset() -> None:
    # D5: at the defender's square the pairing planes hold the (Δfile, Δrank)
    # offset to its oldest active protege, normalized to [-1, 1] (÷7).
    from simult_chess.interop.openspiel_adapter import SimultChessGame, SimultChessState

    native_state, _ = _state_with_a_white_reservation()
    game = SimultChessGame()
    state = SimultChessState(game, native_state)
    observer = game.make_py_observer()
    observer.set_from(state, 0)
    planes = observer.dict["planes"]

    # Defender e3-pawn at (file 4, rank 2); protege d4-pawn at (file 3, rank 3).
    # Actor planes: white_defenders=13 at defender square, white_proteges=14.
    assert planes[13, 2, 4] == 1.0
    assert planes[14, 3, 3] == 1.0
    # Pairing planes 17 (Δfile) and 18 (Δrank) keyed at the DEFENDER square.
    assert planes[17, 2, 4] == pytest.approx((3 - 4) / 7.0)
    assert planes[18, 2, 4] == pytest.approx((3 - 2) / 7.0)
    # Nothing written at the protege square in the pairing planes.
    assert planes[17, 3, 3] == 0.0
    assert planes[18, 3, 3] == 0.0
    # Black pairing planes (19, 20) untouched -- no black reservation.
    assert planes[19].sum() == 0.0
    assert planes[20].sum() == 0.0


def test_full_random_game_reaches_a_terminal_outcome() -> None:
    import random

    game = pyspiel.load_game("simult_chess")
    rng = random.Random(0)
    state = game.new_initial_state()
    for _ in range(80):
        if state.is_terminal():
            break
        action_white = rng.choice(state.legal_actions(0))
        action_black = rng.choice(state.legal_actions(1))
        state.apply_actions([action_white, action_black])
    assert state.is_terminal()
    assert state.returns() in ([1.0, -1.0], [-1.0, 1.0], [0.0, 0.0])
