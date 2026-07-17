from __future__ import annotations

import pytest

pytest.importorskip("pyspiel")
pytest.importorskip("numpy")

import pyspiel  # noqa: E402

from simult_chess.core import legality  # noqa: E402
from simult_chess.core.types import Color  # noqa: E402
from simult_chess.interop import (
    openspiel_adapter,  # noqa: E402, F401 (registers the game)
)
from simult_chess.interop.openspiel_adapter import (
    enumerate_legal_programs,  # noqa: E402
)
from simult_chess.referee.setup import standard_starting_state  # noqa: E402
from simult_chess.rules.ruleset import RuleSet  # noqa: E402

RULESET = RuleSet()


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
    game = pyspiel.load_game("simult_chess")
    state = game.new_initial_state()
    observer = game.make_py_observer()
    observer.set_from(state, 0)
    assert observer.tensor.shape == (17 * 8 * 8 + 7,)
    assert observer.dict["planes"].shape == (17, 8, 8)
    assert observer.dict["scalars"].shape == (7,)
    # Standard start: all four castling rights true, no-progress 0, phase
    # parity 0, horizon = RuleSet default 50.
    assert list(observer.dict["scalars"]) == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 50.0]
    # 16 white + 16 black pieces on the board -> 32 marked squares total
    # across the 12 (color, type) planes.
    assert observer.dict["planes"][:12].sum() == 32


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
