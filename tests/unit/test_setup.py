from __future__ import annotations

from simult_chess.core.types import Color
from simult_chess.invariants.checks import check_all_state
from simult_chess.referee.setup import standard_starting_state
from simult_chess.rules.ruleset import RuleSet

RULESET = RuleSet()


def test_standard_starting_state_has_32_pieces() -> None:
    state = standard_starting_state()
    assert len(state.board) == 32


def test_standard_starting_state_passes_all_wf_checks() -> None:
    state = standard_starting_state()
    assert check_all_state(state, RULESET) == []


def test_standard_starting_state_is_symmetric_16_per_color() -> None:
    state = standard_starting_state()
    white = [t for t in state.board if t.color is Color.WHITE]
    black = [t for t in state.board if t.color is Color.BLACK]
    assert len(white) == 16
    assert len(black) == 16
    assert {t.typ for t in white} == {t.typ for t in black}
