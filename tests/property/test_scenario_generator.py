from __future__ import annotations

from conftest import legal_scenarios
from hypothesis import given, settings

from simult_chess.core import legality
from simult_chess.core.types import Color
from simult_chess.rules.ruleset import RuleSet

RULESET = RuleSet()


@given(legal_scenarios())
@settings(max_examples=100)
def test_generated_scenarios_are_legal(scenario: object) -> None:
    state, program_white, program_black = scenario  # type: ignore[misc]
    assert legality.is_legal_program(state, program_white, Color.WHITE, RULESET)
    assert legality.is_legal_program(state, program_black, Color.BLACK, RULESET)
