from __future__ import annotations

from conftest import legal_scenarios
from hypothesis import given, settings

from simult_chess.core.phi import phi
from simult_chess.rules.ruleset import RuleSet

RULESET = RuleSet()


@given(legal_scenarios())
@settings(max_examples=200)
def test_m1_purity_repeated_evaluation_is_bit_identical(scenario: object) -> None:
    """M1 — Φ is pure: same inputs yield bit-identical state, outcome, trace."""
    state, program_white, program_black = scenario  # type: ignore[misc]

    first = phi(state, program_white, program_black, RULESET)
    second = phi(state, program_white, program_black, RULESET)

    assert first.state == second.state
    assert first.outcome == second.outcome
    assert first.trace == second.trace
