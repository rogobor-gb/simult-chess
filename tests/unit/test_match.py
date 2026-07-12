from __future__ import annotations

import random

from simult_chess.agents.random_legal import random_legal_program
from simult_chess.referee.match import play_match
from simult_chess.referee.setup import standard_starting_state
from simult_chess.rules.ruleset import RuleSet

RULESET = RuleSet()


def test_play_match_terminates_and_records_phases() -> None:
    result = play_match(
        standard_starting_state(),
        random_legal_program,
        random_legal_program,
        RULESET,
        rng_white=random.Random(0),
        rng_black=random.Random(1),
        max_phases=80,
    )

    assert result.outcome != "ongoing"
    assert len(result.phases) >= 1
    assert result.phases[-1].state_after == result.final_state
    # a decisive result means we broke out on that phase's own outcome;
    # otherwise the loop exhausted and MatchResult overrides to the limit
    if result.outcome != "phase_limit_reached":
        assert result.phases[-1].outcome == result.outcome


def test_play_match_is_deterministic_given_the_same_seeds() -> None:
    def run() -> str:
        result = play_match(
            standard_starting_state(),
            random_legal_program,
            random_legal_program,
            RULESET,
            rng_white=random.Random(7),
            rng_black=random.Random(13),
            max_phases=60,
        )
        return str(result.final_state)

    assert run() == run()


def test_play_match_respects_the_phase_limit() -> None:
    result = play_match(
        standard_starting_state(),
        random_legal_program,
        random_legal_program,
        RULESET,
        rng_white=random.Random(0),
        rng_black=random.Random(1),
        max_phases=2,
    )
    assert len(result.phases) <= 2
