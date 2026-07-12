from __future__ import annotations

from simult_chess.agents.random_legal import random_legal_program
from simult_chess.harness.selfplay import play_one_game, run_sweep
from simult_chess.referee.setup import standard_starting_state
from simult_chess.rules.ruleset import RuleSet

RULESET = RuleSet()


def test_play_one_game_is_deterministic_given_the_same_seed() -> None:
    def run() -> tuple[str, int]:
        report = play_one_game(
            standard_starting_state(),
            random_legal_program,
            random_legal_program,
            RULESET,
            rng_seed=123,
            max_phases=80,
        )
        return report.outcome, report.phases_played

    assert run() == run()


def test_small_sweep_has_zero_violations() -> None:
    """The dev brief Phase 6 DoD, at a scale that stays fast in CI.

    A full 10^4-game soak test is a manual/nightly job (see
    scripts/README or PROJECT_STATUS.md), not part of the routine suite --
    this locks in the same zero-S0/S1 guarantee at a much smaller scale so
    a regression is still caught quickly.
    """
    report = run_sweep(
        standard_starting_state,
        random_legal_program,
        random_legal_program,
        RULESET,
        num_games=30,
        base_seed=1000,
        max_phases=100,
    )

    assert report.violations_of_severity("S0", "S1") == ()
    assert report.all_violations == ()


def test_sweep_seeds_are_independent_and_reproducible() -> None:
    report_a = run_sweep(
        standard_starting_state,
        random_legal_program,
        random_legal_program,
        RULESET,
        num_games=5,
        base_seed=42,
        max_phases=50,
    )
    report_b = run_sweep(
        standard_starting_state,
        random_legal_program,
        random_legal_program,
        RULESET,
        num_games=5,
        base_seed=42,
        max_phases=50,
    )
    assert [g.outcome for g in report_a.games] == [g.outcome for g in report_b.games]
    assert [g.phases_played for g in report_a.games] == [
        g.phases_played for g in report_b.games
    ]
