"""matrix_1ply vs random_legal mini-match (Phase 10 DoD,
docs/DEVELOPMENT_addendum_v1.1.md): a seeded 500-game sanity check that the
first game-theoretic agent is actually stronger than uniform-random play,
plus the harness sweep's own zero-S0/S1 guarantee over the same games. This
is a strength sanity check, not a balance claim (color is fixed: matrix_1ply
always plays White).
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")

from scipy.stats import binomtest  # noqa: E402

from simult_chess.agents.random_legal import random_legal_program  # noqa: E402
from simult_chess.harness.selfplay import run_sweep  # noqa: E402
from simult_chess.referee.setup import standard_starting_state  # noqa: E402
from simult_chess.rules.ruleset import RuleSet  # noqa: E402
from simult_chess.solver.agent import matrix_1ply  # noqa: E402

RULESET = RuleSet()


@pytest.mark.slow
def test_matrix_1ply_beats_random_legal_in_a_seeded_minimatch() -> None:
    report = run_sweep(
        standard_starting_state,
        matrix_1ply,
        random_legal_program,
        RULESET,
        num_games=500,
        base_seed=20000,
        max_phases=60,
    )

    # Zero S0/S1 across the whole mini-match (this doubles as the harness
    # sweep the DoD asks for).
    assert report.violations_of_severity("S0", "S1") == ()

    outcomes = [game.outcome for game in report.games]
    white_wins = outcomes.count("white_wins")
    black_wins = outcomes.count("black_wins")
    decisive = white_wins + black_wins
    assert decisive > 0

    result = binomtest(white_wins, decisive, p=0.5, alternative="greater")
    assert result.pvalue < 0.01, (
        f"matrix_1ply (White) won {white_wins}/{decisive} decisive games "
        f"({outcomes.count('draw')} draws, "
        f"{outcomes.count('phase_limit_reached')} phase-limit) "
        f"vs random_legal (Black); one-sided binomial p={result.pvalue}"
    )
