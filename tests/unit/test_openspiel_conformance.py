"""Conformance test (Phase 12 DoD, docs/DEVELOPMENT_addendum_v1.1.md):
seeded random co-play driven through the OpenSpiel adapter and through the
native referee from identical seeds produces bit-identical state
trajectories.

Marked slow: >=10^2 games, each re-enumerating legal programs for both
colors every phase (a few hundred to ~1600 candidates at typical
positions, empirically measured while sizing _MAX_DISTINCT_ACTIONS in
openspiel_adapter.py).
"""

from __future__ import annotations

import random

import pytest

pytest.importorskip("pyspiel")
pytest.importorskip("numpy")

import pyspiel  # noqa: E402

from simult_chess.core.phi import phi  # noqa: E402
from simult_chess.core.types import Color  # noqa: E402
from simult_chess.interop import openspiel_adapter  # noqa: E402, F401
from simult_chess.referee.setup import standard_starting_state  # noqa: E402
from simult_chess.rules.ruleset import RuleSet  # noqa: E402

RULESET = RuleSet()


def _play_one_conformance_game(seed: int, max_phases: int) -> None:
    """Drive the adapter and the native referee in lockstep from the same
    seed, asserting equal states after *every* phase, not just at the end."""
    rng = random.Random(seed)
    game = pyspiel.load_game("simult_chess")
    adapter_state = game.new_initial_state()
    native_state = standard_starting_state()

    for phase in range(max_phases):
        if adapter_state.is_terminal():
            break

        legal_white = adapter_state.legal_actions(0)
        legal_black = adapter_state.legal_actions(1)
        action_white = rng.choice(legal_white)
        action_black = rng.choice(legal_black)

        # Translate the chosen pyspiel action integers back to native
        # Programs via the adapter's own (already-cached, so this is free)
        # enumeration -- the same program-indexing scheme _apply_actions
        # itself uses.
        # noqa comments: intentional use of a "private" method to access
        # the cache without a second, redundant O(k^2) enumeration.
        program_white = adapter_state._legal_programs(Color.WHITE)[action_white]  # noqa: SLF001
        program_black = adapter_state._legal_programs(Color.BLACK)[action_black]  # noqa: SLF001

        result = phi(native_state, program_white, program_black, RULESET)
        adapter_state.apply_actions([action_white, action_black])
        native_state = result.state

        assert adapter_state.state == native_state, (
            f"seed {seed} phase {phase}: adapter/native states diverged"
        )
        assert adapter_state.is_terminal() == (result.outcome != "ongoing")

    if adapter_state.is_terminal():
        expected_returns = {
            "white_wins": [1.0, -1.0],
            "black_wins": [-1.0, 1.0],
            "draw": [0.0, 0.0],
        }
        # result is defined here since is_terminal() can only be True after
        # at least one applied phase in this loop.
        assert adapter_state.returns() == expected_returns[result.outcome]


@pytest.mark.slow
def test_adapter_matches_native_referee_over_100_seeded_games() -> None:
    for seed in range(100):
        _play_one_conformance_game(seed, max_phases=60)
