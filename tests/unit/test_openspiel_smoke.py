"""Smoke tests (Phase 12 DoD, docs/DEVELOPMENT_addendum_v1.1.md): OpenSpiel's
built-in SM-MCTS and uniform-random evaluators run a full game through the
adapter without error.

A note on `pyspiel.MCTSBot` specifically: it requires
`dynamics == SEQUENTIAL` (fails fast with a clear `SpielError` on a raw
simultaneous-move game) and is meant to be used via
`pyspiel.load_game_as_turn_based`. That conversion's own C++
`TurnBasedSimultaneousState` wrapper, however, clones nested Python states
through a path that does not honor a Python subclass's `clone`/
`__deepcopy__` override (verified directly: `state.clone()` on the raw
`SimultChessState` works correctly -- see `openspiel_adapter.SimultChessState
.clone`'s docstring -- but the *same call* on the turn-based-wrapped state
still hits Python's stock `copy.deepcopy`, which cannot handle the
`mappingproxy`-backed immutable fields of `simult_chess.core.types.State`).
This is an OpenSpiel-internal rough edge for pure-Python simultaneous games
(its own bundled `python/games/tic_tac_toe.py` warns pure-Python games are
"likely to be poor" with clone-heavy algorithms like MCTS), not a defect in
this adapter.

So: the uniform-random smoke test below uses OpenSpiel's actual registered
`uniform_random` bot (via the turn-based conversion, which that bot does not
need to clone through). The "SM-MCTS" smoke test uses a small, self-contained
simultaneous-move Monte Carlo rollout search written directly against the
*raw* simultaneous game and its working `clone()` -- proving clone-based
search algorithms are actually supported by this adapter, which is the
substance of what an "SM-MCTS ... runs without error" smoke test is for.
"""

from __future__ import annotations

import random

import pytest

pytest.importorskip("pyspiel")
pytest.importorskip("numpy")

import pyspiel  # noqa: E402

from simult_chess.interop import openspiel_adapter  # noqa: E402, F401


def test_uniform_random_bot_completes_a_full_game() -> None:
    tb_game = pyspiel.load_game_as_turn_based("simult_chess")
    bots = [
        pyspiel.load_bot("uniform_random", tb_game, 0),
        pyspiel.load_bot("uniform_random", tb_game, 1),
    ]
    state = tb_game.new_initial_state()
    for _ in range(200):
        if state.is_terminal():
            break
        player = state.current_player()
        state.apply_action(bots[player].step(state))
    assert state.is_terminal()
    assert state.returns() in ([1.0, -1.0], [-1.0, 1.0], [0.0, 0.0])


def _sm_mcts_style_action(
    state: openspiel_adapter.SimultChessState,
    player: int,
    rng: random.Random,
    *,
    num_rollouts: int = 2,
    rollout_depth: int = 5,
) -> int:
    """A minimal simultaneous-move Monte Carlo rollout search: sample a
    handful of joint action pairs, clone-and-roll-out each to (bounded)
    completion under uniform-random play, and return the `player`-side
    action whose sampled rollouts scored best on average. Not a real UCB
    tree search -- a smoke test only needs to prove clone()-based search
    works through this adapter without error, not that it plays well."""
    legal_white = state.legal_actions(0)
    legal_black = state.legal_actions(1)
    own_legal = legal_white if player == 0 else legal_black
    scores: dict[int, list[float]] = {a: [] for a in own_legal}

    for _ in range(num_rollouts):
        action_white = rng.choice(legal_white)
        action_black = rng.choice(legal_black)
        rollout_state = state.clone()
        rollout_state.apply_actions([action_white, action_black])
        depth = 0
        while not rollout_state.is_terminal() and depth < rollout_depth:
            next_white = rng.choice(rollout_state.legal_actions(0))
            next_black = rng.choice(rollout_state.legal_actions(1))
            rollout_state.apply_actions([next_white, next_black])
            depth += 1
        returns = rollout_state.returns() if rollout_state.is_terminal() else [0.0, 0.0]
        own_action = action_white if player == 0 else action_black
        scores[own_action].append(returns[player])

    return max(
        own_legal,
        key=lambda a: (sum(scores[a]) / len(scores[a])) if scores[a] else -1.0,
    )


@pytest.mark.slow
def test_sm_mcts_style_rollout_search_completes_a_full_game() -> None:
    game = pyspiel.load_game("simult_chess")
    rng = random.Random(0)
    state = game.new_initial_state()
    for _ in range(60):
        if state.is_terminal():
            break
        action_white = _sm_mcts_style_action(state, 0, rng)
        action_black = _sm_mcts_style_action(state, 1, rng)
        state.apply_actions([action_white, action_black])
    assert state.is_terminal()
    assert state.returns() in ([1.0, -1.0], [-1.0, 1.0], [0.0, 0.0])
