"""v3 18a'.5 ("analogous symmetrisation of the two policy heads"):
`action_grid.mirror_grid_index`/`MIRROR_PERMUTATION` are a pure, colour-
independent function of the grid index alone (`encode_action` never reads a
colour, only token *squares*), built from `core.collision.mirror_square`'s
own rank-flip transform applied to each of the four action-kind sub-blocks.

This is exactly the kind of small, closed-form, easy-to-get-subtly-wrong
permutation that needs checking against the *real* mirror machinery, not
just internal algebraic self-consistency: `core.collision.mirror_action`/
`mirror_state`, already used by the M3/M5 property tests, are the ground
truth `net.py`'s architectural symmetry claim depends on.
"""

from __future__ import annotations

import random

from simult_chess.agents.random_legal import random_legal_program
from simult_chess.core.collision import mirror_action, mirror_state
from simult_chess.core.phi import phi
from simult_chess.core.types import Castle, Color
from simult_chess.learn.action_grid import (
    CASTLE_OFFSET,
    MIRROR_PERMUTATION,
    MIRROR_PERMUTATION_WITH_NO_SECOND,
    NO_SECOND_INDEX,
    SLOT_SIZE,
    encode_action,
    mirror_grid_index,
    slot1_legal_actions,
    slot2_legal_actions,
)
from simult_chess.referee.setup import standard_starting_state
from simult_chess.rules.ruleset import RuleSet


def test_mirror_grid_index_is_an_involution_over_the_whole_grid() -> None:
    for index in range(SLOT_SIZE):
        assert mirror_grid_index(mirror_grid_index(index)) == index


def test_mirror_permutation_matches_mirror_grid_index() -> None:
    assert MIRROR_PERMUTATION == tuple(mirror_grid_index(i) for i in range(SLOT_SIZE))
    assert MIRROR_PERMUTATION_WITH_NO_SECOND == (*MIRROR_PERMUTATION, NO_SECOND_INDEX)


def test_castle_entries_are_fixed_points() -> None:
    # Files (kingside/queenside) are preserved by a rank-only flip.
    for side in ("king", "queen"):
        action = Castle(side=side)
        index = CASTLE_OFFSET + (0 if side == "king" else 1)
        mirrored_action = mirror_action(action)
        # encode_action never reads state for a Castle.
        assert encode_action(mirrored_action, standard_starting_state()) == index
        assert mirror_grid_index(index) == index


def test_mirror_grid_index_matches_the_real_mirror_action_and_mirror_state() -> None:
    """Cross-validates `mirror_grid_index` against the project's own
    ground-truth χ machinery (`mirror_action`/`mirror_state`), over a real
    self-play rollout (`n_actions=2` so Reserve/Cancel actually appear, not
    just Move) -- covers all four action kinds, thousands of legal actions
    across evolving positions, both slot-1 and slot-2 indices."""
    ruleset = RuleSet(n_actions=2)
    state = standard_starting_state()
    rng_white = random.Random(1)
    rng_black = random.Random(2)
    kinds_seen: set[str] = set()
    checked = 0

    def check(state: object, color: Color, index: int, action: object) -> None:
        nonlocal checked
        kinds_seen.add(type(action).__name__)
        mirrored_state = mirror_state(state)  # type: ignore[arg-type]
        mirrored_action = mirror_action(action)  # type: ignore[arg-type]
        expected = encode_action(mirrored_action, mirrored_state)
        assert mirror_grid_index(index) == expected, (color, action, index)
        checked += 1

    for _ in range(12):
        for color in (Color.WHITE, Color.BLACK):
            first_actions = slot1_legal_actions(state, color, ruleset)
            for index, action in first_actions.items():
                check(state, color, index, action)
                slot2, _ = slot2_legal_actions(state, color, ruleset, action)
                for index2, second in slot2.items():
                    check(state, color, index2, second)
        program_white = random_legal_program(state, Color.WHITE, ruleset, rng_white)
        program_black = random_legal_program(state, Color.BLACK, ruleset, rng_black)
        result = phi(state, program_white, program_black, ruleset)
        if result.outcome != "ongoing":
            break
        state = result.state

    # Sanity: the rollout actually exercised Reserve and Cancel, not just
    # Move -- otherwise this test would only be re-proving the trivial
    # standard-opening case.
    assert {"Move", "Reserve", "Cancel"} <= kinds_seen
    assert checked > 1000
