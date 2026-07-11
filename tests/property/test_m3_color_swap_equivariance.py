from __future__ import annotations

from conftest import legal_scenarios
from hypothesis import given, settings

from simult_chess.core.collision import mirror_program, mirror_state
from simult_chess.core.phi import phi
from simult_chess.core.types import State
from simult_chess.rules.ruleset import RuleSet

RULESET = RuleSet()

_MIRRORED_OUTCOME = {
    "white_wins": "black_wins",
    "black_wins": "white_wins",
    "draw": "draw",
    "ongoing": "ongoing",
}


def _equal_ignoring_repetition_ledger(a: State, b: State) -> bool:
    """Compare everything except the repetition ledger.

    χ does not remap the ledger's public-position keys (spec/INVARIANTS.md
    §1 documents this as passed through unchanged), so two independently
    resolved mirrored games generally end up with structurally-equivalent
    but not literally-equal ledger entries (different key values for "the
    same" position under symmetry). Every other field is directly comparable.
    """
    return (
        a.board == b.board
        and a.cooldown == b.cooldown
        and a.reservations_white == b.reservations_white
        and a.reservations_black == b.reservations_black
        and a.bookkeeping.castling_rights == b.bookkeeping.castling_rights
        and a.bookkeeping.no_progress_counter == b.bookkeeping.no_progress_counter
        and a.bookkeeping.phase_index == b.bookkeeping.phase_index
    )


@given(legal_scenarios())
@settings(max_examples=200)
def test_m3_color_swap_equivariance(scenario: object) -> None:
    """M3 — χ(Φ(s,π_W,π_B)) = Φ(χ(s), χ(π_B), χ(π_W)); the program swap included.

    The single most important non-trivial test in the suite (spec §0):
    priority (§6.3) and defensive precedence (§6.4) confer no first-mover
    advantage. Any failure here is S0 and should be treated as a halt-worthy
    soundness bug, not a routine regression.
    """
    state, program_white, program_black = scenario  # type: ignore[misc]

    direct = phi(state, program_white, program_black, RULESET)

    mirrored_state = mirror_state(state)
    mirrored_white_program = mirror_program(program_black)
    mirrored_black_program = mirror_program(program_white)
    mirrored = phi(
        mirrored_state, mirrored_white_program, mirrored_black_program, RULESET
    )

    expected = mirror_state(direct.state)

    assert _equal_ignoring_repetition_ledger(expected, mirrored.state)
    assert _MIRRORED_OUTCOME[direct.outcome] == mirrored.outcome
