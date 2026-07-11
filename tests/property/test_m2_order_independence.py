from __future__ import annotations

from conftest import legal_scenarios
from hypothesis import given, settings

from simult_chess.core.collision import conflicts
from simult_chess.core.moves import extract_declared_moves
from simult_chess.core.stages.annihilate import resolve_annihilation
from simult_chess.core.stages.defense import resolve_defense
from simult_chess.core.stages.fizzle import resolve_fizzles
from simult_chess.core.types import Color
from simult_chess.rules.ruleset import RuleSet

RULESET = RuleSet()


@given(legal_scenarios())
@settings(max_examples=200)
def test_m2_fizzle_order_independence(scenario: object) -> None:
    """M2(a) — Stage F does not depend on backward-induction order (Lemma 6.2)."""
    state, program_white, program_black = scenario  # type: ignore[misc]
    declared = extract_declared_moves(state, program_white, program_black)

    baseline = resolve_fizzles(declared, state, RULESET)
    reordered = resolve_fizzles(
        declared, state, RULESET, tie_break=tuple(reversed(declared))
    )

    assert baseline.fizzled == reordered.fizzled


@given(legal_scenarios())
@settings(max_examples=200)
def test_m2_annihilation_order_independence(scenario: object) -> None:
    """M2(b) — equal-rank annihilation edges commute (Lemma 6.3a)."""
    state, program_white, program_black = scenario  # type: ignore[misc]
    declared = extract_declared_moves(state, program_white, program_black)
    fizzle_result = resolve_fizzles(declared, state, RULESET)
    executing = tuple(m for m in declared if fizzle_result.executes(m))

    white_moves = [m for m in executing if m.color is Color.WHITE]
    black_moves = [m for m in executing if m.color is Color.BLACK]
    edges = [
        (w, b)
        for w in white_moves
        for b in black_moves
        if conflicts(w.trajectory, b.trajectory)
    ]

    baseline = resolve_annihilation(executing, RULESET)
    reordered = resolve_annihilation(
        executing, RULESET, tie_break=tuple(reversed(edges))
    )

    assert baseline.annihilated == reordered.annihilated


@given(legal_scenarios())
@settings(max_examples=200)
def test_m2_defense_order_independence(scenario: object) -> None:
    """M2(c) — any topological order of the precedence DAG agrees (Lemma 6.4a)."""
    state, program_white, program_black = scenario  # type: ignore[misc]
    declared = extract_declared_moves(state, program_white, program_black)
    fizzle_result = resolve_fizzles(declared, state, RULESET)
    executing = tuple(m for m in declared if fizzle_result.executes(m))
    annihilation_result = resolve_annihilation(executing, RULESET)
    survivors = tuple(m for m in executing if annihilation_result.survives(m))

    pending_squares = [m.trajectory.destination for m in survivors]

    baseline = resolve_defense(executing, survivors, state, (), (), RULESET)
    reordered = resolve_defense(
        executing,
        survivors,
        state,
        (),
        (),
        RULESET,
        tie_break=tuple(reversed(pending_squares)),
    )

    # `captured`/`fired` are ordered event *logs* — their sequence legitimately
    # depends on processing order. M2's actual claim is about the result (who
    # ends up captured/fired and where things land), not log order.
    assert set(baseline.captured) == set(reordered.captured)
    assert set(baseline.fired) == set(reordered.fired)
    assert baseline.occupancy == reordered.occupancy
