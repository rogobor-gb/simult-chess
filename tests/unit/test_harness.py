from __future__ import annotations

import pytest
from conftest import build_state

from simult_chess.core.stages.annihilate import AnnihilationResult
from simult_chess.core.types import Color, Move, Square, Token, Trajectory
from simult_chess.invariants import harness
from simult_chess.invariants.repro import replay
from simult_chess.rules import registry
from simult_chess.rules.ruleset import RuleSet

RULESET = RuleSet()
WHITE_KING = Token(id=100, color=Color.WHITE, typ="k")
BLACK_KING = Token(id=200, color=Color.BLACK, typ="k")


def _move(token: Token, path: tuple[Square, ...]) -> Move:
    return Move(token=token, trajectory=Trajectory(path=path))


def test_run_phase_strict_mode_clean_program_has_no_violations() -> None:
    pawn = Token(id=1, color=Color.WHITE, typ="p")
    filler = Token(id=2, color=Color.BLACK, typ="p")
    state = build_state(
        {
            WHITE_KING: Square(4, 0),
            BLACK_KING: Square(4, 7),
            pawn: Square(4, 1),
            filler: Square(4, 6),
        }
    )
    program_white = (_move(pawn, (Square(4, 1), Square(4, 2))),)
    program_black = (_move(filler, (Square(4, 6), Square(4, 5))),)

    result = harness.run_phase(
        state, program_white, program_black, RULESET, mode="strict"
    )

    assert result.violations == ()
    assert result.phi_result is not None
    assert result.repro_dumps == ()


def test_run_phase_strict_mode_raises_on_illegal_program_before_phi_runs() -> None:
    state = build_state({WHITE_KING: Square(4, 0), BLACK_KING: Square(4, 7)})
    # empty white program: illegal (L2), since the king has legal moves
    with pytest.raises(harness.InvariantViolationError):
        harness.run_phase(
            state, (), (_move(BLACK_KING, (Square(4, 7), Square(3, 7))),), RULESET
        )


def test_run_phase_lenient_mode_reports_pre_violation_without_raising() -> None:
    state = build_state({WHITE_KING: Square(4, 0), BLACK_KING: Square(4, 7)})

    result = harness.run_phase(
        state,
        (),
        (_move(BLACK_KING, (Square(4, 7), Square(3, 7))),),
        RULESET,
        mode="lenient",
    )

    assert result.phi_result is None
    assert any(v.invariant_id == "L2" for v in result.violations)
    assert len(result.repro_dumps) == len(result.violations)


def test_harness_catches_a_corrupted_annihilation_matcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The DoD case: a deliberately corrupted Φ is caught and produces a
    replayable repro dump; lenient mode aggregates without halting."""

    def broken_matcher(executing, ruleset, *, tie_break=None):  # type: ignore[no-untyped-def]
        return AnnihilationResult(events=())  # never annihilates anything

    monkeypatch.setattr(
        registry, "get_annihilation_matcher", lambda ruleset: broken_matcher
    )

    white_rook = Token(id=1, color=Color.WHITE, typ="r")
    black_rook = Token(id=2, color=Color.BLACK, typ="r")
    state = build_state(
        {
            WHITE_KING: Square(7, 0),
            BLACK_KING: Square(7, 7),
            white_rook: Square(3, 0),
            black_rook: Square(0, 3),
        }
    )
    # a perpendicular crossing at d4: a (V) conflict that must annihilate both
    white_path = tuple(Square(3, r) for r in range(8))
    black_path = tuple(Square(f, 3) for f in range(8))
    program_white = (_move(white_rook, white_path),)
    program_black = (_move(black_rook, black_path),)

    result = harness.run_phase(
        state, program_white, program_black, RULESET, mode="lenient"
    )

    assert result.phi_result is not None  # PRE was fine; corruption is in TRACE
    assert any(v.invariant_id == "R4" for v in result.violations)
    assert len(result.repro_dumps) == len(result.violations)

    dump = next(d for d in result.repro_dumps if d.violation.invariant_id == "R4")
    replayed_once = replay(dump)
    replayed_twice = replay(dump)
    assert replayed_once.state == replayed_twice.state  # deterministic replay
    assert replayed_once.state == result.phi_result.state  # matches the original run

    # strict mode must raise on the same corruption
    with pytest.raises(harness.InvariantViolationError):
        harness.run_phase(state, program_white, program_black, RULESET, mode="strict")
