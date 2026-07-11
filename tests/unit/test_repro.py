from __future__ import annotations

from conftest import build_state

from simult_chess.core.types import Color, Move, Square, Token, Trajectory
from simult_chess.core.violation import Violation
from simult_chess.invariants.repro import build_repro_dump, replay
from simult_chess.rules.ruleset import RuleSet

RULESET = RuleSet()


def test_build_repro_dump_captures_phase_index_from_state() -> None:
    king = Token(id=1, color=Color.WHITE, typ="k")
    state = build_state({king: Square(4, 0)}, phase_index=7)
    violation = Violation("R1", "something went wrong")

    dump = build_repro_dump(violation, "TRACE", state, (), (), RULESET, rng_seed=42)

    assert dump.phase_index == 7
    assert dump.rng_seed == 42
    assert dump.violation is violation
    assert dump.check_point == "TRACE"


def test_repro_dump_to_dict_is_json_shaped() -> None:
    king = Token(id=1, color=Color.WHITE, typ="k")
    state = build_state({king: Square(4, 0)})
    violation = Violation("WF1", "duplicate square")
    dump = build_repro_dump(violation, "STATE", state, (), (), RULESET)

    payload = dump.to_dict()

    assert payload["invariant_id"] == "WF1"
    assert payload["check_point"] == "STATE"
    assert payload["ruleset"]["horizon"] == RULESET.horizon
    assert "state_pre" in payload


def test_replay_is_deterministic() -> None:
    white_king = Token(id=1, color=Color.WHITE, typ="k")
    black_king = Token(id=2, color=Color.BLACK, typ="k")
    pawn = Token(id=3, color=Color.WHITE, typ="p")
    filler = Token(id=4, color=Color.BLACK, typ="p")
    state = build_state(
        {
            white_king: Square(0, 0),
            black_king: Square(7, 7),
            pawn: Square(4, 1),
            filler: Square(0, 6),
        }
    )
    program_white = (
        Move(token=pawn, trajectory=Trajectory(path=(Square(4, 1), Square(4, 2)))),
    )
    program_black = (
        Move(token=filler, trajectory=Trajectory(path=(Square(0, 6), Square(0, 5)))),
    )
    violation = Violation("R1", "placeholder")
    dump = build_repro_dump(
        violation, "TRACE", state, program_white, program_black, RULESET
    )

    first = replay(dump)
    second = replay(dump)

    assert first.state == second.state
    assert first.outcome == second.outcome
