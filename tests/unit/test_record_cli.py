"""Phase 15d: the replay / to-fixture CLI."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from simult_chess.agents.random_legal import random_legal_program
from simult_chess.referee import record_cli
from simult_chess.referee.match import play_match
from simult_chess.referee.record import GamePhase, build_record, write_record
from simult_chess.referee.setup import standard_starting_state
from simult_chess.rules.ruleset import RuleSet


def _write_game(path: Path, seed: int = 1) -> None:
    ruleset = RuleSet()
    initial = standard_starting_state()
    result = play_match(
        initial,
        random_legal_program,
        random_legal_program,
        ruleset,
        rng_white=random.Random(seed),
        rng_black=random.Random(seed + 1),
    )
    phases = tuple(
        GamePhase(p.program_white, p.program_black, p.outcome) for p in result.phases
    )
    record = build_record(
        initial_state=initial,
        ruleset=ruleset,
        phases=phases,
        final_state=result.final_state,
        raw_outcome=result.outcome,
    )
    path.write_text(write_record(record), encoding="utf-8")


def test_replay_ok(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    scn = tmp_path / "game.scn"
    _write_game(scn)
    assert record_cli.main(["replay", str(scn)]) == 0
    assert "replay ok" in capsys.readouterr().out


def test_replay_refuses_a_variant_mismatch(tmp_path: Path) -> None:
    scn = tmp_path / "game.scn"
    _write_game(scn)
    assert record_cli.main(["replay", str(scn), "--expect-variant", "horizon_30"]) == 2


def test_to_fixture_writes_valid_json(tmp_path: Path) -> None:
    scn = tmp_path / "game.scn"
    _write_game(scn)
    out = tmp_path / "phase.json"
    args = ["to-fixture", str(scn), "--phase", "1", "--out", str(out)]
    assert record_cli.main(args) == 0
    fixture = json.loads(out.read_text())
    assert fixture["expected_state_hash"]
    assert len(fixture["programs"]) == 2  # phases 0 and 1
