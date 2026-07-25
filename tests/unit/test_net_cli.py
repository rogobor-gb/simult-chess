"""Phase 15a net CLI: the human decider's control-word parsing and the parser."""

from __future__ import annotations

import random
from collections.abc import Iterator

import pytest

from simult_chess.core.types import Color
from simult_chess.net import cli
from simult_chess.net.session import (
    AbortDecision,
    AcceptDrawDecision,
    PlayDecision,
    ResignDecision,
)
from simult_chess.referee.setup import standard_starting_state
from simult_chess.rules.ruleset import RuleSet

RULESET = RuleSet()


def _decide(
    monkeypatch: pytest.MonkeyPatch, lines: list[str], *, offered: bool
) -> object:
    supply: Iterator[str] = iter(lines)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(supply))
    monkeypatch.setattr("builtins.print", lambda *_a, **_k: None)
    return cli._human_decider(
        standard_starting_state(),
        Color.WHITE,
        RULESET,
        random.Random(0),
        peer_offered_draw=offered,
    )


def test_resign_command(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(_decide(monkeypatch, ["resign"], offered=False), ResignDecision)


def test_abort_command(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(_decide(monkeypatch, ["abort"], offered=False), AbortDecision)


def test_accept_only_when_a_draw_stands(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(
        _decide(monkeypatch, ["accept"], offered=True), AcceptDrawDecision
    )
    # With no standing offer, 'accept' is rejected and the player is re-prompted.
    decision = _decide(monkeypatch, ["accept", "e4"], offered=False)
    assert isinstance(decision, PlayDecision)
    assert not decision.offer_draw


def test_plain_program_has_no_offer(monkeypatch: pytest.MonkeyPatch) -> None:
    decision = _decide(monkeypatch, ["e4"], offered=False)
    assert isinstance(decision, PlayDecision)
    assert not decision.offer_draw


def test_draw_prefix_attaches_an_offer(monkeypatch: pytest.MonkeyPatch) -> None:
    decision = _decide(monkeypatch, ["draw e4"], offered=False)
    assert isinstance(decision, PlayDecision)
    assert decision.offer_draw


def test_reprompts_on_an_illegal_program(monkeypatch: pytest.MonkeyPatch) -> None:
    # A wildly illegal first line, then a legal one.
    decision = _decide(monkeypatch, ["zzz", "e4"], offered=False)
    assert isinstance(decision, PlayDecision)


def test_parser_defaults() -> None:
    args = cli._build_parser().parse_args(
        ["host", "--port", "5000", "--color", "white"]
    )
    assert args.color == "white"
    assert args.agent == "human"
    assert args.transport_timeout == 15.0
    assert args.keepalive_interval == 5.0
    assert args.liveness_deadline == 20.0
    assert args.max_phases == 500
    assert args.time_control is None


def test_parser_accepts_a_time_control() -> None:
    args = cli._build_parser().parse_args(
        ["host", "--port", "5000", "--color", "white", "--time-control", "3|0|2"]
    )
    assert args.time_control == "3|0|2"
