from __future__ import annotations

import pytest

import simult_chess.ui.session as session_module
from simult_chess.core.types import Color
from simult_chess.ui import cli


def test_build_parser_hotseat_mode() -> None:
    args = cli._build_parser().parse_args(["hotseat"])
    assert args.mode == "hotseat"


def test_build_parser_agent_mode_defaults() -> None:
    args = cli._build_parser().parse_args(["agent"])
    assert args.mode == "agent"
    assert args.human == "white"
    assert args.agent == "random"
    assert args.seed == 0


def test_build_parser_agent_mode_overrides() -> None:
    args = cli._build_parser().parse_args(
        ["agent", "--human", "black", "--agent", "greedy", "--seed", "5"]
    )
    assert args.human == "black"
    assert args.agent == "greedy"
    assert args.seed == 5


def test_build_parser_requires_a_mode() -> None:
    with pytest.raises(SystemExit):
        cli._build_parser().parse_args([])


def test_main_hotseat_dispatches_to_run_hot_seat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_run_hot_seat(state: object, ruleset: object, **_kwargs: object) -> None:
        calls.append("hot_seat")

    monkeypatch.setattr(session_module, "run_hot_seat", fake_run_hot_seat)
    assert cli.main(["hotseat"]) == 0
    assert calls == ["hot_seat"]


def test_main_agent_mode_dispatches_with_chosen_color_and_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_human_vs_agent(
        state: object,
        ruleset: object,
        human_color: object,
        agent: object,
        rng: object,
        **_kwargs: object,
    ) -> None:
        captured["human_color"] = human_color
        captured["agent"] = agent

    monkeypatch.setattr(session_module, "run_human_vs_agent", fake_run_human_vs_agent)
    assert cli.main(["agent", "--human", "black", "--agent", "greedy"]) == 0
    assert captured["human_color"] is Color.BLACK
