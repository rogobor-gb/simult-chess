from __future__ import annotations

import pytest

import simult_chess.ui.session as session_module
from simult_chess.core.types import Color
from simult_chess.rules.variants import BASELINE_NAME, get_variant, variant_names
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
    assert args.variant == BASELINE_NAME


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


def test_build_parser_window_mode_defaults() -> None:
    args = cli._build_parser().parse_args(["window"])
    assert args.mode == "window"
    assert args.human == "white"
    assert args.agent == "greedy"  # the window defaults to a real opponent
    assert args.variant == BASELINE_NAME


def test_main_window_mode_launches_with_chosen_color_and_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    import simult_chess.ui.window as window_module

    captured: dict[str, object] = {}

    class FakeWindow:
        def __init__(
            self, state: object, ruleset: object, *,
            human_color: object, agent: object, seed: int,
        ) -> None:
            captured["human_color"] = human_color
            captured["agent"] = agent
            self.root = types.SimpleNamespace(
                mainloop=lambda: captured.__setitem__("looped", True)
            )

    monkeypatch.setattr(window_module, "SimultChessWindow", FakeWindow)
    assert cli.main(["window", "--human", "black", "--agent", "greedy"]) == 0
    assert captured["human_color"] is Color.BLACK
    assert captured["looped"] is True


def test_variants_subcommand_lists_every_registered_variant(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["variants"]) == 0
    listing = capsys.readouterr().out
    for name in variant_names():
        assert name in listing


def test_variant_flag_selects_the_named_ruleset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gate 14: a declined arm value is *playable*, not merely registered."""
    captured: dict[str, object] = {}

    def fake_run_hot_seat(state: object, ruleset: object, **_kwargs: object) -> None:
        captured["ruleset"] = ruleset

    monkeypatch.setattr(session_module, "run_hot_seat", fake_run_hot_seat)
    assert cli.main(["hotseat", "--variant", "any_same_square_fizzle"]) == 0
    assert captured["ruleset"] == get_variant("any_same_square_fizzle")
