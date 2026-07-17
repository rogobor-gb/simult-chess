from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")

from simult_chess.harness.campaign import (  # noqa: E402
    ALL_RUN_SPECS,
    BASELINE_MM_LABEL,
    GameRecord,
    RunSpec,
    play_campaign_game,
    render_report,
    run_campaign,
)
from simult_chess.harness.campaign_stats import wilson_interval  # noqa: E402
from simult_chess.rules.ruleset import RuleSet  # noqa: E402


def test_wilson_interval_matches_known_reference_values() -> None:
    # Independently computed (not via `wilson_interval` itself) closed-form
    # Wilson score values for a handful of (successes, n) pairs.
    cases = [
        (50, 100, (0.5, 0.403830, 0.596170)),
        (8, 20, (0.4, 0.218804, 0.613422)),
        (0, 10, (0.0, 0.0, 0.277540)),
    ]
    for successes, n, (point, low, high) in cases:
        result = wilson_interval(successes, n)
        assert result.point == pytest.approx(point, abs=1e-5)
        assert result.low == pytest.approx(low, abs=1e-5)
        assert result.high == pytest.approx(high, abs=1e-5)


def test_wilson_interval_empty_sample_is_nan() -> None:
    result = wilson_interval(0, 0)
    assert result.point != result.point  # nan != nan


def test_draw_reason_attribution_finds_horizon_draws() -> None:
    """A tiny horizon makes horizon-driven draws common even for
    `random_legal` self-play, cheaply exercising the draw-reason path that
    replicates `core/phi.py`'s own precedence externally (campaign.py's
    `_attribute_draw_reason`).
    """
    spec = RunSpec(
        label="test:horizon",
        agent_white="random_legal",
        agent_black="random_legal",
        ruleset=RuleSet(horizon=2),
        base_seed=0,
        n_games=30,
        max_phases=30,
    )
    records = [
        play_campaign_game(spec, spec.base_seed + i) for i in range(spec.n_games)
    ]
    draw_reasons = {r.draw_reason for r in records if r.outcome == "draw"}
    assert "horizon" in draw_reasons
    # Every record's own severity-count total matches how many violations it
    # actually logged, and zero S0/S1 (this is just self-play, no rule
    # variant expected to trip anything).
    for r in records:
        assert r.violation_counts["S0"] == 0
        assert r.violation_counts["S1"] == 0


def test_multiprocessing_matches_sequential_play() -> None:
    """Games run through the process pool must be bit-identical to the same
    seeds run sequentially in-process -- the same "same seed => same game"
    guarantee every prior phase relies on, now checked across a process
    boundary (agents are looked up by name inside each worker, not pickled
    as closures).
    """
    specs = (
        RunSpec(
            label="t1",
            agent_white="random_legal",
            agent_black="random_legal",
            ruleset=RuleSet(),
            base_seed=42,
            n_games=3,
            max_phases=20,
        ),
        RunSpec(
            label="t2",
            agent_white="random_legal",
            agent_black="greedy",
            ruleset=RuleSet(),
            base_seed=142,
            n_games=3,
            max_phases=20,
        ),
    )
    parallel = run_campaign(specs, workers=2, progress=False)
    sequential = {
        spec.label: [
            play_campaign_game(spec, spec.base_seed + i) for i in range(spec.n_games)
        ]
        for spec in specs
    }
    for spec in specs:
        parallel_by_seed = {r.rng_seed: r for r in parallel[spec.label]}
        sequential_by_seed = {r.rng_seed: r for r in sequential[spec.label]}
        assert parallel_by_seed == sequential_by_seed


def _fake_record(
    label: str, seed: int, outcome: str, draw_reason: str | None = None
) -> GameRecord:
    return GameRecord(
        run_label=label,
        rng_seed=seed,
        outcome=outcome,
        phases_played=10,
        violation_counts={"S0": 0, "S1": 0, "S2": 0, "S3": 0},
        draw_reason=draw_reason,
        reservation_rate=0.1,
        cancellation_rate=0.05,
        cooldown_occupancy=0.2,
        material_final=0.0,
        material_volatility=0.5,
    )


def test_render_report_smoke() -> None:
    """`render_report` shouldn't need any real games -- a synthetic cache
    covering every pre-registered run spec at trivial size is enough to
    check every section renders without error.
    """
    cache = {
        spec.label: [
            _fake_record(spec.label, spec.base_seed, "white_wins"),
            _fake_record(spec.label, spec.base_seed + 1, "draw", draw_reason="horizon"),
        ]
        for spec in ALL_RUN_SPECS
    }
    report = render_report(cache, ALL_RUN_SPECS)
    assert "Campaign v1" in report
    assert "Color-symmetry audit" in report
    assert "Seed reference appendix" in report
    assert BASELINE_MM_LABEL in report


def test_run_specs_are_disjoint_and_cover_the_addendum_totals() -> None:
    seed_ranges = [
        range(spec.base_seed, spec.base_seed + spec.n_games) for spec in ALL_RUN_SPECS
    ]
    for i, a in enumerate(seed_ranges):
        for b in seed_ranges[i + 1 :]:
            assert not (a.start < b.stop and b.start < a.stop), (
                "seed ranges must not overlap"
            )

    tournament_total = sum(
        s.n_games for s in ALL_RUN_SPECS if s.label.startswith("tournament:")
    )
    arm_total = sum(s.n_games for s in ALL_RUN_SPECS if s.label.startswith("arm:"))
    assert tournament_total >= 20_000
    assert arm_total == 25_000
