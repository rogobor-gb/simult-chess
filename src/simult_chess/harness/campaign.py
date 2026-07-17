"""Phase 11b pre-registered empirical campaign (docs/DEVELOPMENT_addendum_
v1.1.md §11b). Estimands, statistics, and every run spec's sample size are
declared here as module-level constants -- before any run, per the
addendum's own design discipline -- not derived from CLI flags or from
results.

    python -m simult_chess.harness.campaign pilot   # validation only, tiny
                                                      # N, disjoint seeds,
                                                      # never touches reports/
    python -m simult_chess.harness.campaign run [--resume]
    python -m simult_chess.harness.campaign report  # renders
                                                      # reports/campaign_v1.md

Interpretive caveat (stated in the report itself, per the maintainer): every
balance statistic here is a functional of the state distributions induced by
*these agents*, not of equilibrium play. The freeze this report supports is
therefore provisional by construction (ruling A5) and is re-estimated after
Phase 13 under learned agents.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from simult_chess.agents.base import Agent
from simult_chess.agents.greedy import greedy_program
from simult_chess.agents.random_legal import random_legal_program
from simult_chess.core.stages.closure import detect_terminal
from simult_chess.core.types import Program, Reserve, State
from simult_chess.harness.campaign_stats import (
    WilsonInterval,
    color_symmetry_test,
    median_iqr,
    minimum_detectable_effect,
    volatility,
    wilson_interval,
)
from simult_chess.harness.selfplay import GameReport, play_one_game
from simult_chess.invariants.harness import HarnessResult
from simult_chess.invariants.severity import severity_of
from simult_chess.referee.serialize import public_position_key
from simult_chess.referee.setup import standard_starting_state
from simult_chess.rules.ruleset import RuleSet
from simult_chess.solver.agent import matrix_1ply
from simult_chess.solver.stage_matrix import material_difference

_REPO_ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN_RUNS_DIR = _REPO_ROOT / "campaign_runs"
REPORTS_DIR = _REPO_ROOT / "reports"
FULL_CACHE_PATH = CAMPAIGN_RUNS_DIR / "full.json"
PILOT_CACHE_PATH = CAMPAIGN_RUNS_DIR / "pilot.json"
REPORT_PATH = REPORTS_DIR / "campaign_v1.md"

AGENT_NAMES: tuple[str, ...] = ("random_legal", "greedy", "matrix_1ply")
AGENT_REGISTRY: dict[str, Agent] = {
    "random_legal": random_legal_program,
    "greedy": greedy_program,
    "matrix_1ply": matrix_1ply,
}

DEFAULT_MAX_PHASES = 300
BASELINE = RuleSet()

# ---------------------------------------------------------------------------
# Pre-registered run specs
# ---------------------------------------------------------------------------

TOURNAMENT_TOTAL_GAMES = 20_000
"""Spread evenly across the 9 ordered {random_legal, greedy, matrix_1ply}^2
pairings on BASELINE -- the addendum's own >= 2x10^4 floor, split so the
matrix is diagonal-symmetric (equal n per ordered pair) as estimand 6 needs.
"""

AB_ARM_DEFS: tuple[tuple[str, RuleSet, int], ...] = (
    ("cancellation_enabled=off", replace(BASELINE, cancellation_enabled=False), 5_000),
    ("intermezzo_reading=i", replace(BASELINE, intermezzo_reading="i"), 5_000),
    (
        "pawn_fizzle=any_same_square",
        replace(BASELINE, pawn_same_square_fizzle_scope="any_same_square"),
        5_000,
    ),
    ("recapture_cooldown=off", replace(BASELINE, recapture_cooldown=False), 5_000),
    ("horizon=30", replace(BASELINE, horizon=30), 2_500),
    ("horizon=80", replace(BASELINE, horizon=80), 2_500),
)
"""Each arm's *control* group is the matrix_1ply-vs-matrix_1ply slice of the
baseline tournament matrix (reused, not re-run) -- BASELINE already equals
the "on"/default level of every arm (rules/ruleset.py), which is why this is
a 25,000-new-game budget total, not per level: H splits its 5,000-game
budget across its two non-baseline values (30, 80) since 50 is baseline.
All arms run matrix_1ply self-play (the only strategic agent -- rule
effects on random/greedy play carry little evidentiary weight for a balance
report), per the maintainer's own call for this campaign.
"""

_TOURNAMENT_SEED_BLOCK = 1_000_000
_ARM_SEED_BASE = 100_000_000
_ARM_SEED_BLOCK = 1_000_000
_PILOT_SEED_OFFSET = 900_000_000
_PILOT_N = 40
"""Uniform per-run-spec game count for `pilot` mode: enough to exercise
every code path (every agent pairing, every RuleSet variant, the violation
harness, draw-reason attribution) cheaply, on seeds that can never collide
with a real pre-registered run.
"""


@dataclass(frozen=True, slots=True)
class RunSpec:
    """One pre-registered slice of the campaign: an agent pairing playing a
    fixed `RuleSet` for `n_games` seeded games starting at `base_seed`.
    """

    label: str
    agent_white: str
    agent_black: str
    ruleset: RuleSet
    base_seed: int
    n_games: int
    max_phases: int = DEFAULT_MAX_PHASES


def _tournament_run_specs() -> tuple[RunSpec, ...]:
    pairs = [(w, b) for w in AGENT_NAMES for b in AGENT_NAMES]
    base_n, remainder = divmod(TOURNAMENT_TOTAL_GAMES, len(pairs))
    specs = []
    for idx, (white, black) in enumerate(pairs):
        n = base_n + (1 if idx < remainder else 0)
        specs.append(
            RunSpec(
                label=f"tournament:{white}_vs_{black}",
                agent_white=white,
                agent_black=black,
                ruleset=BASELINE,
                base_seed=idx * _TOURNAMENT_SEED_BLOCK,
                n_games=n,
            )
        )
    return tuple(specs)


def _arm_run_specs() -> tuple[RunSpec, ...]:
    specs = []
    for idx, (label, ruleset, n) in enumerate(AB_ARM_DEFS):
        specs.append(
            RunSpec(
                label=f"arm:{label}",
                agent_white="matrix_1ply",
                agent_black="matrix_1ply",
                ruleset=ruleset,
                base_seed=_ARM_SEED_BASE + idx * _ARM_SEED_BLOCK,
                n_games=n,
            )
        )
    return tuple(specs)


ALL_RUN_SPECS: tuple[RunSpec, ...] = _tournament_run_specs() + _arm_run_specs()

BASELINE_MM_LABEL = "tournament:matrix_1ply_vs_matrix_1ply"
"""The A/B arms' control group: the mm slice of the tournament matrix."""


def _pilot_variant(spec: RunSpec) -> RunSpec:
    return replace(
        spec,
        n_games=min(_PILOT_N, spec.n_games),
        base_seed=spec.base_seed + _PILOT_SEED_OFFSET,
    )


def _ruleset_diff(ruleset: RuleSet) -> str:
    diffs = [
        f"{f.name}={getattr(ruleset, f.name)}"
        for f in fields(ruleset)
        if getattr(ruleset, f.name) != getattr(BASELINE, f.name)
    ]
    return ", ".join(diffs) if diffs else "baseline"


# ---------------------------------------------------------------------------
# Per-game instrumentation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CriticalViolation:
    """An S0/S1 violation, kept in full so the DoD's "zero S0/S1" claim is
    checkable and any surfacing case is reproducible from its seed.
    """

    rng_seed: int
    phase_index: int
    invariant_id: str
    severity: str
    detail: str


@dataclass(frozen=True, slots=True)
class GameRecord:
    """One seeded game's outcome plus every per-phase estimand this campaign
    needs, aggregated so the raw per-phase trace needn't be retained.
    """

    run_label: str
    rng_seed: int
    outcome: str
    phases_played: int
    violation_counts: dict[str, int]
    draw_reason: str | None
    reservation_rate: float
    cancellation_rate: float
    cooldown_occupancy: float
    material_final: float
    material_volatility: float
    critical_violations: tuple[CriticalViolation, ...] = field(default_factory=tuple)


def _attribute_draw_reason(state: State, ruleset: RuleSet) -> str:
    """Replicate `core/phi.py`'s own draw-cause precedence externally (no
    core changes): mutual king loss, then repetition, then the T4 horizon.
    `public_position_key` depends only on (board, cooldown) -- not on the
    ledger -- so recomputing it here reproduces the exact key `phi` used.
    """
    if detect_terminal(state.board) == "draw":
        return "mutual_king_loss"
    position_key = public_position_key(state)
    if state.bookkeeping.repetition_ledger.get(position_key, 0) >= 3:
        return "repetition"
    if state.bookkeeping.no_progress_counter >= ruleset.horizon:
        return "horizon"
    return "other"


def play_campaign_game(run_spec: RunSpec, seed: int) -> GameRecord:
    """Play one seeded game and derive every campaign estimand from it via
    `play_one_game`'s `on_phase_result` hook, instead of forking the phase
    loop (harness/selfplay.py).
    """
    agent_white = AGENT_REGISTRY[run_spec.agent_white]
    agent_black = AGENT_REGISTRY[run_spec.agent_black]
    initial_state = standard_starting_state()

    material_trajectory: list[float] = [material_difference(initial_state)]
    cooldown_fractions: list[float] = []
    reservation_phases = 0
    cancellation_phases = 0
    total_phases = 0
    violation_counts = {"S0": 0, "S1": 0, "S2": 0, "S3": 0}
    critical: list[CriticalViolation] = []
    draw_reason: str | None = None

    def hook(
        phase_index: int,
        _state_before: State,
        program_white: Program,
        program_black: Program,
        result: HarnessResult,
    ) -> None:
        nonlocal reservation_phases, cancellation_phases, total_phases, draw_reason
        total_phases += 1
        for violation in result.violations:
            severity = severity_of(violation.invariant_id)
            violation_counts[severity] += 1
            if severity in ("S0", "S1"):
                critical.append(
                    CriticalViolation(
                        rng_seed=seed,
                        phase_index=phase_index,
                        invariant_id=violation.invariant_id,
                        severity=severity,
                        detail=violation.detail,
                    )
                )
        if any(isinstance(a, Reserve) for a in (*program_white, *program_black)):
            reservation_phases += 1
        if result.phi_result is not None:
            if result.phi_result.trace.cancelled:
                cancellation_phases += 1
            state_after = result.phi_result.state
            material_trajectory.append(material_difference(state_after))
            live = len(state_after.board)
            cooldown_fractions.append(len(state_after.cooldown) / live if live else 0.0)
            if result.phi_result.outcome == "draw":
                draw_reason = _attribute_draw_reason(state_after, run_spec.ruleset)

    report: GameReport = play_one_game(
        initial_state,
        agent_white,
        agent_black,
        run_spec.ruleset,
        seed,
        max_phases=run_spec.max_phases,
        on_phase_result=hook,
    )

    n_phases = max(total_phases, 1)
    return GameRecord(
        run_label=run_spec.label,
        rng_seed=seed,
        outcome=report.outcome,
        phases_played=report.phases_played,
        violation_counts=violation_counts,
        draw_reason=draw_reason,
        reservation_rate=reservation_phases / n_phases,
        cancellation_rate=cancellation_phases / n_phases,
        cooldown_occupancy=(
            sum(cooldown_fractions) / len(cooldown_fractions)
            if cooldown_fractions
            else 0.0
        ),
        material_final=material_trajectory[-1],
        material_volatility=volatility(material_trajectory),
        critical_violations=tuple(critical),
    )


def _play_one_game_worker(args: tuple[RunSpec, int]) -> GameRecord:
    """Top-level (picklable) entry point for `ProcessPoolExecutor`. Work
    items carry the agent *names* already (via `RunSpec`), not closures --
    each worker process resolves them against the module-level
    `AGENT_REGISTRY`, since bound callables don't pickle across the spawn
    boundary macOS's default multiprocessing start method uses.
    """
    run_spec, seed = args
    return play_campaign_game(run_spec, seed)


# ---------------------------------------------------------------------------
# Parallel runner + checkpointed cache
# ---------------------------------------------------------------------------


def run_campaign(
    run_specs: Sequence[RunSpec], *, workers: int | None = None, progress: bool = True
) -> dict[str, list[GameRecord]]:
    """Run every game across `run_specs` through a process pool. Games are
    fully independent (each is a pure function of its own seed), so this is
    embarrassingly parallel -- no shared state crosses the pool boundary.
    """
    n_workers = workers or os.cpu_count() or 1
    work_items = [
        (spec, spec.base_seed + i) for spec in run_specs for i in range(spec.n_games)
    ]
    results: dict[str, list[GameRecord]] = {spec.label: [] for spec in run_specs}
    total = len(work_items)
    start = time.monotonic()

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_play_one_game_worker, item) for item in work_items]
        for done, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            results[record.run_label].append(record)
            if progress and (done % 200 == 0 or done == total):
                elapsed = time.monotonic() - start
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (total - done) / rate if rate > 0 else float("inf")
                print(
                    f"[{done}/{total}] elapsed={elapsed:.0f}s eta={eta:.0f}s",
                    file=sys.stderr,
                )
    return results


def _save_cache(path: Path, cache: dict[str, list[GameRecord]]) -> None:
    payload = {label: [asdict(r) for r in records] for label, records in cache.items()}
    path.write_text(json.dumps(payload))


def _game_record_from_dict(d: dict[str, Any]) -> GameRecord:
    return GameRecord(
        run_label=d["run_label"],
        rng_seed=d["rng_seed"],
        outcome=d["outcome"],
        phases_played=d["phases_played"],
        violation_counts=dict(d["violation_counts"]),
        draw_reason=d["draw_reason"],
        reservation_rate=d["reservation_rate"],
        cancellation_rate=d["cancellation_rate"],
        cooldown_occupancy=d["cooldown_occupancy"],
        material_final=d["material_final"],
        material_volatility=d["material_volatility"],
        critical_violations=tuple(
            CriticalViolation(**cv) for cv in d["critical_violations"]
        ),
    )


def _load_cache(path: Path) -> dict[str, list[GameRecord]]:
    payload = json.loads(path.read_text())
    return {
        label: [_game_record_from_dict(d) for d in records]
        for label, records in payload.items()
    }


def run_full_campaign(
    run_specs: Sequence[RunSpec],
    cache_path: Path,
    *,
    workers: int | None = None,
    resume: bool = False,
) -> dict[str, list[GameRecord]]:
    """Run every run spec one at a time, checkpointing the cache file after
    each completes. `--resume` skips specs already fully present in an
    existing cache -- a multi-hour full run survives an interruption.
    """
    cache: dict[str, list[GameRecord]] = {}
    if resume and cache_path.exists():
        cache = _load_cache(cache_path)

    for spec in run_specs:
        if resume and len(cache.get(spec.label, [])) >= spec.n_games:
            print(
                f"skip {spec.label}: already complete ({spec.n_games} games)",
                file=sys.stderr,
            )
            continue
        print(f"running {spec.label}: {spec.n_games} games", file=sys.stderr)
        records = run_campaign([spec], workers=workers)[spec.label]
        cache[spec.label] = sorted(records, key=lambda r: r.rng_seed)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _save_cache(cache_path, cache)

    return cache


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _mean(values: Iterable[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")


@dataclass(frozen=True, slots=True)
class Aggregate:
    n: int
    white_wins: int
    black_wins: int
    draws: int
    other: int
    draw_rate: WilsonInterval
    phase_median: float
    phase_q1: float
    phase_q3: float
    draw_reasons: dict[str, int]
    reservation_rate_mean: float
    cancellation_rate_mean: float
    cooldown_occupancy_mean: float
    volatility_mean: float
    violation_counts: dict[str, int]
    critical: tuple[CriticalViolation, ...]


def _aggregate(records: Sequence[GameRecord]) -> Aggregate:
    n = len(records)
    white_wins = sum(1 for r in records if r.outcome == "white_wins")
    black_wins = sum(1 for r in records if r.outcome == "black_wins")
    draws = sum(1 for r in records if r.outcome == "draw")
    draw_reasons: dict[str, int] = {}
    for r in records:
        if r.draw_reason is not None:
            draw_reasons[r.draw_reason] = draw_reasons.get(r.draw_reason, 0) + 1
    violation_counts = {"S0": 0, "S1": 0, "S2": 0, "S3": 0}
    critical: list[CriticalViolation] = []
    for r in records:
        for severity, count in r.violation_counts.items():
            violation_counts[severity] = violation_counts.get(severity, 0) + count
        critical.extend(r.critical_violations)
    phases = median_iqr([float(r.phases_played) for r in records])
    return Aggregate(
        n=n,
        white_wins=white_wins,
        black_wins=black_wins,
        draws=draws,
        other=n - white_wins - black_wins - draws,
        draw_rate=wilson_interval(draws, n),
        phase_median=phases.median,
        phase_q1=phases.q1,
        phase_q3=phases.q3,
        draw_reasons=draw_reasons,
        reservation_rate_mean=_mean(r.reservation_rate for r in records),
        cancellation_rate_mean=_mean(r.cancellation_rate for r in records),
        cooldown_occupancy_mean=_mean(r.cooldown_occupancy for r in records),
        volatility_mean=_mean(r.material_volatility for r in records),
        violation_counts=violation_counts,
        critical=tuple(critical),
    )


def render_report(
    cache: dict[str, list[GameRecord]], run_specs: Sequence[RunSpec]
) -> str:
    tournament_specs = [s for s in run_specs if s.label.startswith("tournament:")]
    arm_specs = [s for s in run_specs if s.label.startswith("arm:")]
    tournament_aggs = {
        s.label: _aggregate(cache.get(s.label, [])) for s in tournament_specs
    }
    arm_aggs = {s.label: _aggregate(cache.get(s.label, [])) for s in arm_specs}
    mm_agg = tournament_aggs[BASELINE_MM_LABEL]
    total_games = sum(a.n for a in tournament_aggs.values()) + sum(
        a.n for a in arm_aggs.values()
    )

    lines: list[str] = []
    lines.append("# Campaign v1 — Phase 11b pre-registered empirical report\n")
    generated_at = datetime.now(timezone.utc).isoformat()
    lines.append(
        f"Generated {generated_at}. {total_games} games total. "
        "Estimands, statistics, and sample sizes were declared before any run "
        "(`docs/DEVELOPMENT_addendum_v1.1.md` §11b); every table below is "
        "traceable to the run-spec constants in `harness/campaign.py` "
        "(agent pair, `RuleSet` diff from baseline, seed range).\n"
    )
    lines.append(
        "> **Interpretive caveat.** All balance statistics here are "
        "functionals of the state distributions induced by *these agents*, "
        "not of equilibrium play. The freeze this report supports is "
        "therefore provisional by construction (ruling A5) and is "
        "re-estimated after Phase 13 under learned agents.\n"
    )

    lines.append("## 1–2. Tournament matrix: draw rate & phase-count distribution\n")
    lines.append(
        "| pairing | n | white wins | black wins | draws | other | "
        "draw rate (95% Wilson) | phase median (IQR) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for spec in tournament_specs:
        agg = tournament_aggs[spec.label]
        draw_ci = f"[{agg.draw_rate.low:.3f}, {agg.draw_rate.high:.3f}]"
        phases = f"{agg.phase_median:.0f} ({agg.phase_q1:.0f}–{agg.phase_q3:.0f})"
        lines.append(
            f"| {spec.agent_white} vs {spec.agent_black} | {agg.n} | "
            f"{agg.white_wins} | {agg.black_wins} | {agg.draws} | {agg.other} | "
            f"{agg.draw_rate.point:.3f} {draw_ci} | {phases} |"
        )
    lines.append("")

    lines.append("## 3. T4-horizon attribution (fraction of draws by cause)\n")
    lines.append(
        "| pairing | draws | mutual_king_loss | repetition | horizon | other |"
    )
    lines.append("|---|---|---|---|---|---|")
    for spec in tournament_specs:
        agg = tournament_aggs[spec.label]
        d = max(agg.draws, 1)
        lines.append(
            f"| {spec.agent_white} vs {spec.agent_black} | {agg.draws} | "
            f"{agg.draw_reasons.get('mutual_king_loss', 0) / d:.3f} | "
            f"{agg.draw_reasons.get('repetition', 0) / d:.3f} | "
            f"{agg.draw_reasons.get('horizon', 0) / d:.3f} | "
            f"{agg.draw_reasons.get('other', 0) / d:.3f} |"
        )
    lines.append("")

    lines.append(
        "## 4–5. Reservation / cancellation / cooldown usage & material volatility\n"
    )
    lines.append(
        "| pairing | reservation rate | cancellation rate | cooldown occupancy | "
        "material volatility (stdev of increments) |"
    )
    lines.append("|---|---|---|---|---|")
    for spec in tournament_specs:
        agg = tournament_aggs[spec.label]
        lines.append(
            f"| {spec.agent_white} vs {spec.agent_black} | "
            f"{agg.reservation_rate_mean:.3f} | {agg.cancellation_rate_mean:.3f} | "
            f"{agg.cooldown_occupancy_mean:.3f} | {agg.volatility_mean:.3f} |"
        )
    lines.append(
        "\n> **Cancellation-usage caveat, found during the campaign pilot.** "
        "`agents/candidates.py` (shared by `random_legal`, `greedy`, and "
        "`matrix_1ply` via `solver/supports.py`) only enumerates "
        "Move/Castle/Reserve candidates -- no agent in this roster can ever "
        "declare a `Cancel` action. Cancellation-usage rate is therefore "
        "structurally 0.000 everywhere in this campaign, and the "
        "`cancellation_enabled` A/B arm below is a confirmed null by "
        "construction, not evidence the rule has no effect. Revisit under "
        "Phase 13's learned agents, which aren't limited to this candidate "
        "set.\n"
    )

    lines.append(
        "## 6. Color-symmetry audit (baseline config, pooled decisive games)\n"
    )
    pooled_white = sum(a.white_wins for a in tournament_aggs.values())
    pooled_decisive = pooled_white + sum(a.black_wins for a in tournament_aggs.values())
    symmetry = color_symmetry_test(pooled_white, pooled_decisive)
    lines.append(
        f"H0: p_White = 1/2. Pooled across all 9 baseline pairings: "
        f"{symmetry.white_wins}/{symmetry.decisive} decisive games White, "
        f"p = {symmetry.wilson.point:.4f} "
        f"[{symmetry.wilson.low:.4f}, {symmetry.wilson.high:.4f}], "
        f"two-sided exact binomial p-value = {symmetry.p_value:.4g}. "
        "Power target was ~4.9×10³ decisive games at δ=0.02, α=.05, "
        "1-β=.8 (§11b); since M3 proves operator-level symmetry exactly, "
        "any rejection here localizes to agent asymmetry, not a rules bug.\n"
    )

    lines.append("## A/B arms (matrix_1ply self-play; control = baseline mm slice)\n")
    lines.append(
        f"Control: `{BASELINE_MM_LABEL}` — n={mm_agg.n}, "
        f"draw rate {mm_agg.draw_rate.point:.3f} "
        f"[{mm_agg.draw_rate.low:.3f}, {mm_agg.draw_rate.high:.3f}].\n"
    )
    lines.append(
        "| arm | RuleSet diff | n | draw rate (95% Wilson) | Δ draw rate vs control | "
        "MDE at this n (α=.05, 1-β=.8) |"
    )
    lines.append("|---|---|---|---|---|---|")
    for spec in arm_specs:
        agg = arm_aggs[spec.label]
        mde = minimum_detectable_effect(spec.n_games, mm_agg.draw_rate.point)
        lines.append(
            f"| {spec.label.removeprefix('arm:')} | {_ruleset_diff(spec.ruleset)} | "
            f"{agg.n} | {agg.draw_rate.point:.3f} "
            f"[{agg.draw_rate.low:.3f}, {agg.draw_rate.high:.3f}] | "
            f"{agg.draw_rate.point - mm_agg.draw_rate.point:+.3f} | ±{mde:.3f} |"
        )
    lines.append("")

    lines.append(
        "### A/B arms: estimands 3–5 detail (same breakdown as the "
        "tournament matrix, for interpreting *why* a draw-rate delta moved)\n"
    )
    lines.append(
        "| arm | mutual_king_loss | repetition | horizon | reservation rate | "
        "cooldown occupancy | volatility |"
    )
    lines.append("|---|---|---|---|---|---|---|")

    def _arm_detail_row(name: str, agg: Aggregate) -> str:
        d = max(agg.draws, 1)
        mkl = agg.draw_reasons.get("mutual_king_loss", 0) / d
        rep = agg.draw_reasons.get("repetition", 0) / d
        hzn = agg.draw_reasons.get("horizon", 0) / d
        return (
            f"| {name} | {mkl:.3f} | {rep:.3f} | {hzn:.3f} | "
            f"{agg.reservation_rate_mean:.3f} | {agg.cooldown_occupancy_mean:.3f} | "
            f"{agg.volatility_mean:.3f} |"
        )

    lines.append(_arm_detail_row("*(control) mm baseline*", mm_agg))
    for spec in arm_specs:
        lines.append(
            _arm_detail_row(spec.label.removeprefix("arm:"), arm_aggs[spec.label])
        )
    lines.append("")

    lines.append("## Violations summary (DoD: zero S0/S1 across all arms)\n")
    all_aggs = list(tournament_aggs.values()) + list(arm_aggs.values())
    totals = {"S0": 0, "S1": 0, "S2": 0, "S3": 0}
    all_critical: list[CriticalViolation] = []
    for agg in all_aggs:
        for severity, count in agg.violation_counts.items():
            totals[severity] += count
        all_critical.extend(agg.critical)
    lines.append(
        f"S0={totals['S0']}, S1={totals['S1']}, S2={totals['S2']}, S3={totals['S3']} "
        f"across {total_games} games.\n"
    )
    if all_critical:
        lines.append("**S0/S1 violations found — maintainer follow-up required:**\n")
        lines.append("| seed | phase | invariant | severity | detail |")
        lines.append("|---|---|---|---|---|")
        for cv in all_critical:
            lines.append(
                f"| {cv.rng_seed} | {cv.phase_index} | {cv.invariant_id} | "
                f"{cv.severity} | {cv.detail} |"
            )
        lines.append("")
    else:
        lines.append("None found.\n")

    lines.append("## Seed reference appendix\n")
    lines.append("| run spec | agent pair | RuleSet diff | base seed | n games |")
    lines.append("|---|---|---|---|---|")
    for spec in run_specs:
        lines.append(
            f"| {spec.label} | {spec.agent_white} vs {spec.agent_black} | "
            f"{_ruleset_diff(spec.ruleset)} | {spec.base_seed} | {spec.n_games} |"
        )
    lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_pilot_summary(cache: dict[str, list[GameRecord]]) -> None:
    for label, records in cache.items():
        agg = _aggregate(records)
        print(
            f"{label}: n={agg.n} draws={agg.draws} "
            f"S0={agg.violation_counts['S0']} S1={agg.violation_counts['S1']} "
            f"reservation_rate={agg.reservation_rate_mean:.3f} "
            f"cancellation_rate={agg.cancellation_rate_mean:.3f} "
            f"volatility={agg.volatility_mean:.3f}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m simult_chess.harness.campaign")
    sub = parser.add_subparsers(dest="command", required=True)

    pilot_p = sub.add_parser(
        "pilot",
        help="Scaled-down validation run -- not part of the pre-registered statistics.",
    )
    pilot_p.add_argument("--workers", type=int, default=None)

    run_p = sub.add_parser("run", help="Full pre-registered campaign.")
    run_p.add_argument("--workers", type=int, default=None)
    run_p.add_argument("--resume", action="store_true")

    sub.add_parser(
        "report", help="Render reports/campaign_v1.md from campaign_runs/full.json."
    )

    args = parser.parse_args(argv)
    CAMPAIGN_RUNS_DIR.mkdir(parents=True, exist_ok=True)

    if args.command == "pilot":
        specs = [_pilot_variant(s) for s in ALL_RUN_SPECS]
        cache = run_full_campaign(
            specs, PILOT_CACHE_PATH, workers=args.workers, resume=False
        )
        _print_pilot_summary(cache)
    elif args.command == "run":
        run_full_campaign(
            list(ALL_RUN_SPECS),
            FULL_CACHE_PATH,
            workers=args.workers,
            resume=args.resume,
        )
    elif args.command == "report":
        if not FULL_CACHE_PATH.exists():
            raise SystemExit(
                f"{FULL_CACHE_PATH} not found -- run `campaign run` first."
            )
        cache = _load_cache(FULL_CACHE_PATH)
        missing = [
            s.label for s in ALL_RUN_SPECS if len(cache.get(s.label, [])) < s.n_games
        ]
        if missing:
            raise SystemExit(
                f"incomplete run specs: {missing} -- run `campaign run --resume` first."
            )
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(render_report(cache, ALL_RUN_SPECS))
        print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
