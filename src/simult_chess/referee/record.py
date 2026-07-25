"""The `.scn` game record: a replayable, citable record of one game (Phase 15d).

One artifact does four jobs (`docs/DEVELOPMENT_roadmap_v2.md` §15d): replay and
review; regression fixtures harvested from real play; worked examples for the
paper; and a citable object for the Zenodo deposit.

A record is **self-describing and self-verifying**. Its header pins the
`spec_version`, `protocol_version`, the Phase-14 `ruleset_fingerprint`, the full
`RuleSet` field dump, the initial position, player labels, the time control
(Phase 15b; ``none`` for now), and the seed. Its body is one line per phase
carrying both programs in `ui.notation` form and the resolved per-phase
outcome. Reading a record does not *trust* it: :func:`read_record` re-derives
every state through Φ from the initial position and checks that each recorded
outcome is the one Φ actually produces, refusing on any mismatch — and refusing
outright if the stored fingerprint disagrees with the dumped rules, so a record
can never silently replay under rules other than the ones it was made under.

This module formats programs with `ui.notation`, so `referee` depends on `ui`
here for presentation only; `ui.notation` itself imports nothing above `core`,
so no import cycle forms.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import Any, Literal

from simult_chess.core.phi import phi
from simult_chess.core.stages.closure import detect_terminal
from simult_chess.core.types import Color, Program, State
from simult_chess.referee.serialize import public_position_key, state_hash
from simult_chess.referee.setup import standard_starting_state
from simult_chess.rules.ruleset import RuleSet
from simult_chess.ui import notation

#: Bumped on any breaking change to the ``.scn`` grammar.
SCN_VERSION = 1

SPEC_VERSION = "1.1"
PROTOCOL_VERSION = 1

#: Named initial positions. v1 games all start from the standard position; a
#: custom start would register a builder here (and, to be replayable, a
#: deserializer), which is a deliberate future extension, not v1 scope.
_SETUPS: dict[str, Callable[[], State]] = {"standard": standard_starting_state}

MatchOutcome = Literal["white_wins", "black_wins", "draw"]
TerminationReason = Literal[
    "regicide",
    "repetition",
    "no_progress",
    "resignation",
    "timeout",
    "illegal_program",
    "abort",
    "draw_agreement",
    "phase_limit",
]

_FIELD_SEP = " | "


class RecordError(Exception):
    """A record is malformed, internally inconsistent, or fails to replay."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GamePhase:
    """One phase of a recorded game.

    Parameters
    ----------
    program_white, program_black : Program
        The two declared programs, resolved against the phase's pre-state.
    outcome : str
        Φ's per-phase outcome (``ongoing`` until the terminal phase).
    clock : str
        The clock-ledger entry (Phase 15b); ``-`` while there is no clock.
    """

    program_white: Program
    program_black: Program
    outcome: str
    clock: str = "-"


@dataclass(frozen=True, slots=True)
class GameRecord:
    """A complete, replayable game.

    The `phases` carry `Program` objects resolved against each pre-state; the
    header metadata is everything needed to reconstruct those states and verify
    the rules they were played under.
    """

    ruleset: RuleSet
    initial_state: State
    phases: tuple[GamePhase, ...]
    outcome: MatchOutcome
    termination_reason: TerminationReason
    white_label: str = "white"
    black_label: str = "black"
    setup: str = "standard"
    time_control: str = "none"
    seed: int | None = None
    scn_version: int = SCN_VERSION
    spec_version: str = SPEC_VERSION
    protocol_version: int = PROTOCOL_VERSION

    @property
    def ruleset_fingerprint(self) -> str:
        return self.ruleset.fingerprint()


# ---------------------------------------------------------------------------
# RuleSet <-> field dump (round-trips through repr, cross-checked by fingerprint)
# ---------------------------------------------------------------------------


def _ruleset_fields(ruleset: RuleSet) -> str:
    """The rule-bearing fields as ``name=repr(value)``, name-sorted."""
    return " ".join(
        f"{spec.name}={getattr(ruleset, spec.name)!r}"
        for spec in sorted(fields(ruleset), key=lambda f: f.name)
    )


def _ruleset_from_fields(text: str) -> RuleSet:
    """Rebuild a `RuleSet` from a :func:`_ruleset_fields` dump."""
    kwargs: dict[str, Any] = {}
    for token in text.split():
        name, _, value = token.partition("=")
        try:
            kwargs[name] = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise RecordError(f"bad ruleset field {token!r}: {exc}") from exc
    try:
        return RuleSet(**kwargs)
    except TypeError as exc:
        raise RecordError(f"bad ruleset fields: {exc}") from exc


# ---------------------------------------------------------------------------
# Termination-reason attribution (for the loops that only produce a Φ outcome)
# ---------------------------------------------------------------------------


def normalize_outcome_and_reason(
    final_state: State, ruleset: RuleSet, raw_outcome: str
) -> tuple[MatchOutcome, TerminationReason]:
    """Map a self-play/hot-seat loop's raw outcome to ``(payoff, reason)``.

    Φ collapses the three drawing causes to ``draw``; this recovers the reason
    the same way `phi` decides it (mutual king loss, then repetition, then the
    T4 horizon), and maps a defensive phase-cap hit to a ``phase_limit`` draw.
    """
    if raw_outcome in ("white_wins", "black_wins"):
        return raw_outcome, "regicide"  # type: ignore[return-value]
    if raw_outcome == "draw":
        if detect_terminal(final_state.board) == "draw":
            return "draw", "regicide"  # both kings captured
        key = public_position_key(final_state)
        if final_state.bookkeeping.repetition_ledger.get(key, 0) >= 3:
            return "draw", "repetition"
        if final_state.bookkeeping.no_progress_counter >= ruleset.horizon:
            return "draw", "no_progress"
        return "draw", "no_progress"
    # "ongoing" (cap reached) / "phase_limit_reached"
    return "draw", "phase_limit"


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def build_record(
    *,
    initial_state: State,
    ruleset: RuleSet,
    phases: tuple[GamePhase, ...],
    final_state: State,
    raw_outcome: str,
    white_label: str = "white",
    black_label: str = "black",
    seed: int | None = None,
    setup: str = "standard",
    time_control: str = "none",
) -> GameRecord:
    """Build a `GameRecord` from a Φ-terminated loop's result.

    For the self-play (`referee.match`) and hot-seat (`ui.session`) loops, whose
    only end-of-game datum is Φ's outcome; :func:`normalize_outcome_and_reason`
    recovers the payoff and reason. A network match instead constructs
    `GameRecord` directly, since its outcome and reason (resignation, agreement,
    …) are session facts Φ cannot report.
    """
    outcome, reason = normalize_outcome_and_reason(final_state, ruleset, raw_outcome)
    return GameRecord(
        ruleset=ruleset,
        initial_state=initial_state,
        phases=phases,
        outcome=outcome,
        termination_reason=reason,
        white_label=white_label,
        black_label=black_label,
        setup=setup,
        time_control=time_control,
        seed=seed,
    )


def write_record(record: GameRecord) -> str:
    """Serialize a `GameRecord` to ``.scn`` text.

    The programs are rendered against each phase's pre-state, which is
    re-derived through Φ exactly as :func:`read_record` re-derives it, so a
    written record round-trips.
    """
    lines: list[str] = ["# simult-chess game record (.scn)"]
    lines.append(f"scn_version {record.scn_version}")
    lines.append(f"spec_version {record.spec_version}")
    lines.append(f"protocol_version {record.protocol_version}")
    lines.append(f"ruleset_fingerprint {record.ruleset_fingerprint}")
    lines.append(f"ruleset {_ruleset_fields(record.ruleset)}")
    lines.append(f"setup {record.setup}")
    lines.append(f"initial_state_hash {state_hash(record.initial_state)}")
    lines.append(f"white {record.white_label}")
    lines.append(f"black {record.black_label}")
    lines.append(f"time_control {record.time_control}")
    lines.append(f"seed {'-' if record.seed is None else record.seed}")
    lines.append("# phase | white | black | outcome | clock")

    state = record.initial_state
    for index, phase in enumerate(record.phases):
        white_txt = notation.format_program(phase.program_white, state, Color.WHITE)
        black_txt = notation.format_program(phase.program_black, state, Color.BLACK)
        fields_out = [
            str(index),
            white_txt,
            black_txt,
            phase.outcome,
            phase.clock,
        ]
        lines.append(_FIELD_SEP.join(fields_out))
        state = phi(
            state, phase.program_white, phase.program_black, record.ruleset
        ).state

    lines.append(f"result {record.outcome} {record.termination_reason}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Reader / replayer (re-derives every state through Φ and verifies)
# ---------------------------------------------------------------------------


def _parse_header(lines: list[str]) -> tuple[dict[str, str], int]:
    """Return ``(header, body_start)`` — key→value pairs up to the phase table."""
    header: dict[str, str] = {}
    for i, raw in enumerate(lines):
        line = raw.rstrip("\n")
        if not line or line.startswith("#"):
            if line.startswith("# phase"):
                return header, i + 1
            continue
        key, _, value = line.partition(" ")
        if key.isdigit() or key == "result":
            return header, i
        header[key] = value
    return header, len(lines)


def read_record(
    text: str, *, expected_fingerprint: str | None = None
) -> GameRecord:
    """Parse and **verify** a ``.scn`` record by re-deriving it through Φ.

    Every state is recomputed from the initial position; each recorded phase
    outcome is checked against the one Φ produces. The record refuses to
    replay (raising :class:`RecordError`) if the stored fingerprint disagrees
    with the dumped rules, if `expected_fingerprint` is given and differs, if
    the initial position does not match its recorded hash, or if any resolution
    disagrees — never silently producing a different result.

    Parameters
    ----------
    text : str
        The ``.scn`` content.
    expected_fingerprint : str, optional
        If given, the record's fingerprint must equal it, else the record is
        refused. Used by the replay CLI to reject a record made under other
        rules than the ones the caller intends.
    """
    lines = text.splitlines()
    header, body_start = _parse_header(lines)

    ruleset = _ruleset_from_fields(_require(header, "ruleset"))
    stored_fp = _require(header, "ruleset_fingerprint")
    if ruleset.fingerprint() != stored_fp:
        raise RecordError(
            "ruleset fingerprint does not match the dumped rules: stored "
            f"{stored_fp[:12]}…, rebuilt {ruleset.fingerprint()[:12]}… — the "
            "record has been altered"
        )
    if expected_fingerprint is not None and stored_fp != expected_fingerprint:
        raise RecordError(
            f"record fingerprint {stored_fp[:12]}… does not match the expected "
            f"{expected_fingerprint[:12]}…; refusing to replay under other rules"
        )

    setup = header.get("setup", "standard")
    if setup not in _SETUPS:
        raise RecordError(f"unknown setup {setup!r}")
    initial_state = _SETUPS[setup]()
    if state_hash(initial_state) != _require(header, "initial_state_hash"):
        raise RecordError("initial position does not match its recorded hash")

    phases: list[GamePhase] = []
    state = initial_state
    for raw in lines[body_start:]:
        line = raw.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        if line.startswith("result "):
            break
        phase, state = _replay_phase_line(line, state, ruleset)
        phases.append(phase)

    outcome, reason = _parse_result(lines)
    return GameRecord(
        ruleset=ruleset,
        initial_state=initial_state,
        phases=tuple(phases),
        outcome=outcome,
        termination_reason=reason,
        white_label=header.get("white", "white"),
        black_label=header.get("black", "black"),
        setup=setup,
        time_control=header.get("time_control", "none"),
        seed=_parse_seed(header.get("seed", "-")),
        scn_version=int(header.get("scn_version", SCN_VERSION)),
        spec_version=header.get("spec_version", SPEC_VERSION),
        protocol_version=int(header.get("protocol_version", PROTOCOL_VERSION)),
    )


def _replay_phase_line(
    line: str, state: State, ruleset: RuleSet
) -> tuple[GamePhase, State]:
    parts = line.split(_FIELD_SEP)
    if len(parts) != 5:
        raise RecordError(f"malformed phase line: {line!r}")
    index_txt, white_txt, black_txt, outcome_txt, clock_txt = parts
    try:
        program_white = notation.parse_program(white_txt, state, Color.WHITE)
        program_black = notation.parse_program(black_txt, state, Color.BLACK)
    except notation.NotationError as exc:
        raise RecordError(f"phase {index_txt}: unparseable program: {exc}") from exc
    result = phi(state, program_white, program_black, ruleset)
    if result.outcome != outcome_txt:
        raise RecordError(
            f"phase {index_txt}: recorded outcome {outcome_txt!r} but Φ "
            f"resolved {result.outcome!r} — the record is not reproducible"
        )
    return (
        GamePhase(program_white, program_black, result.outcome, clock_txt),
        result.state,
    )


def _require(header: Mapping[str, str], key: str) -> str:
    if key not in header:
        raise RecordError(f"record header is missing {key!r}")
    return header[key]


def _parse_result(lines: list[str]) -> tuple[MatchOutcome, TerminationReason]:
    for raw in lines:
        line = raw.rstrip("\n")
        if line.startswith("result "):
            _, _, rest = line.partition(" ")
            outcome, _, reason = rest.partition(" ")
            return outcome, reason  # type: ignore[return-value]
    raise RecordError("record is missing its 'result' line")


def _parse_seed(value: str) -> int | None:
    if value == "-":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise RecordError(f"bad seed {value!r}") from exc


# ---------------------------------------------------------------------------
# Phase fixtures (harvest one phase as a self-contained regression case)
# ---------------------------------------------------------------------------


def phase_fixture(record: GameRecord, phase_index: int) -> dict[str, Any]:
    """A self-contained regression fixture pinned to one phase of `record`.

    Emits the rules, the setup, and the program sequence up to and including
    ``phase_index`` (as notation), with the expected post-phase outcome and
    public-position hash — enough for :func:`verify_phase_fixture` to reproduce
    that phase through Φ with no reference to the ``.scn`` file. No full-state
    dump is needed (and none is available: v1 has no state deserializer), since
    the pre-state is re-derived from the setup.
    """
    if not 0 <= phase_index < len(record.phases):
        raise RecordError(
            f"phase {phase_index} out of range (0..{len(record.phases) - 1})"
        )
    programs: list[list[str]] = []
    state = record.initial_state
    expected_outcome = ""
    for index in range(phase_index + 1):
        phase = record.phases[index]
        programs.append(
            [
                notation.format_program(phase.program_white, state, Color.WHITE),
                notation.format_program(phase.program_black, state, Color.BLACK),
            ]
        )
        result = phi(state, phase.program_white, phase.program_black, record.ruleset)
        state = result.state
        expected_outcome = result.outcome
    return {
        "source": f"{record.setup} phase {phase_index}",
        "ruleset_fingerprint": record.ruleset_fingerprint,
        "ruleset": _ruleset_fields(record.ruleset),
        "setup": record.setup,
        "programs": programs,
        "expected_outcome": expected_outcome,
        "expected_state_hash": state_hash(state),
    }


def write_phase_fixture(record: GameRecord, phase_index: int) -> str:
    """:func:`phase_fixture` as pretty JSON."""
    return json.dumps(phase_fixture(record, phase_index), indent=2) + "\n"


def verify_phase_fixture(fixture: Mapping[str, Any]) -> bool:
    """Reproduce a :func:`phase_fixture` through Φ; ``True`` iff it still holds.

    Raises :class:`RecordError` if the fixture's fingerprint disagrees with its
    dumped rules (an altered fixture), the same integrity guard as the reader.
    """
    ruleset = _ruleset_from_fields(str(fixture["ruleset"]))
    if ruleset.fingerprint() != fixture["ruleset_fingerprint"]:
        raise RecordError("phase fixture fingerprint does not match its rules")
    setup = str(fixture["setup"])
    if setup not in _SETUPS:
        raise RecordError(f"unknown setup {setup!r}")
    state = _SETUPS[setup]()
    result = None
    for white_txt, black_txt in fixture["programs"]:
        program_white = notation.parse_program(white_txt, state, Color.WHITE)
        program_black = notation.parse_program(black_txt, state, Color.BLACK)
        result = phi(state, program_white, program_black, ruleset)
        state = result.state
    if result is None:
        return False
    return bool(
        result.outcome == fixture["expected_outcome"]
        and state_hash(state) == fixture["expected_state_hash"]
    )
