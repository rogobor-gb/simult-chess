"""Phase 15d: the .scn game record — write, replay-verify, fixtures."""

from __future__ import annotations

import random

import pytest

from simult_chess.agents.random_legal import random_legal_program
from simult_chess.core.phi import phi
from simult_chess.core.types import State
from simult_chess.referee.match import play_match
from simult_chess.referee.record import (
    GamePhase,
    GameRecord,
    RecordError,
    build_record,
    phase_fixture,
    read_record,
    verify_phase_fixture,
    write_phase_fixture,
    write_record,
)
from simult_chess.referee.serialize import state_hash
from simult_chess.referee.setup import standard_starting_state
from simult_chess.rules.ruleset import RuleSet
from simult_chess.rules.variants import get_variant


def _selfplay_record(seed: int, ruleset: RuleSet | None = None) -> GameRecord:
    ruleset = ruleset or RuleSet()
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
    return build_record(
        initial_state=initial,
        ruleset=ruleset,
        phases=phases,
        final_state=result.final_state,
        raw_outcome=result.outcome,
        seed=seed,
    )


def _final_state(record: GameRecord) -> State:
    state = record.initial_state
    for phase in record.phases:
        state = phi(
            state, phase.program_white, phase.program_black, record.ruleset
        ).state
    return state


# --- round-trip ---------------------------------------------------------------


@pytest.mark.parametrize("seed", range(30))
def test_write_read_round_trips_to_identical_states(seed: int) -> None:
    record = _selfplay_record(seed)
    replayed = read_record(write_record(record))
    assert replayed.outcome == record.outcome
    assert replayed.termination_reason == record.termination_reason
    assert len(replayed.phases) == len(record.phases)
    assert _final_state(replayed).board == _final_state(record).board


@pytest.mark.slow
def test_round_trip_over_1000_seeded_self_play_games() -> None:
    """DoD: write -> read -> re-resolve -> byte-identical states over >=10^3 games."""
    for seed in range(1000):
        record = _selfplay_record(seed)
        replayed = read_record(write_record(record))
        original = _final_state(record)
        assert _final_state(replayed).board == original.board
        assert state_hash(replayed.initial_state) == state_hash(record.initial_state)


def test_a_variant_game_round_trips_and_preserves_its_rules() -> None:
    ruleset = get_variant("horizon_30")
    record = _selfplay_record(3, ruleset)
    replayed = read_record(write_record(record))
    assert replayed.ruleset == ruleset
    assert replayed.ruleset_fingerprint == ruleset.fingerprint()


# --- refusals -----------------------------------------------------------------


def test_altered_rule_without_updated_fingerprint_refuses() -> None:
    text = write_record(_selfplay_record(1))
    tampered = text.replace("horizon=50", "horizon=30")
    with pytest.raises(RecordError, match="does not match the dumped rules"):
        read_record(tampered)


def test_expected_fingerprint_mismatch_refuses() -> None:
    text = write_record(_selfplay_record(1))
    with pytest.raises(RecordError, match="does not match the expected"):
        read_record(text, expected_fingerprint=get_variant("horizon_30").fingerprint())


def test_tampered_phase_outcome_refuses_as_unreproducible() -> None:
    record = _selfplay_record(1)
    text = write_record(record)
    # Flip the first phase's recorded outcome to a lie.
    tampered = text.replace(
        "| ongoing | -", "| white_wins | -", 1
    )
    with pytest.raises(RecordError, match="not reproducible"):
        read_record(tampered)


def test_unknown_setup_refuses() -> None:
    text = write_record(_selfplay_record(1)).replace("setup standard", "setup martian")
    with pytest.raises(RecordError, match="unknown setup"):
        read_record(text)


def test_missing_result_line_refuses() -> None:
    text = write_record(_selfplay_record(1))
    without_result = "\n".join(
        line for line in text.splitlines() if not line.startswith("result ")
    )
    with pytest.raises(RecordError, match="missing its 'result'"):
        read_record(without_result)


# --- phase fixtures -----------------------------------------------------------


def test_phase_fixture_reproduces_and_detects_tampering() -> None:
    record = _selfplay_record(5)
    index = min(3, len(record.phases) - 1)
    fixture = phase_fixture(record, index)
    assert verify_phase_fixture(fixture)

    # A JSON round-trip is still valid.
    import json

    assert verify_phase_fixture(json.loads(write_phase_fixture(record, index)))

    # An altered expected hash fails verification (but is well-formed).
    fixture["expected_state_hash"] = "0" * 64
    assert not verify_phase_fixture(fixture)


def test_phase_fixture_out_of_range_refuses() -> None:
    record = _selfplay_record(1)
    with pytest.raises(RecordError, match="out of range"):
        phase_fixture(record, len(record.phases))


# --- a "real two-human game" replays exactly ----------------------------------


def test_network_game_record_replays_exactly() -> None:
    """A game played through the online session (Phase 15a) round-trips."""
    import asyncio

    from _memory_transport import memory_peer_pair

    from simult_chess.core.types import Color
    from simult_chess.net.session import (
        OnlineMatchResult,
        agent_decider,
        run_online_match,
    )

    async def scenario() -> OnlineMatchResult:
        peer_a, peer_b = memory_peer_pair()
        initial = standard_starting_state()
        ruleset = RuleSet()

        async def side(peer: object, color: Color, seed: int) -> OnlineMatchResult:
            return await run_online_match(
                initial, ruleset, color,
                agent_decider(random_legal_program), peer, random.Random(seed),  # type: ignore[arg-type]
                print_fn=lambda _line: None, keepalive_interval=0.02,
                liveness_deadline=1.0, transport_timeout=2.0, max_phases=60,
            )

        white, _black = await asyncio.gather(
            side(peer_a, Color.WHITE, 1), side(peer_b, Color.BLACK, 2)
        )
        return white

    result = asyncio.run(scenario())
    record = GameRecord(
        ruleset=RuleSet(),
        initial_state=standard_starting_state(),
        phases=result.phases,
        outcome=result.outcome,
        termination_reason=result.termination_reason,
        white_label="human_a",
        black_label="human_b",
    )
    replayed = read_record(write_record(record))
    assert replayed.outcome == result.outcome
    assert replayed.termination_reason == result.termination_reason
    assert _final_state(replayed).board == result.final_state.board
