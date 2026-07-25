"""Phase 15b: the clock inside a live online session (flag-fall, cross-check)."""

from __future__ import annotations

import asyncio
import random
import time

import pytest
from _memory_transport import memory_peer_pair

from simult_chess.agents.random_legal import random_legal_program
from simult_chess.core.types import Color, State
from simult_chess.net.clock import Banks, TimeControl
from simult_chess.net.protocol import ProtocolError
from simult_chess.net.session import (
    Decision,
    OnlineMatchResult,
    PlayDecision,
    _advance_clock,
    _ClockState,
    agent_decider,
    run_online_match,
)
from simult_chess.referee.setup import standard_starting_state
from simult_chess.rules.ruleset import RuleSet

RULESET = RuleSet()
FAST = {"keepalive_interval": 0.02, "liveness_deadline": 2.0, "transport_timeout": 2.0}


def _run(
    decider_white: object,
    decider_black: object,
    time_control: TimeControl,
    *,
    max_phases: int = 40,
    delay_a: float = 0.0,
    delay_b: float = 0.0,
) -> tuple[OnlineMatchResult, OnlineMatchResult]:
    async def scenario() -> tuple[OnlineMatchResult, OnlineMatchResult]:
        peer_a, peer_b = memory_peer_pair(delay_a=delay_a, delay_b=delay_b)
        state = standard_starting_state()

        async def side(peer: object, color: Color, decider: object, seed: int):
            return await run_online_match(
                state, RULESET, color, decider, peer, random.Random(seed),  # type: ignore[arg-type]
                print_fn=lambda _line: None, time_control=time_control,
                max_phases=max_phases, **FAST,  # type: ignore[arg-type]
            )

        return await asyncio.gather(
            side(peer_a, Color.WHITE, decider_white, 1),
            side(peer_b, Color.BLACK, decider_black, 2),
        )

    return asyncio.run(scenario())


# --- end-to-end ---------------------------------------------------------------


def test_timed_agent_game_completes_and_agrees() -> None:
    tc = TimeControl(initial_bank=180.0, bonus=2.0)
    white, black = _run(
        agent_decider(random_legal_program),
        agent_decider(random_legal_program),
        tc,
    )
    # No clock/state divergence was raised, and both sides agree.
    assert white.outcome == black.outcome
    assert white.termination_reason == black.termination_reason
    assert white.final_state.board == black.final_state.board


def test_ledger_survives_asymmetric_latency() -> None:
    """Byte-identical ledgers over a match with injected one-way latency (§15b)."""
    tc = TimeControl(initial_bank=180.0, bonus=2.0)
    white, black = _run(
        agent_decider(random_legal_program),
        agent_decider(random_legal_program),
        tc,
        delay_a=0.03,  # asymmetric: only A→B is delayed
        max_phases=12,
    )
    assert white.outcome == black.outcome
    assert white.final_state.board == black.final_state.board


def test_flag_fall_loses_on_time() -> None:
    """A side that thinks past its bank loses on time; both peers agree (§15b)."""

    def slow_white(
        state: State, color: Color, ruleset: RuleSet, rng: random.Random, *,
        peer_offered_draw: bool,
    ) -> Decision:
        time.sleep(0.4)  # off the event loop; exceeds the 0.2 s bank
        return PlayDecision(random_legal_program(state, color, ruleset, rng))

    tc = TimeControl(initial_bank=0.2)
    white, black = _run(slow_white, agent_decider(random_legal_program), tc)
    assert white.outcome == "black_wins"
    assert black.outcome == "black_wins"
    assert white.termination_reason == "timeout"
    assert black.termination_reason == "timeout"


# --- the clock cross-check and flag adjudication (unit) -----------------------


def _clock(d_max: float = 0.05, banks: Banks | None = None) -> _ClockState:
    tc = TimeControl(initial_bank=180.0, bonus=2.0)
    return _ClockState(tc=tc, d_max=d_max, banks=banks or Banks(180.0, 180.0))


def test_out_of_tolerance_time_claim_is_rejected() -> None:
    clock = _clock()
    with pytest.raises(ProtocolError, match="outside tolerance"):
        _advance_clock(
            clock,
            Color.WHITE,
            local_elapsed_us=1_000_000,
            remote_elapsed_us=100_000_000,  # claims 100 s
            phase=0,
            phase_start=0.0,
            remote_arrival=0.1,  # but arrived in 0.1 s
        )


def test_an_honest_claim_within_tolerance_is_accepted() -> None:
    clock = _clock()
    terminal, entry = _advance_clock(
        clock,
        Color.WHITE,
        local_elapsed_us=1_000_000,
        remote_elapsed_us=1_050_000,  # claims 1.05 s
        phase=0,
        phase_start=0.0,
        remote_arrival=1.1,  # arrived in 1.1 s (within d_max + ε)
    )
    assert terminal is None
    assert entry.think_black == pytest.approx(1.05)


def test_advance_clock_flags_the_slow_side() -> None:
    clock = _clock(banks=Banks(white=100.0, black=0.5))
    terminal, _entry = _advance_clock(
        clock,
        Color.WHITE,
        local_elapsed_us=1_000_000,  # White (local) spends 1 s, bank fine
        remote_elapsed_us=1_000_000,  # Black (remote) spends 1 s, bank 0.5 -> flag
        phase=3,
        phase_start=0.0,
        remote_arrival=1.0,
    )
    assert terminal is not None
    assert terminal.outcome == "white_wins"
    assert terminal.reason == "timeout"
