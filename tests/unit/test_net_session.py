"""Phase 15a online session: full games, match services, keepalive, forfeits.

Uses the in-memory `Transport` (`_memory_transport.py`) so the concurrency —
non-blocking entry, keepalive, action-slot resolution — runs deterministically
without loopback TCP.
"""

from __future__ import annotations

import asyncio
import random
import time

import pytest
from _memory_transport import MemoryPeer, memory_peer_pair

from simult_chess.agents.random_legal import random_legal_program
from simult_chess.core.types import Color, Program, State
from simult_chess.net.commitment import commitment_hash
from simult_chess.net.handshake import Handshake, state_hash
from simult_chess.net.protocol import (
    PROTOCOL_VERSION,
    SPEC_VERSION,
    ProtocolError,
)
from simult_chess.net.session import (
    AbortDecision,
    AcceptDrawDecision,
    Decision,
    OnlineMatchResult,
    PlayDecision,
    ResignDecision,
    agent_decider,
    run_online_match,
)
from simult_chess.net.transport import connect_peer, host_peer
from simult_chess.referee.setup import standard_starting_state
from simult_chess.rules.ruleset import RuleSet

RULESET = RuleSet()
_TEST_PORT = 18771
MAX_PHASES = 40
FAST = {
    "transport_timeout": 2.0,
    "keepalive_interval": 0.02,
    "liveness_deadline": 0.5,
    "max_phases": MAX_PHASES,
}


def _both(
    decider_white: object,
    decider_black: object,
    *,
    max_phases: int = MAX_PHASES,
    **kwargs: float,
) -> tuple[OnlineMatchResult, OnlineMatchResult]:
    """Run two full `run_online_match` coroutines against each other."""

    async def scenario() -> tuple[OnlineMatchResult, OnlineMatchResult]:
        peer_a, peer_b = memory_peer_pair()
        state = standard_starting_state()
        opts = {**FAST, "max_phases": max_phases, **kwargs}

        async def side(peer: MemoryPeer, color: Color, decider: object, seed: int):
            return await run_online_match(
                state, RULESET, color, decider, peer, random.Random(seed),  # type: ignore[arg-type]
                print_fn=lambda _line: None, **opts,  # type: ignore[arg-type]
            )

        return await asyncio.gather(
            side(peer_a, Color.WHITE, decider_white, 1),
            side(peer_b, Color.BLACK, decider_black, 2),
        )

    return asyncio.run(scenario())


# --- a full game --------------------------------------------------------------


def test_full_agent_game_agrees_on_both_peers() -> None:
    white, black = _both(
        agent_decider(random_legal_program), agent_decider(random_legal_program)
    )
    assert white.outcome == black.outcome
    assert white.termination_reason == black.termination_reason
    assert white.final_state.board == black.final_state.board


def test_full_agent_game_over_real_tcp() -> None:
    """The production `Peer` (loopback TCP), not just the in-memory fake."""

    async def scenario() -> None:
        state = standard_starting_state()

        async def host_side() -> OnlineMatchResult:
            peer, _bound = await host_peer(_TEST_PORT, host="127.0.0.1")
            try:
                return await run_online_match(
                    state, RULESET, Color.WHITE,
                    agent_decider(random_legal_program), peer, random.Random(1),
                    print_fn=lambda _line: None, max_phases=6,
                )
            finally:
                await peer.close()

        async def client_side() -> OnlineMatchResult:
            for _ in range(200):
                try:
                    peer = await connect_peer("127.0.0.1", _TEST_PORT)
                    break
                except OSError:
                    await asyncio.sleep(0.01)
            else:
                raise AssertionError("could not connect")
            try:
                return await run_online_match(
                    state, RULESET, Color.BLACK,
                    agent_decider(random_legal_program), peer, random.Random(2),
                    print_fn=lambda _line: None, max_phases=6,
                )
            finally:
                await peer.close()

        host_result, client_result = await asyncio.gather(host_side(), client_side())
        assert host_result.outcome == client_result.outcome
        assert host_result.termination_reason == client_result.termination_reason
        assert host_result.final_state.board == client_result.final_state.board

    asyncio.run(scenario())


# --- match services: resignation and draw agreement ---------------------------


def _resign(*_a: object, **_k: object) -> Decision:
    return ResignDecision()


def test_resignation_forfeits_the_resigner_and_both_agree() -> None:
    white, black = _both(_resign, agent_decider(random_legal_program))
    assert white.outcome == "black_wins"
    assert black.outcome == "black_wins"
    assert white.termination_reason == "resignation"
    assert black.termination_reason == "resignation"


def test_abort_ends_in_a_no_result_draw() -> None:
    def _abort(*_a: object, **_k: object) -> Decision:
        return AbortDecision()

    white, black = _both(_abort, agent_decider(random_legal_program))
    assert white.outcome == "draw" and black.outcome == "draw"
    assert white.termination_reason == "abort"
    assert black.termination_reason == "abort"


def test_draw_offer_then_accept_round_trip() -> None:
    """White offers on phase 0 (riding its commit); Black accepts on phase 1."""

    def white_offer_then_play(
        state: State, color: Color, ruleset: RuleSet, rng: random.Random, *,
        peer_offered_draw: bool,
    ) -> Decision:
        program = random_legal_program(state, color, ruleset, rng)
        return PlayDecision(program, offer_draw=state.bookkeeping.phase_index == 0)

    def black_accept_when_offered(
        state: State, color: Color, ruleset: RuleSet, rng: random.Random, *,
        peer_offered_draw: bool,
    ) -> Decision:
        if peer_offered_draw:
            return AcceptDrawDecision()
        return PlayDecision(random_legal_program(state, color, ruleset, rng))

    white, black = _both(white_offer_then_play, black_accept_when_offered)
    assert white.outcome == "draw" and black.outcome == "draw"
    assert white.termination_reason == "draw_agreement"
    assert black.termination_reason == "draw_agreement"
    # It ended by agreement at phase 1, not by running to a natural terminal.
    assert white.final_state.bookkeeping.phase_index == 1


# --- B1: survives a long think without a false transport timeout ---------------


def test_survives_a_think_longer_than_the_transport_timeout() -> None:
    """A slow decider must not trip the transport timeout — keepalive covers it."""

    def slow_white(
        state: State, color: Color, ruleset: RuleSet, rng: random.Random, *,
        peer_offered_draw: bool,
    ) -> Decision:
        if state.bookkeeping.phase_index == 0:
            time.sleep(0.25)  # off the event loop (to_thread); > transport_timeout
        return PlayDecision(random_legal_program(state, color, ruleset, rng))

    white, black = _both(
        slow_white,
        agent_decider(random_legal_program),
        transport_timeout=0.1,
        keepalive_interval=0.02,
        liveness_deadline=2.0,
        max_phases=3,
    )
    assert white.outcome == black.outcome
    assert white.final_state.board == black.final_state.board


# --- B5: keepalive detects a silent peer --------------------------------------


def test_keepalive_detects_a_dead_peer() -> None:
    """A peer that handshakes then goes silent is caught within the deadline."""

    async def scenario() -> None:
        peer_a, peer_b = memory_peer_pair()
        state = standard_starting_state()

        async def silent_peer() -> None:
            await peer_b.recv()  # A's handshake
            remote = Handshake(
                protocol_version=PROTOCOL_VERSION,
                spec_version=SPEC_VERSION,
                ruleset_fingerprint=RULESET.fingerprint(),
                initial_state_hash=state_hash(state),
                max_phases=MAX_PHASES,
                proposed_color=Color.BLACK,
                nonce=123,
            )
            await peer_b.send(remote.to_message())
            # ... then never respond again: a kill -STOP partition.

        async def honest() -> OnlineMatchResult:
            return await run_online_match(
                state, RULESET, Color.WHITE,
                agent_decider(random_legal_program), peer_a, random.Random(1),
                print_fn=lambda _line: None,
                transport_timeout=2.0, keepalive_interval=0.02,
                liveness_deadline=0.3, max_phases=MAX_PHASES,
            )

        start = time.monotonic()
        with pytest.raises(ProtocolError, match="unresponsive"):
            await asyncio.gather(honest(), silent_peer())
        assert time.monotonic() - start < 3.0  # detected, not hung

    asyncio.run(scenario())


# --- B4: an illegal revealed program forfeits its sender ----------------------


def test_illegal_remote_program_forfeits_the_sender() -> None:
    """A peer that reveals an empty (L1-illegal) program loses; we win."""

    async def scenario() -> None:
        peer_a, peer_b = memory_peer_pair()
        state = standard_starting_state()

        async def cheater() -> None:
            await peer_b.recv()  # A's handshake
            await peer_b.send(
                Handshake(
                    protocol_version=PROTOCOL_VERSION,
                    spec_version=SPEC_VERSION,
                    ruleset_fingerprint=RULESET.fingerprint(),
                    initial_state_hash=state_hash(state),
                    max_phases=MAX_PHASES,
                    proposed_color=Color.BLACK,
                    nonce=7,
                ).to_message()
            )
            illegal: Program = ()  # empty program violates L1
            salt = b"saltsaltsaltsalt"
            await peer_b.send(
                {
                    "type": "commit",
                    "phase_index": 0,
                    "hash": commitment_hash(salt, []),
                    "offer_draw": False,
                }
            )
            await peer_b.send(
                {
                    "type": "reveal",
                    "phase_index": 0,
                    "salt": salt.hex(),
                    "program": [],
                }
            )
            del illegal

        async def honest() -> OnlineMatchResult:
            return await run_online_match(
                state, RULESET, Color.WHITE,
                agent_decider(random_legal_program), peer_a, random.Random(1),
                print_fn=lambda _line: None, **FAST,
            )

        result, _ = await asyncio.gather(honest(), cheater())
        assert result.outcome == "white_wins"
        assert result.termination_reason == "illegal_program"

    asyncio.run(scenario())


# --- commitment integrity (carried over from Phase 8) -------------------------


def test_tampered_reveal_is_detected() -> None:
    """A reveal not matching the earlier commitment is caught, not accepted."""

    async def scenario() -> None:
        peer_a, peer_b = memory_peer_pair()
        state = standard_starting_state()

        async def cheater() -> None:
            await peer_b.recv()  # A's handshake
            await peer_b.send(
                Handshake(
                    protocol_version=PROTOCOL_VERSION,
                    spec_version=SPEC_VERSION,
                    ruleset_fingerprint=RULESET.fingerprint(),
                    initial_state_hash=state_hash(state),
                    max_phases=MAX_PHASES,
                    proposed_color=Color.BLACK,
                    nonce=7,
                ).to_message()
            )
            await peer_b.send(
                {
                    "type": "commit",
                    "phase_index": 0,
                    "hash": commitment_hash(b"x", []),
                    "offer_draw": False,
                }
            )
            await peer_b.send(
                {
                    "type": "reveal",
                    "phase_index": 0,
                    "salt": "ff",
                    "program": [{"kind": "castle", "side": "king"}],
                }
            )

        async def honest() -> OnlineMatchResult:
            return await run_online_match(
                state, RULESET, Color.WHITE,
                agent_decider(random_legal_program), peer_a, random.Random(1),
                print_fn=lambda _line: None, **FAST,
            )

        with pytest.raises(ProtocolError, match="does not match its commitment"):
            await asyncio.gather(honest(), cheater())

    asyncio.run(scenario())
