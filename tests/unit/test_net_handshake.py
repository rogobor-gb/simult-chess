"""Phase 15a handshake: agreement checks and colour negotiation (fixes B3)."""

from __future__ import annotations

import asyncio
import random

import pytest
from _memory_transport import MemoryPeer, memory_peer_pair

from simult_chess.core.types import Color
from simult_chess.net.handshake import (
    ColorNegotiationError,
    Handshake,
    InitialStateMismatch,
    MaxPhasesMismatch,
    ProtocolVersionMismatch,
    RulesetMismatch,
    SpecVersionMismatch,
    _negotiate_color,
    perform_handshake,
    state_hash,
)
from simult_chess.net.protocol import PROTOCOL_VERSION, SPEC_VERSION
from simult_chess.referee.setup import standard_starting_state
from simult_chess.rules.ruleset import RuleSet
from simult_chess.rules.variants import get_variant

RULESET = RuleSet()
MAX_PHASES = 40


def _handshake(
    *,
    color: Color,
    nonce: int,
    ruleset: RuleSet = RULESET,
    protocol_version: int = PROTOCOL_VERSION,
    spec_version: str = SPEC_VERSION,
    initial_state_hash: str | None = None,
    max_phases: int = MAX_PHASES,
) -> Handshake:
    state = standard_starting_state()
    return Handshake(
        protocol_version=protocol_version,
        spec_version=spec_version,
        ruleset_fingerprint=ruleset.fingerprint(),
        initial_state_hash=initial_state_hash or state_hash(state),
        max_phases=max_phases,
        proposed_color=color,
        nonce=nonce,
    )


# --- colour negotiation (pure) ------------------------------------------------


def test_opposite_proposals_are_honoured() -> None:
    local = _handshake(color=Color.WHITE, nonce=1)
    remote = _handshake(color=Color.BLACK, nonce=2)
    assert _negotiate_color(local, remote) is Color.WHITE


def test_colliding_proposals_break_by_nonce_and_agree() -> None:
    """Both peers wanted white; the higher nonce keeps it, both sides concur."""
    high = _handshake(color=Color.WHITE, nonce=999)
    low = _handshake(color=Color.WHITE, nonce=7)
    # From the high-nonce peer's view it is `local`; from the low-nonce peer's
    # view the roles swap. The two computed colours must be opposite.
    assert _negotiate_color(high, low) is Color.WHITE
    assert _negotiate_color(low, high) is Color.BLACK


def test_equal_nonce_collision_aborts() -> None:
    same = _handshake(color=Color.BLACK, nonce=42)
    other = _handshake(color=Color.BLACK, nonce=42)
    with pytest.raises(ColorNegotiationError, match="colour negotiation tie"):
        _negotiate_color(same, other)


# --- end-to-end over the wire -------------------------------------------------


def _run_against(local_color: Color, remote: Handshake) -> Color:
    """Drive one `perform_handshake` against a hand-crafted opposing message."""

    async def scenario() -> Color:
        peer_a, peer_b = memory_peer_pair()

        async def opposing() -> None:
            await peer_b.recv()  # consume A's handshake
            await peer_b.send(remote.to_message())

        result, _ = await asyncio.gather(
            perform_handshake(
                peer_a,
                ruleset=RULESET,
                initial_state=standard_starting_state(),
                proposed_color=local_color,
                max_phases=MAX_PHASES,
                rng=random.Random(0),
                transport_timeout=1.0,
            ),
            opposing(),
        )
        return result

    return asyncio.run(scenario())


def test_successful_handshake_returns_the_proposed_colour() -> None:
    assert _run_against(Color.WHITE, _handshake(color=Color.BLACK, nonce=5)) is (
        Color.WHITE
    )


def test_protocol_version_mismatch_is_named() -> None:
    with pytest.raises(ProtocolVersionMismatch, match="protocol version"):
        _run_against(
            Color.WHITE,
            _handshake(
                color=Color.BLACK, nonce=5, protocol_version=PROTOCOL_VERSION + 1
            ),
        )


def test_spec_version_mismatch_is_named() -> None:
    with pytest.raises(SpecVersionMismatch, match="spec version"):
        _run_against(
            Color.WHITE, _handshake(color=Color.BLACK, nonce=5, spec_version="9.9")
        )


def test_ruleset_fingerprint_mismatch_is_named() -> None:
    with pytest.raises(RulesetMismatch, match="fingerprint"):
        _run_against(
            Color.WHITE,
            _handshake(color=Color.BLACK, nonce=5, ruleset=get_variant("horizon_30")),
        )


def test_initial_state_mismatch_is_named() -> None:
    with pytest.raises(InitialStateMismatch, match="different"):
        _run_against(
            Color.WHITE,
            _handshake(color=Color.BLACK, nonce=5, initial_state_hash="deadbeef"),
        )


def test_max_phases_mismatch_is_named() -> None:
    with pytest.raises(MaxPhasesMismatch, match="max_phases"):
        _run_against(
            Color.WHITE, _handshake(color=Color.BLACK, nonce=5, max_phases=999)
        )


def test_two_peers_negotiate_a_colliding_proposal_end_to_end() -> None:
    """Two ``--color white`` peers get a playable assignment, not a crash (B3)."""

    async def scenario() -> tuple[Color, Color]:
        peer_a, peer_b = memory_peer_pair()
        state = standard_starting_state()

        async def side(peer: MemoryPeer, seed: int) -> Color:
            return await perform_handshake(
                peer,
                ruleset=RULESET,
                initial_state=state,
                proposed_color=Color.WHITE,
                max_phases=MAX_PHASES,
                rng=random.Random(seed),
                transport_timeout=1.0,
            )

        return await asyncio.gather(side(peer_a, 1), side(peer_b, 2))

    color_a, color_b = asyncio.run(scenario())
    assert {color_a, color_b} == {Color.WHITE, Color.BLACK}
