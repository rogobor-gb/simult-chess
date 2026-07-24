"""Pre-match handshake and colour negotiation (Phase 15a, fixes audit B3).

Before phase 0, each peer sends exactly one ``handshake`` message and reads
the other's. The exchange does two jobs:

1. **Agreement.** ``protocol_version``, ``spec_version``,
   ``ruleset_fingerprint`` (Phase 14), ``initial_state_hash``, and
   ``max_phases`` must match on both sides. Any mismatch aborts *before* phase
   0 with its own named error — the old engine asserted colours independently
   and surfaced a disagreement as a bare ``ValueError`` out of Φ (B3); here a
   fingerprint or spec-version mismatch cannot even reach the phase loop.

2. **Colour negotiation.** Colour is *proposed*, not asserted. If the two
   proposals differ they are accepted as-is; if they collide, a random
   tiebreak nonce (also exchanged) assigns colours deterministically and
   identically on both peers, so two ``--color white`` peers get a playable
   game instead of a crash. An astronomically unlikely equal-nonce collision
   is the one case that aborts.

This module runs Φ never and touches no rules; it only compares hashes and
strings, so it stays a pure transport concern.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

from simult_chess.core.types import Color, State
from simult_chess.net.protocol import (
    MSG_HANDSHAKE,
    PROTOCOL_VERSION,
    SPEC_VERSION,
    ProtocolError,
)
from simult_chess.net.transport import Transport
from simult_chess.referee.serialize import public_position_key
from simult_chess.rules.ruleset import RuleSet

_NONCE_BITS = 64


class HandshakeError(ProtocolError):
    """A handshake field disagreed, or colour could not be assigned.

    Base class for every named pre-match abort so a caller can catch the whole
    family, while each subclass names the specific field that disagreed.
    """


class ProtocolVersionMismatch(HandshakeError):
    """Peers run incompatible wire-protocol versions."""


class SpecVersionMismatch(HandshakeError):
    """Peers implement different rules specifications."""


class RulesetMismatch(HandshakeError):
    """Peers proposed different ``RuleSet`` fingerprints (Phase 14)."""


class InitialStateMismatch(HandshakeError):
    """Peers start from different positions."""


class MaxPhasesMismatch(HandshakeError):
    """Peers disagree on the phase limit."""


class ColorNegotiationError(HandshakeError):
    """Colour could not be assigned (equal proposals, equal nonces)."""


def state_hash(state: State) -> str:
    """Hex SHA-256 of a state's public position key (board + cooldown).

    Shared by the handshake (initial-state agreement) and the per-phase
    divergence check, so both sides hash a state the same way.
    """
    return hashlib.sha256(repr(public_position_key(state)).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class Handshake:
    """One peer's opening proposal.

    Parameters
    ----------
    protocol_version, spec_version : int, str
        Wire and rules versions; must match the peer's.
    ruleset_fingerprint : str
        ``RuleSet.fingerprint()`` (Phase 14) — the rule *values* both sides
        will run Φ under.
    initial_state_hash : str
        :func:`state_hash` of the agreed starting position.
    max_phases : int
        Phase-limit draw bound for the match (session metadata, not a rule).
    proposed_color : Color
        The colour this peer would like to play.
    nonce : int
        Uniform random tiebreak, used only if both peers propose the same
        colour.
    """

    protocol_version: int
    spec_version: str
    ruleset_fingerprint: str
    initial_state_hash: str
    max_phases: int
    proposed_color: Color
    nonce: int

    def to_message(self) -> dict[str, Any]:
        """Render to a JSON-safe ``handshake`` message."""
        return {
            "type": MSG_HANDSHAKE,
            "protocol_version": self.protocol_version,
            "spec_version": self.spec_version,
            "ruleset_fingerprint": self.ruleset_fingerprint,
            "initial_state_hash": self.initial_state_hash,
            "max_phases": self.max_phases,
            "proposed_color": self.proposed_color.value,
            "nonce": self.nonce,
        }

    @staticmethod
    def from_message(message: dict[str, Any]) -> Handshake:
        """Parse a received ``handshake`` message, or raise ``ProtocolError``."""
        if message.get("type") != MSG_HANDSHAKE:
            raise ProtocolError(
                f"expected a {MSG_HANDSHAKE!r} message, got {message.get('type')!r}"
            )
        try:
            return Handshake(
                protocol_version=int(message["protocol_version"]),
                spec_version=str(message["spec_version"]),
                ruleset_fingerprint=str(message["ruleset_fingerprint"]),
                initial_state_hash=str(message["initial_state_hash"]),
                max_phases=int(message["max_phases"]),
                proposed_color=Color(message["proposed_color"]),
                nonce=int(message["nonce"]),
            )
        except (KeyError, ValueError) as exc:
            raise ProtocolError(f"malformed handshake message: {exc}") from exc


def _negotiate_color(local: Handshake, remote: Handshake) -> Color:
    """Assign this peer's colour from the two proposals; identical on both sides."""
    if local.proposed_color is not remote.proposed_color:
        # Proposals already disagree, i.e. name opposite colours: honour them.
        return local.proposed_color
    if local.nonce == remote.nonce:
        raise ColorNegotiationError(
            "colour negotiation tie: both peers proposed "
            f"{local.proposed_color.value!r} with the same nonce; retry"
        )
    # Same colour wanted: the higher nonce keeps it, the other flips. Both
    # peers see the same pair of (colour, nonce) proposals, so they agree.
    local_keeps = local.nonce > remote.nonce
    return local.proposed_color if local_keeps else local.proposed_color.opponent


async def perform_handshake(
    peer: Transport,
    *,
    ruleset: RuleSet,
    initial_state: State,
    proposed_color: Color,
    max_phases: int,
    rng: random.Random,
    transport_timeout: float,
) -> Color:
    """Exchange handshakes with `peer`, validate agreement, negotiate colour.

    Returns the colour this peer will play. Raises a :class:`HandshakeError`
    subclass, named for the disagreeing field, on any mismatch — before the
    phase loop begins.

    Parameters
    ----------
    peer : Transport
        The message channel.
    ruleset : RuleSet
        The rules; its fingerprint is proposed and cross-checked.
    initial_state : State
        The agreed starting position; its :func:`state_hash` is cross-checked.
    proposed_color : Color
        The colour this peer requests (from ``--color``).
    max_phases : int
        Proposed phase limit; must match the peer's.
    rng : random.Random
        Source of the tiebreak nonce.
    transport_timeout : float
        Liveness bound on receiving the peer's handshake — a handshake is not
        a decision, so it is a transport concern, not decision time.
    """
    local = Handshake(
        protocol_version=PROTOCOL_VERSION,
        spec_version=SPEC_VERSION,
        ruleset_fingerprint=ruleset.fingerprint(),
        initial_state_hash=state_hash(initial_state),
        max_phases=max_phases,
        proposed_color=proposed_color,
        nonce=rng.getrandbits(_NONCE_BITS),
    )
    await peer.send(local.to_message())
    remote = Handshake.from_message(await peer.recv(timeout=transport_timeout))

    if remote.protocol_version != local.protocol_version:
        raise ProtocolVersionMismatch(
            f"protocol version mismatch: local {local.protocol_version}, "
            f"peer {remote.protocol_version}"
        )
    if remote.spec_version != local.spec_version:
        raise SpecVersionMismatch(
            f"spec version mismatch: local {local.spec_version!r}, "
            f"peer {remote.spec_version!r}"
        )
    if remote.ruleset_fingerprint != local.ruleset_fingerprint:
        raise RulesetMismatch(
            "ruleset fingerprint mismatch: local "
            f"{local.ruleset_fingerprint[:12]}…, peer "
            f"{remote.ruleset_fingerprint[:12]}… — are you on the same "
            "--variant?"
        )
    if remote.initial_state_hash != local.initial_state_hash:
        raise InitialStateMismatch(
            "initial state mismatch: peers are starting from different "
            "positions"
        )
    if remote.max_phases != local.max_phases:
        raise MaxPhasesMismatch(
            f"max_phases mismatch: local {local.max_phases}, "
            f"peer {remote.max_phases}"
        )
    return _negotiate_color(local, remote)
