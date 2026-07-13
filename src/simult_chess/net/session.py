"""The online phase loop: commit -> reveal -> resolve over a `Peer` (Phase 8).

Networking is a *transport* for the same commit-reveal contract Phases 6-7
already route through (`referee/observe.py`), not new game logic: each
phase, both peers exchange a salted commitment hash, then the salt and
program, verify the reveal against the commitment, resolve `phi` locally
(identically on both sides, since it is pure), and finally exchange a hash
of the resulting public position as a cheap divergence check -- the online
analogue of the DoD's "post-phase event logs are byte-identical."
"""

from __future__ import annotations

import hashlib
import os
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from simult_chess.agents.base import Agent
from simult_chess.core.phi import phi
from simult_chess.core.types import Color, State
from simult_chess.net.commitment import commitment_hash
from simult_chess.net.protocol import (
    ProtocolError,
    deserialize_program,
    serialize_program,
)
from simult_chess.net.transport import Peer
from simult_chess.referee.serialize import public_position_key
from simult_chess.rules.ruleset import RuleSet

OnlineOutcome = Literal[
    "ongoing", "white_wins", "black_wins", "draw", "phase_limit_reached"
]

PrintFn = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class OnlineMatchResult:
    """The outcome of one complete online match, from this peer's view."""

    final_state: State
    outcome: OnlineOutcome


def _state_hash(state: State) -> str:
    return hashlib.sha256(repr(public_position_key(state)).encode()).hexdigest()


def _check_envelope(message: dict[str, object], expected_type: str, phase: int) -> None:
    if message.get("type") != expected_type:
        detail = f"expected a {expected_type!r} message, got {message.get('type')!r}"
        raise ProtocolError(detail)
    if message.get("phase_index") != phase:
        got = message.get("phase_index")
        raise ProtocolError(f"phase index mismatch: expected {phase}, got {got}")


async def run_online_match(
    initial_state: State,
    ruleset: RuleSet,
    local_color: Color,
    program_source: Agent,
    peer: Peer,
    rng: random.Random,
    *,
    message_timeout: float = 30.0,
    max_phases: int = 500,
    print_fn: PrintFn = print,
) -> OnlineMatchResult:
    """Play a full game against `peer`, deciding `local_color`'s programs
    via `program_source` (an `Agent`, or a human-prompting adapter of the
    same shape).

    Raises `ProtocolError` if a peer's reveal doesn't match its earlier
    commitment (a dropped/garbled commitment), if a message arrives out of
    sequence, or if the two peers' resolved states diverge.
    """
    remote_color = local_color.opponent
    state = initial_state
    outcome: OnlineOutcome = "ongoing"

    for _ in range(max_phases):
        phase = state.bookkeeping.phase_index
        local_program = program_source(state, local_color, ruleset, rng)
        local_json = serialize_program(local_program, state, local_color)
        salt = os.urandom(16)
        local_commitment = commitment_hash(salt, local_json)

        commit_msg = {"type": "commit", "phase_index": phase, "hash": local_commitment}
        await peer.send(commit_msg)
        remote_commit = await peer.recv(timeout=message_timeout)
        _check_envelope(remote_commit, "commit", phase)
        remote_commitment = remote_commit["hash"]

        reveal_msg = {
            "type": "reveal",
            "phase_index": phase,
            "salt": salt.hex(),
            "program": local_json,
        }
        await peer.send(reveal_msg)
        remote_reveal = await peer.recv(timeout=message_timeout)
        _check_envelope(remote_reveal, "reveal", phase)
        remote_salt = bytes.fromhex(remote_reveal["salt"])
        remote_json = remote_reveal["program"]
        if commitment_hash(remote_salt, remote_json) != remote_commitment:
            detail = f"phase {phase}: peer's reveal does not match its commitment"
            raise ProtocolError(detail)
        remote_program = deserialize_program(remote_json, state, remote_color)

        programs = {local_color: local_program, remote_color: remote_program}
        result = phi(state, programs[Color.WHITE], programs[Color.BLACK], ruleset)
        local_hash = _state_hash(result.state)

        await peer.send({"type": "ack", "phase_index": phase, "state_hash": local_hash})
        remote_ack = await peer.recv(timeout=message_timeout)
        _check_envelope(remote_ack, "ack", phase)
        if remote_ack["state_hash"] != local_hash:
            detail = f"phase {phase}: post-phase state diverged from peer"
            raise ProtocolError(detail)

        print_fn(f"phase {phase} resolved: {result.outcome}")
        state = result.state
        outcome = result.outcome
        if outcome != "ongoing":
            break
    else:
        outcome = "phase_limit_reached"

    return OnlineMatchResult(final_state=state, outcome=outcome)
