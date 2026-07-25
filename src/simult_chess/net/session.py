"""The online phase loop: handshake -> (commit -> reveal -> resolve)* (Phase 8,
hardened in Phase 15a).

Networking is a *transport* for the same commit-reveal contract Phases 6-7
route through (`referee/observe.py`), not new game logic: each phase, both
peers exchange a salted commitment, then the salt and program, verify the
reveal against the commitment, resolve `phi` locally (identically on both
sides, since it is pure), and exchange a hash of the resulting public position
as a divergence check.

Phase 15a makes that loop survive a real online game between two humans:

- **Handshake** (`net/handshake.py`) agrees versions, the Phase-14 ruleset
  fingerprint, the initial position, and a negotiated colour before phase 0
  (fixes B3).
- **Timeout separation** (B1): *transport* liveness (a message the peer should
  already have sent) is bounded by ``transport_timeout``; *decision* time (how
  long the peer thinks before committing) is unbounded here — governed by the
  Phase-15b clock when there is one — and kept honest instead by a keepalive.
- **Non-blocking entry** (B2): the local decision runs off the event loop via
  ``asyncio.to_thread``, concurrently with the keepalive, so a human thinking
  never freezes the ping/pong that proves this peer is still alive.
- **Keepalive** (B5): while waiting for the peer's commit, ping/pong runs with
  a ``liveness_deadline`` independent of decision time, so a silent partition
  (``kill -STOP``) is detected instead of hanging forever.
- **Illegal-remote-program adjudication** (B4): a revealed program that fails
  ``L`` forfeits its sender (ruling of 2026-07-24), naming the L-clause.
- **Match services** (B6): resign, draw offer/accept, and abort, encoded as
  the per-phase *action slot* (`net/protocol.py`).

**Outcome refactor.** The competitive outcome stays exactly
``{white_wins, black_wins, draw}`` (payoff ``u ∈ {-1,0,+1}``, spec §10); a
separate ``termination_reason`` says *why* the game ended. A resignation or a
flag-fall is not a property of Φ and must not enter ``State`` or the payoff
functional — inv M1 (no wall-clock in the operator) stays literally true.
"""

from __future__ import annotations

import asyncio
import os
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from simult_chess.agents.base import Agent
from simult_chess.core.legality import check_legal_program
from simult_chess.core.phi import phi
from simult_chess.core.stages.closure import detect_terminal
from simult_chess.core.types import Color, Program, State
from simult_chess.net.commitment import commitment_hash
from simult_chess.net.handshake import perform_handshake, state_hash
from simult_chess.net.protocol import (
    ACTION_SLOT_TYPES,
    MSG_ABORT,
    MSG_ACCEPT_DRAW,
    MSG_ACK,
    MSG_COMMIT,
    MSG_PING,
    MSG_PONG,
    MSG_RESIGN,
    MSG_REVEAL,
    ProtocolError,
    TransportTimeout,
    deserialize_program,
    serialize_program,
)
from simult_chess.net.transport import Transport
from simult_chess.referee.record import GamePhase
from simult_chess.referee.serialize import public_position_key
from simult_chess.rules.ruleset import RuleSet

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
"""Why a match ended. A superset of the roadmap's list by one entry,
``draw_agreement``: implementing the required draw-offer service (B6) surfaced
that a consensually agreed draw has no reason among ``{regicide, repetition,
no_progress, resignation, timeout, illegal_program, abort, phase_limit}`` — it
is not a horizon draw, a repetition, or a no-contest abort. It carries payoff
0 like any draw; the added reason is session metadata only, so inv M1 is
untouched.
"""

PrintFn = Callable[[str], None]


# ---------------------------------------------------------------------------
# Local decisions (what this peer does with its action slot each phase)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlayDecision:
    """Commit a program; optionally attach a standing draw offer."""

    program: Program
    offer_draw: bool = False


@dataclass(frozen=True, slots=True)
class ResignDecision:
    """Concede the game."""


@dataclass(frozen=True, slots=True)
class AbortDecision:
    """Void the game with no result (a draw payoff, reason ``abort``)."""


@dataclass(frozen=True, slots=True)
class AcceptDrawDecision:
    """Accept the peer's standing draw offer."""


Decision = PlayDecision | ResignDecision | AbortDecision | AcceptDrawDecision


class LocalDecider(Protocol):
    """What this peer decides each phase: a program or a match-service action.

    ``peer_offered_draw`` tells a human whether an :class:`AcceptDrawDecision`
    is available this phase; agents ignore it.
    """

    def __call__(
        self,
        state: State,
        color: Color,
        ruleset: RuleSet,
        rng: random.Random,
        *,
        peer_offered_draw: bool,
    ) -> Decision: ...


def agent_decider(agent: Agent) -> LocalDecider:
    """Adapt an `Agent` (which only ever produces a program) to a decider."""

    def decide(
        state: State,
        color: Color,
        ruleset: RuleSet,
        rng: random.Random,
        *,
        peer_offered_draw: bool,
    ) -> Decision:
        return PlayDecision(agent(state, color, ruleset, rng))

    return decide


@dataclass(frozen=True, slots=True)
class OnlineMatchResult:
    """The outcome of one complete online match, from this peer's view.

    ``outcome`` is the competitive result (payoff domain, spec §10);
    ``termination_reason`` is why the game ended (session metadata); ``phases``
    are the resolved phases with their programs, ready to write to a `.scn`
    game record (Phase 15d) — empty on a game that ended before any phase
    resolved (an immediate resign/abort).
    """

    final_state: State
    outcome: MatchOutcome
    termination_reason: TerminationReason
    phases: tuple[GamePhase, ...] = ()


def _color_wins(color: Color) -> MatchOutcome:
    return "white_wins" if color is Color.WHITE else "black_wins"


def _terminal_reason(result_state: State, ruleset: RuleSet) -> TerminationReason:
    """Attribute a resolved terminal to a `termination_reason` (Φ collapses draws).

    Mirrors `phi`'s own draw-cause precedence (mutual king loss, then
    repetition, then the T4 horizon), reading it back off the successor state;
    a decisive game is always a king capture (regicide).
    """
    if detect_terminal(result_state.board) != "draw":
        return "regicide"
    if not any(
        token.typ == "k" for token in result_state.board
    ):  # both kings gone: still death by capture
        return "regicide"
    position_key = public_position_key(result_state)
    if result_state.bookkeeping.repetition_ledger.get(position_key, 0) >= 3:
        return "repetition"
    return "no_progress"


@dataclass(frozen=True, slots=True)
class _Terminal:
    """A slot resolution that ends the match."""

    outcome: MatchOutcome
    reason: TerminationReason


def _resolve_slots(
    local: Decision,
    remote_slot: dict[str, Any],
    local_color: Color,
    *,
    local_offer_standing: bool,
    peer_offer_standing: bool,
) -> _Terminal | None:
    """Adjudicate the two peers' action slots; ``None`` means "both committed".

    Every branch is a pure function of ``(local kind, remote kind)`` and the
    two standing-offer flags — all symmetric across peers — so both sides reach
    the identical verdict without an arbiter.
    """
    remote_kind = remote_slot.get("type")
    if remote_kind not in ACTION_SLOT_TYPES:
        raise ProtocolError(f"expected an action slot, got {remote_kind!r}")
    local_abort = isinstance(local, AbortDecision)
    local_resign = isinstance(local, ResignDecision)
    local_accept = isinstance(local, AcceptDrawDecision)

    if local_abort or remote_kind == MSG_ABORT:
        return _Terminal("draw", "abort")
    if local_resign and remote_kind == MSG_RESIGN:
        return _Terminal("draw", "resignation")
    if local_resign:
        return _Terminal(_color_wins(local_color.opponent), "resignation")
    if remote_kind == MSG_RESIGN:
        return _Terminal(_color_wins(local_color), "resignation")
    if local_accept:
        if not peer_offer_standing:
            raise ProtocolError("accepted a draw the peer had not offered")
        return _Terminal("draw", "draw_agreement")
    if remote_kind == MSG_ACCEPT_DRAW:
        if not local_offer_standing:
            raise ProtocolError("peer accepted a draw we had not offered")
        return _Terminal("draw", "draw_agreement")
    return None  # both committed a program: proceed to reveal


async def _keepalive_until_slot(
    peer: Transport,
    *,
    phase: int,
    keepalive_interval: float,
    liveness_deadline: float,
) -> dict[str, Any]:
    """Wait for the peer's action slot, pinging while it thinks (B5).

    Any message — ping, pong, or the slot itself — proves the peer is alive and
    resets the liveness window; a full ``liveness_deadline`` of total silence
    (a partition, a ``kill -STOP``) raises instead of hanging. Decision time is
    otherwise unbounded (B1): a peer may think as long as it likes.
    """
    loop = asyncio.get_running_loop()
    deadline_at = loop.time() + liveness_deadline
    next_ping_at = 0.0
    while True:
        now = loop.time()
        if now >= deadline_at:
            raise ProtocolError(
                f"peer unresponsive: no message for {liveness_deadline}s "
                f"(phase {phase})"
            )
        if now >= next_ping_at:
            await peer.send({"type": MSG_PING, "phase_index": phase})
            next_ping_at = now + keepalive_interval
        wait = min(next_ping_at - now, deadline_at - now)
        try:
            message = await peer.recv(timeout=max(wait, 0.0))
        except TransportTimeout:
            continue
        deadline_at = loop.time() + liveness_deadline
        mtype = message.get("type")
        if mtype == MSG_PING:
            await peer.send({"type": MSG_PONG, "phase_index": phase})
        elif mtype == MSG_PONG:
            continue
        elif mtype in ACTION_SLOT_TYPES:
            return message
        else:
            raise ProtocolError(f"unexpected {mtype!r} while awaiting peer's action")


@dataclass(frozen=True, slots=True)
class _LocalCommit:
    """Artifacts of a committed program, kept for the reveal step."""

    program: Program
    salt: bytes
    program_json: list[dict[str, Any]]


async def _think_and_send_slot(
    peer: Transport,
    decider: LocalDecider,
    state: State,
    local_color: Color,
    ruleset: RuleSet,
    rng: random.Random,
    *,
    phase: int,
    peer_offered_draw: bool,
) -> tuple[Decision, _LocalCommit | None]:
    """Compute this peer's decision off the event loop, then send its slot (B2)."""

    def _decide() -> Decision:
        return decider(
            state, local_color, ruleset, rng, peer_offered_draw=peer_offered_draw
        )

    decision = await asyncio.to_thread(_decide)

    if isinstance(decision, PlayDecision):
        program_json = serialize_program(decision.program, state, local_color)
        salt = os.urandom(16)
        await peer.send(
            {
                "type": MSG_COMMIT,
                "phase_index": phase,
                "hash": commitment_hash(salt, program_json),
                "offer_draw": decision.offer_draw,
            }
        )
        return decision, _LocalCommit(decision.program, salt, program_json)
    if isinstance(decision, ResignDecision):
        await peer.send({"type": MSG_RESIGN, "phase_index": phase})
    elif isinstance(decision, AbortDecision):
        await peer.send({"type": MSG_ABORT, "phase_index": phase})
    else:  # AcceptDrawDecision
        await peer.send({"type": MSG_ACCEPT_DRAW, "phase_index": phase})
    return decision, None


def _check_envelope(message: dict[str, Any], expected_type: str, phase: int) -> None:
    if message.get("type") != expected_type:
        detail = f"expected a {expected_type!r} message, got {message.get('type')!r}"
        raise ProtocolError(detail)
    if message.get("phase_index") != phase:
        got = message.get("phase_index")
        raise ProtocolError(f"phase index mismatch: expected {phase}, got {got}")


async def _recv_skipping_keepalive(
    peer: Transport, *, phase: int, transport_timeout: float
) -> dict[str, Any]:
    """Receive the next non-keepalive message, answering any pings in between.

    Keepalive traffic from the decision window can trail into the reveal/ack
    exchange (a peer's last ping arriving after its commit); this drains it so
    the immediate reveal/ack recv is not derailed by a stray ``ping``/``pong``.
    """
    while True:
        message = await peer.recv(timeout=transport_timeout)
        mtype = message.get("type")
        if mtype == MSG_PING:
            await peer.send({"type": MSG_PONG, "phase_index": phase})
            continue
        if mtype == MSG_PONG:
            continue
        return message


async def _exchange_reveal_and_resolve(
    peer: Transport,
    state: State,
    local_color: Color,
    ruleset: RuleSet,
    local_commit: _LocalCommit,
    remote_commit: dict[str, Any],
    *,
    phase: int,
    transport_timeout: float,
) -> tuple[State, _Terminal | None, GamePhase | None]:
    """Reveal, verify, adjudicate legality, resolve Φ, and cross-check the state.

    Returns ``(new_state, terminal, phase)``: ``terminal`` is the match outcome
    if the game ended this phase (else ``None``); ``phase`` is the resolved
    phase's record entry, or ``None`` on the illegal-program forfeit path (no
    phase actually resolved). Both the reveal and the ack are messages the peer
    should send *immediately* once both committed, so they are bounded by
    ``transport_timeout`` (decision time is already spent).
    """
    remote_color = local_color.opponent
    await peer.send(
        {
            "type": MSG_REVEAL,
            "phase_index": phase,
            "salt": local_commit.salt.hex(),
            "program": local_commit.program_json,
        }
    )
    remote_reveal = await _recv_skipping_keepalive(
        peer, phase=phase, transport_timeout=transport_timeout
    )
    _check_envelope(remote_reveal, MSG_REVEAL, phase)
    remote_salt = bytes.fromhex(remote_reveal["salt"])
    remote_json = remote_reveal["program"]
    if commitment_hash(remote_salt, remote_json) != remote_commit["hash"]:
        raise ProtocolError(
            f"phase {phase}: peer's reveal does not match its commitment"
        )
    remote_program = deserialize_program(remote_json, state, remote_color)

    # Illegal-remote-program adjudication (B4): both peers validate both
    # revealed programs on identical data, so they agree on who forfeits
    # without ever feeding an illegal program to Φ.
    forfeit = _adjudicate_legality(
        state, local_commit.program, remote_program, local_color, ruleset
    )
    if forfeit is not None:
        return state, forfeit, None

    white, black = _ordered(local_color, local_commit.program, remote_program)
    result = phi(state, white, black, ruleset)
    local_hash = state_hash(result.state)
    await peer.send({"type": MSG_ACK, "phase_index": phase, "state_hash": local_hash})
    remote_ack = await _recv_skipping_keepalive(
        peer, phase=phase, transport_timeout=transport_timeout
    )
    _check_envelope(remote_ack, MSG_ACK, phase)
    if remote_ack["state_hash"] != local_hash:
        raise ProtocolError(f"phase {phase}: post-phase state diverged from peer")

    game_phase = GamePhase(white, black, result.outcome)
    if result.outcome == "ongoing":
        return result.state, None, game_phase
    reason = _terminal_reason(result.state, ruleset)
    return result.state, _Terminal(result.outcome, reason), game_phase


def _ordered(
    local_color: Color, local_program: Program, remote_program: Program
) -> tuple[Program, Program]:
    """Return the two programs as ``(white, black)`` for `phi`."""
    if local_color is Color.WHITE:
        return local_program, remote_program
    return remote_program, local_program


def _adjudicate_legality(
    state: State,
    local_program: Program,
    remote_program: Program,
    local_color: Color,
    ruleset: RuleSet,
) -> _Terminal | None:
    """Forfeit whichever side revealed an illegal program (B4), naming the clause.

    An honest peer never reaches this; it fires on a buggy or malicious peer.
    Both sides run the same pure ``L`` check on the same two programs, so the
    verdict — and thus the forfeit — is identical on both.
    """
    remote_color = local_color.opponent
    local_bad = check_legal_program(state, local_program, local_color, ruleset)
    remote_bad = check_legal_program(state, remote_program, remote_color, ruleset)
    if local_bad and remote_bad:
        return _Terminal("draw", "illegal_program")
    if remote_bad:
        return _Terminal(_color_wins(local_color), "illegal_program")
    if local_bad:
        return _Terminal(_color_wins(remote_color), "illegal_program")
    return None


async def run_online_match(
    initial_state: State,
    ruleset: RuleSet,
    proposed_color: Color,
    decider: LocalDecider,
    peer: Transport,
    rng: random.Random,
    *,
    transport_timeout: float = 15.0,
    keepalive_interval: float = 5.0,
    liveness_deadline: float = 20.0,
    max_phases: int = 500,
    print_fn: PrintFn = print,
) -> OnlineMatchResult:
    """Play a full game against `peer`, deciding programs via `decider`.

    Parameters
    ----------
    initial_state, ruleset : State, RuleSet
        The agreed starting position and rules; both are cross-checked in the
        handshake before phase 0.
    proposed_color : Color
        The colour this peer requests; the handshake may reassign it on a
        collision (returns the negotiated colour, used for the whole match).
    decider : LocalDecider
        This peer's decision source. Wrap a bare `Agent` (which only ever
        plays, never resigns) with :func:`agent_decider`.
    peer : Transport
        The message channel.
    rng : random.Random
        Seeds the handshake nonce and, for an agent decider, its sampling.
    transport_timeout : float
        Liveness bound on a message the peer should already have sent (reveal,
        ack, handshake). *Not* a thinking limit (B1).
    keepalive_interval, liveness_deadline : float
        Ping cadence and the total-silence bound while waiting for the peer's
        commit (B5).
    max_phases : int
        Phase-limit draw bound (session metadata; must match the peer's).
    """
    local_color = await perform_handshake(
        peer,
        ruleset=ruleset,
        initial_state=initial_state,
        proposed_color=proposed_color,
        max_phases=max_phases,
        rng=rng,
        transport_timeout=transport_timeout,
    )
    print_fn(f"handshake ok — playing {local_color.value}")

    state = initial_state
    # A draw offer stands from the commit it rides on until the *peer's next
    # slot*, so both flags are carried across iterations, not recomputed from
    # the current phase: at the phase an offer is accepted, the offerer's own
    # decision that phase is a plain program, yet its offer must still count.
    peer_offer_standing = False
    local_offer_standing = False
    outcome: MatchOutcome = "draw"
    reason: TerminationReason = "phase_limit"
    phases: list[GamePhase] = []

    for _ in range(max_phases):
        phase = state.bookkeeping.phase_index
        (decision, local_commit), remote_slot = await asyncio.gather(
            _think_and_send_slot(
                peer,
                decider,
                state,
                local_color,
                ruleset,
                rng,
                phase=phase,
                peer_offered_draw=peer_offer_standing,
            ),
            _keepalive_until_slot(
                peer,
                phase=phase,
                keepalive_interval=keepalive_interval,
                liveness_deadline=liveness_deadline,
            ),
        )
        terminal = _resolve_slots(
            decision,
            remote_slot,
            local_color,
            local_offer_standing=local_offer_standing,
            peer_offer_standing=peer_offer_standing,
        )
        if terminal is not None:
            outcome, reason = terminal.outcome, terminal.reason
            print_fn(f"phase {phase}: {reason}")
            break

        assert local_commit is not None  # both committed a program
        assert isinstance(decision, PlayDecision)
        _check_envelope(remote_slot, MSG_COMMIT, phase)
        # Carry each side's offer forward one phase (until the peer's next slot).
        peer_offer_standing = bool(remote_slot.get("offer_draw"))
        local_offer_standing = decision.offer_draw
        if peer_offer_standing:
            print_fn(f"phase {phase}: peer offers a draw")

        new_state, phase_terminal, game_phase = await _exchange_reveal_and_resolve(
            peer,
            state,
            local_color,
            ruleset,
            local_commit,
            remote_slot,
            phase=phase,
            transport_timeout=transport_timeout,
        )
        state = new_state
        if game_phase is not None:
            phases.append(game_phase)
        if phase_terminal is not None:
            outcome, reason = phase_terminal.outcome, phase_terminal.reason
            print_fn(f"phase {phase} resolved: {outcome} ({reason})")
            break
        print_fn(f"phase {phase} resolved: ongoing")
    else:
        outcome, reason = "draw", "phase_limit"

    return OnlineMatchResult(
        final_state=state,
        outcome=outcome,
        termination_reason=reason,
        phases=tuple(phases),
    )
