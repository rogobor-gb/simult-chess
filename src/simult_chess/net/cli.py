"""Entrypoint: ``python -m simult_chess.net.cli host|connect`` (Phase 8,
hardened in Phase 15a).

One peer hosts on a port, the other connects to it -- no lobby or matchmaking
in v1. Each side proposes its own colour (the handshake negotiates a
collision), rules variant, and decision source (a human prompted at the
terminal, or an agent). A human may type a program, ``resign``, ``abort``,
``accept`` (a standing draw offer), or prefix a program with ``draw`` to offer
one.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys

from simult_chess.agents.base import Agent
from simult_chess.agents.greedy import greedy_program
from simult_chess.agents.random_legal import random_legal_program
from simult_chess.core import legality
from simult_chess.core.types import Color, State
from simult_chess.net.handshake import HandshakeError
from simult_chess.net.protocol import ProtocolError
from simult_chess.net.session import (
    AbortDecision,
    AcceptDrawDecision,
    Decision,
    LocalDecider,
    PlayDecision,
    ResignDecision,
    agent_decider,
    run_online_match,
)
from simult_chess.net.transport import Peer, connect_peer, host_peer
from simult_chess.referee.setup import standard_starting_state
from simult_chess.rules.ruleset import RuleSet
from simult_chess.rules.variants import BASELINE_NAME, get_variant, variant_names
from simult_chess.ui import notation

_AGENTS: dict[str, Agent] = {"random": random_legal_program, "greedy": greedy_program}


def _human_decider(
    state: State,
    color: Color,
    ruleset: RuleSet,
    rng: random.Random,
    *,
    peer_offered_draw: bool,
) -> Decision:
    """Prompt a human for a program or a match-service action.

    ``resign`` / ``abort`` / ``accept`` are whole-line commands; a ``draw``
    prefix on an otherwise-normal program attaches a standing draw offer to the
    commit. Re-prompts on a parse or legality error, never accepting a partial
    program.
    """
    del rng  # a human decides directly
    accept_hint = "/'accept'" if peer_offered_draw else ""
    while True:
        raw = input(
            f"{color.value} — program, 'resign'/'abort'{accept_hint}, "
            f"or 'draw <program>' to offer: "
        ).strip()
        low = raw.lower()
        if low == "resign":
            return ResignDecision()
        if low == "abort":
            return AbortDecision()
        if low == "accept":
            if peer_offered_draw:
                return AcceptDrawDecision()
            print("no draw has been offered")
            continue
        offer_draw = False
        if low.startswith("draw ") or low == "draw":
            offer_draw = True
            raw = raw[len("draw ") :].strip()
        try:
            program = notation.parse_program(raw, state, color)
        except notation.NotationError as exc:
            print(f"parse error: {exc}")
            continue
        violations = legality.check_legal_program(state, program, color, ruleset)
        if violations:
            for violation in violations:
                print(f"illegal ({violation.invariant_id}): {violation.detail}")
            continue
        return PlayDecision(program, offer_draw=offer_draw)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="simult-chess-net", description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    host_parser = subparsers.add_parser("host", help="listen for a peer")
    host_parser.add_argument("--port", type=int, required=True)

    connect_parser = subparsers.add_parser("connect", help="connect to a hosting peer")
    connect_parser.add_argument("--remote-host", required=True)
    connect_parser.add_argument("--port", type=int, required=True)

    for sub in (host_parser, connect_parser):
        sub.add_argument("--color", choices=("white", "black"), required=True)
        sub.add_argument("--agent", choices=(*_AGENTS, "human"), default="human")
        sub.add_argument("--seed", type=int, default=0)
        sub.add_argument(
            "--variant",
            choices=variant_names(),
            default=BASELINE_NAME,
            help="rule set to play (default: the frozen provisional v1.1 "
            "defaults). The handshake cross-checks the fingerprint and aborts "
            "a mismatch by name, so both peers must pass the same one",
        )
        sub.add_argument("--max-phases", type=int, default=500)
        sub.add_argument(
            "--transport-timeout",
            type=float,
            default=15.0,
            help="liveness bound (s) on a message the peer should already have "
            "sent (reveal/ack/handshake) -- NOT a thinking limit",
        )
        sub.add_argument(
            "--keepalive-interval",
            type=float,
            default=5.0,
            help="ping cadence (s) while the peer is thinking",
        )
        sub.add_argument(
            "--liveness-deadline",
            type=float,
            default=20.0,
            help="total silence (s) tolerated before declaring the peer dead",
        )

    return parser


async def _amain(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    state = standard_starting_state()
    ruleset = get_variant(args.variant)
    print(f"rules: {args.variant} [{ruleset.fingerprint()[:12]}]")
    color = Color.WHITE if args.color == "white" else Color.BLACK
    decider: LocalDecider = (
        _human_decider if args.agent == "human" else agent_decider(_AGENTS[args.agent])
    )
    rng = random.Random(args.seed)

    peer: Peer
    if args.mode == "host":
        peer, bound_port = await host_peer(args.port)
        print(f"peer connected (listened on port {bound_port})")
    else:
        peer = await connect_peer(args.remote_host, args.port)
        print(f"connected to {args.remote_host}:{args.port}")

    try:
        result = await run_online_match(
            state,
            ruleset,
            color,
            decider,
            peer,
            rng,
            transport_timeout=args.transport_timeout,
            keepalive_interval=args.keepalive_interval,
            liveness_deadline=args.liveness_deadline,
            max_phases=args.max_phases,
        )
    except HandshakeError as exc:
        print(f"handshake failed: {exc}")
        return 2
    except ProtocolError as exc:
        print(f"protocol error: {exc}")
        return 3
    finally:
        await peer.close()

    print(f"game over: {result.outcome} ({result.termination_reason})")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_amain(argv))
    except (EOFError, KeyboardInterrupt):
        print()
        print("game aborted")
        return 1


if __name__ == "__main__":
    sys.exit(main())
