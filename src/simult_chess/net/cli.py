"""Entrypoint: ``python -m simult_chess.net.cli host|connect`` (Phase 8).

One peer hosts on a port, the other connects to it -- no lobby or
matchmaking in v1. Each side picks its own color and decision source
(a human prompted at the terminal, or an agent).
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys

from simult_chess.agents.base import Agent
from simult_chess.agents.greedy import greedy_program
from simult_chess.agents.random_legal import random_legal_program
from simult_chess.core.types import Color, Program, State
from simult_chess.net.session import run_online_match
from simult_chess.net.transport import Peer, connect_peer, host_peer
from simult_chess.referee.setup import standard_starting_state
from simult_chess.rules.ruleset import RuleSet
from simult_chess.ui.session import prompt_program

_AGENTS: dict[str, Agent] = {"random": random_legal_program, "greedy": greedy_program}


def _human_program_source(
    state: State, color: Color, ruleset: RuleSet, rng: random.Random
) -> Program:
    del rng  # human decides directly; no randomness needed
    return prompt_program(state, color, ruleset, input_fn=input, print_fn=print)


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

    return parser


async def _amain(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    state = standard_starting_state()
    ruleset = RuleSet()
    color = Color.WHITE if args.color == "white" else Color.BLACK
    program_source: Agent = (
        _human_program_source if args.agent == "human" else _AGENTS[args.agent]
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
            state, ruleset, color, program_source, peer, rng
        )
    finally:
        await peer.close()

    print(f"game over: {result.outcome}")
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
