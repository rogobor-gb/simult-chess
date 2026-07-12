"""Entrypoint: ``python -m simult_chess.ui.cli`` (dev brief §4 Phase 7).

Two modes: local hot-seat (two humans, one terminal) or human-vs-agent.
"""

from __future__ import annotations

import argparse
import random
import sys

from simult_chess.agents.greedy import greedy_program
from simult_chess.agents.random_legal import random_legal_program
from simult_chess.core.types import Color
from simult_chess.referee.setup import standard_starting_state
from simult_chess.rules.ruleset import RuleSet

_AGENTS = {"random": random_legal_program, "greedy": greedy_program}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="simult-chess", description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    subparsers.add_parser("hotseat", help="two humans, one terminal")

    agent_parser = subparsers.add_parser("agent", help="human vs. an agent")
    agent_parser.add_argument("--human", choices=("white", "black"), default="white")
    agent_parser.add_argument("--agent", choices=tuple(_AGENTS), default="random")
    agent_parser.add_argument("--seed", type=int, default=0)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    state = standard_starting_state()
    ruleset = RuleSet()

    try:
        if args.mode == "hotseat":
            from simult_chess.ui.session import run_hot_seat

            run_hot_seat(state, ruleset)
            return 0

        from simult_chess.ui.session import run_human_vs_agent

        human_color = Color.WHITE if args.human == "white" else Color.BLACK
        agent = _AGENTS[args.agent]
        rng = random.Random(args.seed)
        run_human_vs_agent(state, ruleset, human_color, agent, rng)
        return 0
    except (EOFError, KeyboardInterrupt):
        print()
        print("game aborted")
        return 1


if __name__ == "__main__":
    sys.exit(main())
