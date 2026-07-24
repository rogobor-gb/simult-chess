"""Entrypoint: ``python -m simult_chess.ui.cli`` (dev brief §4 Phase 7).

Three modes: local hot-seat (two humans, one terminal), human-vs-agent, or
`variants` -- a listing of the rule sets playable by name (Phase 14).
"""

from __future__ import annotations

import argparse
import random
import sys

from simult_chess.agents.greedy import greedy_program
from simult_chess.agents.random_legal import random_legal_program
from simult_chess.core.types import Color
from simult_chess.referee.setup import standard_starting_state
from simult_chess.rules.variants import (
    BASELINE_NAME,
    describe_variants,
    get_variant,
    variant_names,
)

_AGENTS = {"random": random_legal_program, "greedy": greedy_program}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="simult-chess", description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    hotseat_parser = subparsers.add_parser("hotseat", help="two humans, one terminal")

    agent_parser = subparsers.add_parser("agent", help="human vs. an agent")
    agent_parser.add_argument("--human", choices=("white", "black"), default="white")
    agent_parser.add_argument("--agent", choices=tuple(_AGENTS), default="random")
    agent_parser.add_argument("--seed", type=int, default=0)

    subparsers.add_parser("variants", help="list the rule sets playable by name")

    for sub in (hotseat_parser, agent_parser):
        sub.add_argument(
            "--variant",
            choices=variant_names(),
            default=BASELINE_NAME,
            help="rule set to play (default: the frozen provisional v1.1 "
            "defaults); see the `variants` subcommand",
        )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.mode == "variants":
        print(describe_variants())
        return 0

    state = standard_starting_state()
    ruleset = get_variant(args.variant)
    print(f"rules: {args.variant} [{ruleset.fingerprint()[:12]}]")

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
