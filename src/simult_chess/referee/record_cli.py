"""Entrypoint: ``python -m simult_chess.referee.record_cli`` (Phase 15d).

``replay FILE`` re-derives every state in a ``.scn`` record through Φ and
verifies the recorded resolutions, refusing on any mismatch. ``to-fixture FILE
--phase N`` harvests one phase as a self-contained JSON regression fixture.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from simult_chess.referee.record import (
    RecordError,
    read_record,
    write_phase_fixture,
)
from simult_chess.rules.variants import get_variant, variant_names


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m simult_chess.referee.record_cli", description=__doc__
    )
    sub = parser.add_subparsers(dest="command", required=True)

    replay = sub.add_parser("replay", help="re-derive and verify a .scn record")
    replay.add_argument("file", type=Path)
    replay.add_argument(
        "--expect-variant",
        choices=variant_names(),
        default=None,
        help="refuse to replay unless the record's fingerprint matches this "
        "named variant's rules",
    )

    fixture = sub.add_parser(
        "to-fixture", help="emit a phase of a .scn record as a JSON fixture"
    )
    fixture.add_argument("file", type=Path)
    fixture.add_argument("--phase", type=int, required=True)
    fixture.add_argument(
        "--out", type=Path, default=None, help="write here instead of stdout"
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    text = args.file.read_text(encoding="utf-8")

    try:
        if args.command == "replay":
            expected = (
                get_variant(args.expect_variant).fingerprint()
                if args.expect_variant is not None
                else None
            )
            record = read_record(text, expected_fingerprint=expected)
            print(
                f"replay ok: {len(record.phases)} phases, "
                f"{record.outcome} ({record.termination_reason}); "
                f"rules {record.ruleset_fingerprint[:12]}…"
            )
            return 0

        record = read_record(text)
        fixture_json = write_phase_fixture(record, args.phase)
        if args.out is not None:
            args.out.write_text(fixture_json, encoding="utf-8")
            print(f"wrote fixture for phase {args.phase} to {args.out}")
        else:
            sys.stdout.write(fixture_json)
        return 0
    except RecordError as exc:
        print(f"record error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
