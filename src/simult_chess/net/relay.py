"""Rendezvous relay (Phase 15c) — a byte pipe with a room code.

**Deliberately minimal and disposable.** This module contains **zero game
logic**: it never parses a program, never runs Φ, never knows what a phase is,
and never becomes a game server. It imports nothing from
``simult_chess.core`` / ``rules`` / ``referee`` (asserted by a test, the same
quarantine pattern as the optional extras). When a real backend is built this
file is deleted in one commit and nothing else changes.

It exists so two people who cannot port-forward can still play: each connects
out to the relay, sends a room code as its first line, and the relay pipes the
two connections that share a code byte-for-byte until either closes. No TLS, no
accounts, no persistence, no reconnection — the room code is the only secret and
traffic is plaintext. Acceptable for playing your brother; do not expose it
publicly (README).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable

_BUF = 65536


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Copy bytes one way until EOF, then close the writer."""
    try:
        while data := await reader.read(_BUF):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        writer.close()


async def serve(
    host: str = "0.0.0.0",
    port: int = 0,
    *,
    on_listening: Callable[[int], None] | None = None,
) -> None:
    """Run the relay until cancelled. Pairs connections by their first-line code."""
    waiting: dict[str, tuple[asyncio.StreamReader, asyncio.StreamWriter]] = {}

    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        room = (await reader.readline()).decode(errors="replace").strip()
        if not room:
            writer.close()
            return
        other = waiting.pop(room, None)
        if other is None:
            waiting[room] = (reader, writer)
            return  # first peer waits; the partner's handler starts the pipes
        other_reader, other_writer = other
        await asyncio.gather(
            _pipe(reader, other_writer),
            _pipe(other_reader, writer),
        )

    server = await asyncio.start_server(handle, host, port)
    bound_port = server.sockets[0].getsockname()[1]
    if on_listening is not None:
        on_listening(bound_port)
    async with server:
        await server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m simult_chess.net.relay", description=__doc__
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        asyncio.run(
            serve(
                args.host,
                args.port,
                on_listening=lambda p: print(f"relay listening on port {p}"),
            )
        )
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
