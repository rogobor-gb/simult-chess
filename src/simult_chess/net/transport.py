"""Minimal asyncio TCP transport: one peer hosts, the other connects (Phase 8).

No lobby/matchmaking in v1 (spec §14's scope note): a `Peer` is exactly
one accepted/opened connection, framed as newline-delimited JSON.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from simult_chess.net.protocol import ProtocolError


@dataclass(slots=True)
class Peer:
    """One TCP connection to the other decider, JSON-message framed."""

    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter

    async def send(self, message: dict[str, Any]) -> None:
        """Write one JSON message, newline-delimited."""
        data = json.dumps(message, separators=(",", ":")).encode() + b"\n"
        self.writer.write(data)
        await self.writer.drain()

    async def recv(self, *, timeout: float | None = None) -> dict[str, Any]:
        """Read one JSON message, raising on timeout or a closed connection."""
        try:
            line = await asyncio.wait_for(self.reader.readline(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise ProtocolError(f"peer did not respond within {timeout}s") from exc
        if not line:
            raise ProtocolError("peer closed the connection")
        result: dict[str, Any] = json.loads(line.decode())
        return result

    async def close(self) -> None:
        """Close the underlying connection."""
        self.writer.close()
        await self.writer.wait_closed()


async def host_peer(
    port: int,
    host: str = "0.0.0.0",
    *,
    on_listening: Callable[[int], None] | None = None,
) -> tuple[Peer, int]:
    """Listen on `host`:`port` and return the first accepted connection.

    Returns the `Peer` together with the actually-bound port (useful when
    `port=0` lets the OS choose one). `on_listening`, if given, is called
    with that port as soon as the socket is bound -- *before* waiting for a
    connection, so a test can learn an OS-chosen port without a fixed one.
    """
    loop = asyncio.get_running_loop()
    connected: asyncio.Future[Peer] = loop.create_future()

    async def _on_connect(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        if not connected.done():
            connected.set_result(Peer(reader=reader, writer=writer))

    server = await asyncio.start_server(_on_connect, host, port)
    bound_port = server.sockets[0].getsockname()[1]
    if on_listening is not None:
        on_listening(bound_port)
    try:
        peer = await connected
    finally:
        server.close()
    return peer, bound_port


async def connect_peer(remote_host: str, port: int) -> Peer:
    """Open a connection to a hosting peer at `remote_host`:`port`."""
    reader, writer = await asyncio.open_connection(remote_host, port)
    return Peer(reader=reader, writer=writer)
