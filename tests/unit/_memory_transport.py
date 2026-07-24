"""In-memory duplex `Transport` for fast, socket-free session tests (Phase 15a).

A `MemoryPeer` pair satisfies `net.transport.Transport` exactly, so the
session loop's concurrency (keepalive, non-blocking entry, match services) can
be exercised deterministically without loopback TCP. A peer that simply never
puts a message models a `kill -STOP` partition for the keepalive/liveness test.
"""

from __future__ import annotations

import asyncio
from typing import Any

from simult_chess.net.protocol import ProtocolError, TransportTimeout

_CLOSED: Any = object()


class MemoryPeer:
    """One end of an in-memory duplex, matching the `Transport` protocol."""

    def __init__(self, inbox: asyncio.Queue[Any], outbox: asyncio.Queue[Any]) -> None:
        self._inbox = inbox
        self._outbox = outbox
        self._closed = False

    async def send(self, message: dict[str, Any]) -> None:
        if self._closed:
            raise ProtocolError("send on a closed peer")
        await self._outbox.put(dict(message))

    async def recv(self, *, timeout: float | None = None) -> dict[str, Any]:
        try:
            item = await asyncio.wait_for(self._inbox.get(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TransportTimeout(f"peer did not respond within {timeout}s") from exc
        if item is _CLOSED:
            raise ProtocolError("peer closed the connection")
        result: dict[str, Any] = item
        return result

    async def close(self) -> None:
        self._closed = True
        await self._outbox.put(_CLOSED)


def memory_peer_pair() -> tuple[MemoryPeer, MemoryPeer]:
    """Two connected `MemoryPeer`s: what one sends, the other receives."""
    a_to_b: asyncio.Queue[Any] = asyncio.Queue()
    b_to_a: asyncio.Queue[Any] = asyncio.Queue()
    return MemoryPeer(b_to_a, a_to_b), MemoryPeer(a_to_b, b_to_a)
