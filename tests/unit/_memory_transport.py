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
    """One end of an in-memory duplex, matching the `Transport` protocol.

    ``send_delay`` injects a one-way latency on this peer's outgoing messages,
    so a pair can simulate an asymmetric link (Phase 15b clock tests).
    """

    def __init__(
        self,
        inbox: asyncio.Queue[Any],
        outbox: asyncio.Queue[Any],
        *,
        send_delay: float = 0.0,
    ) -> None:
        self._inbox = inbox
        self._outbox = outbox
        self._send_delay = send_delay
        self._closed = False

    async def send(self, message: dict[str, Any]) -> None:
        if self._closed:
            raise ProtocolError("send on a closed peer")
        if self._send_delay:
            await asyncio.sleep(self._send_delay)
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


def memory_peer_pair(
    *, delay_a: float = 0.0, delay_b: float = 0.0
) -> tuple[MemoryPeer, MemoryPeer]:
    """Two connected `MemoryPeer`s: what one sends, the other receives.

    ``delay_a``/``delay_b`` give each peer a one-way send latency (asymmetric
    when they differ).
    """
    a_to_b: asyncio.Queue[Any] = asyncio.Queue()
    b_to_a: asyncio.Queue[Any] = asyncio.Queue()
    return (
        MemoryPeer(b_to_a, a_to_b, send_delay=delay_a),
        MemoryPeer(a_to_b, b_to_a, send_delay=delay_b),
    )
