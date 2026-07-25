"""Phase 15c: the rendezvous relay — quarantine and byte-transparency.

The relay is disposable and must contain zero game logic: its import graph
imports nothing from ``simult_chess.core`` / ``rules`` / ``referee`` (the same
quarantine pattern as the optional extras), and it is under 100 lines.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from simult_chess.net import relay
from simult_chess.net.protocol import TransportTimeout
from simult_chess.net.transport import connect_via_relay

_RELAY_SRC = Path(relay.__file__)


def test_relay_is_under_100_lines() -> None:
    lines = _RELAY_SRC.read_text(encoding="utf-8").splitlines()
    assert len(lines) < 100, f"relay grew to {len(lines)} lines; keep it disposable"


def test_relay_import_graph_is_engine_free() -> None:
    """The relay pulls in no core/rules/referee module (a byte pipe, not a server)."""
    code = (
        "import simult_chess.net.relay, sys; "
        "leaked = sorted(m for m in sys.modules if m.startswith("
        "('simult_chess.core', 'simult_chess.rules', 'simult_chess.referee'))); "
        "assert not leaked, 'engine leaked into the relay: ' + repr(leaked)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


def test_two_peers_exchange_bytes_through_the_relay() -> None:
    """A full round trip of arbitrary messages, transparent to the relay."""

    async def scenario() -> None:
        bound: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        server = asyncio.ensure_future(
            relay.serve("127.0.0.1", 0, on_listening=bound.set_result)
        )
        try:
            port = await bound
            alice = await connect_via_relay("127.0.0.1", port, "room-42")
            bob = await connect_via_relay("127.0.0.1", port, "room-42")

            await alice.send({"type": "hello", "n": 1})
            assert await bob.recv(timeout=2.0) == {"type": "hello", "n": 1}
            await bob.send({"type": "reply", "n": 2})
            assert await alice.recv(timeout=2.0) == {"type": "reply", "n": 2}

            await alice.close()
            await bob.close()
        finally:
            server.cancel()
            with pytest.raises(asyncio.CancelledError):
                await server

    asyncio.run(scenario())


def test_different_rooms_do_not_cross() -> None:
    """Two peers in different rooms are never paired (no partner => no traffic)."""

    async def scenario() -> None:
        bound: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        server = asyncio.ensure_future(
            relay.serve("127.0.0.1", 0, on_listening=bound.set_result)
        )
        try:
            port = await bound
            alice = await connect_via_relay("127.0.0.1", port, "room-A")
            bob = await connect_via_relay("127.0.0.1", port, "room-B")
            await alice.send({"type": "hello"})
            with pytest.raises(TransportTimeout):
                await bob.recv(timeout=0.3)  # no partner in room-B
            await alice.close()
            await bob.close()
        finally:
            server.cancel()
            with pytest.raises(asyncio.CancelledError):
                await server

    asyncio.run(scenario())
