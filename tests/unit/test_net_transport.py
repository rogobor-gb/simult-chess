from __future__ import annotations

import asyncio

from simult_chess.net.transport import connect_peer, host_peer

_TEST_PORT = 18765


def test_peer_send_recv_round_trip() -> None:
    async def host_side() -> None:
        peer, _bound_port = await host_peer(_TEST_PORT, host="127.0.0.1")
        message = await peer.recv(timeout=5)
        assert message == {"type": "ping", "n": 1}
        await peer.send({"type": "pong", "n": 1})
        await peer.close()

    async def client_side() -> None:
        for _ in range(200):
            try:
                peer = await connect_peer("127.0.0.1", _TEST_PORT)
                break
            except OSError:
                await asyncio.sleep(0.01)
        else:
            raise AssertionError("could not connect to the hosting peer")
        await peer.send({"type": "ping", "n": 1})
        reply = await peer.recv(timeout=5)
        assert reply == {"type": "pong", "n": 1}
        await peer.close()

    async def scenario() -> None:
        await asyncio.gather(host_side(), client_side())

    asyncio.run(scenario())


def test_host_peer_with_ephemeral_port_is_reachable_via_on_listening() -> None:
    async def scenario() -> None:
        port_ready: asyncio.Future[int] = asyncio.get_running_loop().create_future()

        async def host_side() -> None:
            peer, bound_port = await host_peer(
                0, host="127.0.0.1", on_listening=port_ready.set_result
            )
            assert bound_port == port_ready.result()
            await peer.recv(timeout=5)
            await peer.close()

        async def client_side() -> None:
            port = await port_ready
            peer = await connect_peer("127.0.0.1", port)
            await peer.send({"type": "hello"})
            await peer.close()

        await asyncio.gather(host_side(), client_side())

    asyncio.run(scenario())
