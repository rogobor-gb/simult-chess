from __future__ import annotations

import asyncio
import random

import pytest

from simult_chess.agents.random_legal import random_legal_program
from simult_chess.core.types import Color
from simult_chess.net.commitment import commitment_hash
from simult_chess.net.protocol import ProtocolError
from simult_chess.net.session import run_online_match
from simult_chess.net.transport import connect_peer, host_peer
from simult_chess.referee.setup import standard_starting_state
from simult_chess.rules.ruleset import RuleSet

RULESET = RuleSet()
_TEST_PORT = 18766


def test_run_online_match_reaches_the_same_outcome_on_both_peers() -> None:
    async def scenario() -> None:
        state = standard_starting_state()

        async def host_side() -> object:
            peer, _bound = await host_peer(_TEST_PORT, host="127.0.0.1")
            try:
                return await run_online_match(
                    state,
                    RULESET,
                    Color.WHITE,
                    random_legal_program,
                    peer,
                    random.Random(1),
                    max_phases=6,
                    print_fn=lambda _line: None,
                )
            finally:
                await peer.close()

        async def client_side() -> object:
            for _ in range(200):
                try:
                    peer = await connect_peer("127.0.0.1", _TEST_PORT)
                    break
                except OSError:
                    await asyncio.sleep(0.01)
            else:
                raise AssertionError("could not connect")
            try:
                return await run_online_match(
                    state,
                    RULESET,
                    Color.BLACK,
                    random_legal_program,
                    peer,
                    random.Random(2),
                    max_phases=6,
                    print_fn=lambda _line: None,
                )
            finally:
                await peer.close()

        host_result, client_result = await asyncio.gather(host_side(), client_side())
        assert host_result.outcome == client_result.outcome
        assert host_result.final_state.board == client_result.final_state.board
        assert host_result.final_state.bookkeeping.phase_index == 6

    asyncio.run(scenario())


def test_run_online_match_detects_a_tampered_reveal() -> None:
    """A peer that reveals a program not matching its own earlier commitment
    (a dropped/garbled commitment, spec §14's DoD) must be caught, not
    silently accepted."""

    async def scenario() -> None:
        state = standard_starting_state()

        async def honest_side() -> None:
            peer, _bound = await host_peer(_TEST_PORT + 1, host="127.0.0.1")
            try:
                with pytest.raises(
                    ProtocolError, match="does not match its commitment"
                ):
                    await run_online_match(
                        state,
                        RULESET,
                        Color.WHITE,
                        random_legal_program,
                        peer,
                        random.Random(1),
                        max_phases=1,
                        print_fn=lambda _line: None,
                    )
            finally:
                await peer.close()

        async def cheating_side() -> None:
            for _ in range(200):
                try:
                    peer = await connect_peer("127.0.0.1", _TEST_PORT + 1)
                    break
                except OSError:
                    await asyncio.sleep(0.01)
            else:
                raise AssertionError("could not connect")

            phase = state.bookkeeping.phase_index
            await peer.recv(timeout=5)  # the honest side's commit
            # commit to one thing, then reveal another
            await peer.send(
                {
                    "type": "commit",
                    "phase_index": phase,
                    "hash": commitment_hash(b"x", []),
                }
            )
            await peer.recv(timeout=5)  # the honest side's reveal
            await peer.send(
                {
                    "type": "reveal",
                    "phase_index": phase,
                    "salt": "ff",
                    "program": [{"kind": "castle", "side": "king"}],
                }
            )
            await peer.close()

        await asyncio.gather(honest_side(), cheating_side())

    asyncio.run(scenario())
