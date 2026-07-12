from __future__ import annotations

from simult_chess.core.types import Color, Move, Square, Token, Trajectory
from simult_chess.referee.observe import ObservationChannel


def test_commit_then_reveal_round_trips_the_program() -> None:
    token = Token(id=1, color=Color.WHITE, typ="p")
    move = Move(token=token, trajectory=Trajectory(path=(Square(4, 1), Square(4, 2))))
    program = (move,)

    channel = ObservationChannel()
    commitment = channel.commit(Color.WHITE, program)

    assert commitment.color is Color.WHITE
    assert channel.reveal(commitment) == program
