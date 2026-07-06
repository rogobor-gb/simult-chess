"""Declared-move extraction: expand two programs into indexed DeclaredMoves.

Spec §3 defines the multiset :math:`M=M_W\\sqcup M_B` of declared moves. A
Castle action occupies one program slot but contributes two synchronized
sub-trajectories (spec §6.6); each is expanded here into its own
`DeclaredMove`, consuming its own sequential index within the owner's move
list.

This indexing scheme is a documented convention **[C]**: the spec fixes
annihilation rank on "declaration order... chosen by its owner" (§6.3) but
does not specify how a Castle's two sub-trajectories share that ordering
with the rest of the program. Giving each sub-trajectory its own sequential
index is the simplest, deterministic, well-defined choice, and is
consistent across a phase regardless of internal processing order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from simult_chess.core import geometry
from simult_chess.core.types import (
    Castle,
    Color,
    Move,
    Program,
    State,
    Token,
    Trajectory,
)

MoveKind = Literal["move", "castle_king", "castle_rook"]


@dataclass(frozen=True, slots=True)
class DeclaredMove:
    """One entry of the multiset :math:`M=M_W\\sqcup M_B` (spec §3).

    Parameters
    ----------
    token : Token
        The token displaced by this move.
    trajectory : Trajectory
        Its declared trajectory.
    color : Color
        The owning player.
    index : int
        1-based declaration order within the owner's expanded move list —
        the "internal order... chosen by its owner" of spec §6.3.
    kind : MoveKind
        Whether this came from a plain Move, or a Castle's king/rook
        sub-trajectory (spec §6.6).
    """

    token: Token
    trajectory: Trajectory
    color: Color
    index: int
    kind: MoveKind


def _expand_program(
    state: State, program: Program, color: Color
) -> tuple[DeclaredMove, ...]:
    moves: list[DeclaredMove] = []
    index = 1
    for action in program:
        if isinstance(action, Move):
            moves.append(
                DeclaredMove(
                    token=action.token,
                    trajectory=action.trajectory,
                    color=color,
                    index=index,
                    kind="move",
                )
            )
            index += 1
        elif isinstance(action, Castle):
            castle = geometry.castle_move(state, color, action.side)
            if castle is None:
                raise ValueError(
                    "Castle action must be legal on declaration (Φ requires L(s,π))"
                )
            moves.append(
                DeclaredMove(
                    token=castle.king_token,
                    trajectory=castle.king_trajectory,
                    color=color,
                    index=index,
                    kind="castle_king",
                )
            )
            index += 1
            moves.append(
                DeclaredMove(
                    token=castle.rook_token,
                    trajectory=castle.rook_trajectory,
                    color=color,
                    index=index,
                    kind="castle_rook",
                )
            )
            index += 1
    return tuple(moves)


def extract_declared_moves(
    state: State, program_white: Program, program_black: Program
) -> tuple[DeclaredMove, ...]:
    """Extract :math:`M=M_W\\sqcup M_B` from both programs (spec §3)."""
    white_moves = _expand_program(state, program_white, Color.WHITE)
    black_moves = _expand_program(state, program_black, Color.BLACK)
    return white_moves + black_moves
