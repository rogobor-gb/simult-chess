"""Stage A — annihilation matching (spec §6.3; INVARIANTS.md R3-R6).

Builds the bipartite conflict graph over :math:`M^\\ast` and greedily
matches candidate edges in increasing rank; a mover none of whose
conflict-partners survive to meet it completes unharmed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from simult_chess.core.collision import annihilation_rank, conflicts
from simult_chess.core.moves import DeclaredMove
from simult_chess.core.types import Color
from simult_chess.rules.ruleset import RuleSet

Edge = tuple[DeclaredMove, DeclaredMove]


@dataclass(frozen=True, slots=True)
class AnnihilationEvent:
    """One fired annihilation: a White/Black mover pair that collided mid-path."""

    white_move: DeclaredMove
    black_move: DeclaredMove
    rank: tuple[int, int]


@dataclass(frozen=True, slots=True)
class AnnihilationResult:
    """The output of Stage A: which executing moves annihilate."""

    events: tuple[AnnihilationEvent, ...]

    @property
    def annihilated(self) -> frozenset[DeclaredMove]:
        """Every move that died in a fired annihilation event."""
        killed: set[DeclaredMove] = set()
        for event in self.events:
            killed.add(event.white_move)
            killed.add(event.black_move)
        return frozenset(killed)

    def survives(self, move: DeclaredMove) -> bool:
        """Whether `move` is not annihilated (a surviving mover, spec §6.3)."""
        return move not in self.annihilated


def resolve_annihilation(
    executing: tuple[DeclaredMove, ...],
    ruleset: RuleSet,
    *,
    tie_break: Sequence[Edge] | None = None,
) -> AnnihilationResult:
    """Resolve Stage A: greedy bipartite matching by increasing rank (spec §6.3).

    Parameters
    ----------
    executing : tuple[DeclaredMove, ...]
        :math:`M^\\ast`, the moves that survived Stage F.
    ruleset : RuleSet
        Unused by the v1 "B" reading; present for the stage-strategy
        signature and the declined timed-model variant (spec §13.2).
    tie_break : Sequence[Edge] | None
        An explicit edge processing order. Only rank-preserving permutations
        are meaningful: equal-rank edges are vertex-disjoint (Lemma 6.3a) and
        provably commute, so the result is invariant to their relative order
        (inv M2b).
    """
    white_moves = [m for m in executing if m.color is Color.WHITE]
    black_moves = [m for m in executing if m.color is Color.BLACK]

    candidate_edges: list[Edge] = [
        (white, black)
        for white in white_moves
        for black in black_moves
        if conflicts(white.trajectory, black.trajectory)
    ]

    if tie_break is not None:
        ordered_edges = list(tie_break)
    else:
        ordered_edges = sorted(
            candidate_edges,
            key=lambda edge: (
                annihilation_rank(edge[0].index, edge[1].index),
                edge[0].index,
            ),
        )

    dead: set[DeclaredMove] = set()
    events: list[AnnihilationEvent] = []
    for white_move, black_move in ordered_edges:
        if white_move in dead or black_move in dead:
            continue
        dead.add(white_move)
        dead.add(black_move)
        events.append(
            AnnihilationEvent(
                white_move=white_move,
                black_move=black_move,
                rank=annihilation_rank(white_move.index, black_move.index),
            )
        )

    return AnnihilationResult(events=tuple(events))
