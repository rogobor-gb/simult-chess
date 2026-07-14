"""The stage matrix :math:`U` over two restricted supports (spec §8.1, A7).

A pure wrapper over :math:`\\Phi` — the transition operator itself is never
touched here, only called.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from simult_chess.core.phi import phi
from simult_chess.core.types import Color, PieceType, Program, State
from simult_chess.rules.ruleset import RuleSet

FloatMatrix = npt.NDArray[np.float64]

_PIECE_VALUES: dict[PieceType, int] = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}

_TERMINAL_PAYOFF: dict[str, float] = {
    "white_wins": 1.0,
    "black_wins": -1.0,
    "draw": 0.0,
}


def material_difference(state: State) -> float:
    """White material minus Black material — a *solver parameter* (spec
    §8.1's v1 payoff functional), never a rule. Anti-symmetric under χ by
    construction: mirroring inverts every token's color, so the white- and
    black-material totals swap and the difference negates (needed for
    inv M5's antisymmetry, spec §0/Lemma 6.3b/6.4b's no-first-mover-advantage
    property to extend to this surrogate)."""
    total = 0
    for token in state.board:
        value = _PIECE_VALUES[token.typ]
        total += value if token.color is Color.WHITE else -value
    return float(total)


def payoff(state: State, outcome: str) -> float:
    """:math:`u(s')`: terminal :math:`\\{-1,0,+1\\}` if `state` is terminal,
    else the material-difference surrogate (spec §8.1)."""
    if outcome in _TERMINAL_PAYOFF:
        return _TERMINAL_PAYOFF[outcome]
    return material_difference(state)


def build_stage_matrix(
    state: State,
    support_white: tuple[Program, ...],
    support_black: tuple[Program, ...],
    ruleset: RuleSet,
) -> FloatMatrix:
    """:math:`U_{ij} = u(\\Phi(s, \\pi_i, \\pi_j))` over the two supports
    (spec §8.1). Rows index `support_white`, columns `support_black`."""
    matrix: FloatMatrix = np.empty(
        (len(support_white), len(support_black)), dtype=np.float64
    )
    for i, program_white in enumerate(support_white):
        for j, program_black in enumerate(support_black):
            result = phi(state, program_white, program_black, ruleset)
            matrix[i, j] = payoff(result.state, result.outcome)
    return matrix
