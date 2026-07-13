"""Cross-validate `core/geometry.py` pseudo-legal trajectories against
python-chess, per the Phase 2 DoD (docs/DEVELOPMENT_addendum_v1.1.md §9b).

Requires the optional `oracle` extra (`pip install simult-chess[oracle]`);
skipped cleanly if `chess` isn't installed, since core/rules never depend
on it (pyproject.toml's own quarantine note).

**Documented exclusions** (semantics that don't coincide between the two
engines, so are dropped from comparison rather than asserted equal):
- **En passant** — dropped in v1 (spec §6.5); python-chess moves flagged
  `is_en_passant` are excluded from its move set.
- **Castling** — legality is owned by our L-clauses/`geometry.castle_move`,
  not `pseudo_legal_trajectories`; python-chess moves flagged `is_castling`
  are excluded from its move set.
- **Check** — neither engine's *pseudo-legal* generation filters on it (spec
  §10/T2: no check concept here; python-chess's `pseudo_legal_moves` likewise
  doesn't require resolving check), so no exclusion is actually needed here,
  but it's worth stating: we never compare against python-chess's *legal*
  (check-filtered) move set.
- **Cooldown and simultaneity** — unknown to python-chess; the boards
  compared here carry no cooldown state, so this is vacuous rather than an
  active exclusion.
"""

from __future__ import annotations

import pytest

chess = pytest.importorskip("chess")

import random  # noqa: E402

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from simult_chess.core import geometry  # noqa: E402
from simult_chess.core.types import (  # noqa: E402
    Bookkeeping,
    CastlingRights,
    Color,
    PieceType,
    Square,
    State,
    Token,
)

_PIECE_SYMBOL_TO_TYPE: dict[str, PieceType] = {
    "p": "p",
    "n": "n",
    "b": "b",
    "r": "r",
    "q": "q",
    "k": "k",
}


def _random_reachable_board(rng: random.Random, max_plies: int) -> chess.Board:
    """Play up to `max_plies` random *legal* python-chess moves from the
    start position, stopping early on game-over. The result is a reachable
    (hence "legal", spec §2's sense of a position a real game can reach)
    board — not a claim that our engine's own legality would produce it,
    since our variant's L-clauses differ (no check, cooldown, etc.)."""
    board = chess.Board()
    for _ in range(max_plies):
        if board.is_game_over():
            break
        moves = list(board.legal_moves)
        board.push(rng.choice(moves))
    return board


def _to_state(board: chess.Board) -> tuple[State, dict[int, Token]]:
    """Convert a python-chess board to our `State` (no cooldown/reservations;
    those concepts don't exist in python-chess). Returns the state plus a
    square-indexed lookup of the token placed on each occupied square."""
    board_map: dict[Token, Square] = {}
    tokens_by_square: dict[int, Token] = {}
    for next_id, square in enumerate(chess.SQUARES):
        piece = board.piece_at(square)
        if piece is None:
            continue
        color = Color.WHITE if piece.color == chess.WHITE else Color.BLACK
        piece_type = _PIECE_SYMBOL_TO_TYPE[piece.symbol().lower()]
        token = Token(id=next_id, color=color, typ=piece_type)
        our_square = Square(chess.square_file(square), chess.square_rank(square))
        board_map[token] = our_square
        tokens_by_square[square] = token
    state = State(
        board=board_map,
        cooldown=frozenset(),
        reservations_white=(),
        reservations_black=(),
        bookkeeping=Bookkeeping(
            castling_rights=CastlingRights(),
            repetition_ledger={},
            no_progress_counter=0,
            phase_index=0,
        ),
    )
    return state, tokens_by_square


def _their_destinations(board: chess.Board, square: int, color: bool) -> set[Square]:
    """python-chess pseudo-legal destinations for the piece at `square`,
    dropping en passant and castling (documented exclusions above)."""
    scratch = board.copy(stack=False)
    scratch.turn = color
    destinations: set[Square] = set()
    for move in scratch.generate_pseudo_legal_moves(from_mask=chess.BB_SQUARES[square]):
        if scratch.is_en_passant(move) or scratch.is_castling(move):
            continue
        file = chess.square_file(move.to_square)
        rank = chess.square_rank(move.to_square)
        destinations.add(Square(file, rank))
    return destinations


def _our_destinations(state: State, token: Token) -> set[Square]:
    return {t.destination for t in geometry.pseudo_legal_trajectories(state, token)}


@given(max_plies=st.integers(min_value=0, max_value=60), seed=st.integers(min_value=0))
@settings(max_examples=1000, deadline=None)
@pytest.mark.slow
def test_pseudo_legal_destinations_match_python_chess(
    max_plies: int, seed: int
) -> None:
    board = _random_reachable_board(random.Random(seed), max_plies)
    state, tokens_by_square = _to_state(board)
    for square, token in tokens_by_square.items():
        ours = _our_destinations(state, token)
        theirs = _their_destinations(board, square, board.piece_at(square).color)
        assert ours == theirs, (
            f"mismatch for {token.typ}{token.color.value} on "
            f"{chess.square_name(square)}: ours={ours} theirs={theirs} "
            f"fen={board.fen()!r}"
        )
