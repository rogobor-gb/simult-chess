from __future__ import annotations

from collections.abc import Mapping

from hypothesis import assume
from hypothesis import strategies as st

from simult_chess.core import geometry
from simult_chess.core.types import (
    Bookkeeping,
    CastlingRights,
    Color,
    Move,
    PieceType,
    Program,
    Reservation,
    Square,
    State,
    Token,
)

_EXTRA_TYPES: tuple[PieceType, ...] = ("p", "n", "b", "r", "q")
_ALL_SQUARES = [Square(file, rank) for file in range(8) for rank in range(8)]
_LAST_RANK = {Color.WHITE: 7, Color.BLACK: 0}


def build_state(
    board: Mapping[Token, Square],
    *,
    cooldown: frozenset[Token] = frozenset(),
    reservations_white: tuple[Reservation, ...] = (),
    reservations_black: tuple[Reservation, ...] = (),
    castling_rights: CastlingRights | None = None,
    no_progress_counter: int = 0,
    phase_index: int = 0,
) -> State:
    return State(
        board=board,
        cooldown=cooldown,
        reservations_white=reservations_white,
        reservations_black=reservations_black,
        bookkeeping=Bookkeeping(
            castling_rights=castling_rights or CastlingRights(),
            repetition_ledger={},
            no_progress_counter=no_progress_counter,
            phase_index=phase_index,
        ),
    )


@st.composite
def legal_scenarios(draw: st.DrawFn) -> tuple[State, Program, Program]:
    """A random *legal* `(state, program_white, program_black)` triple.

    Deliberately simplified relative to the full action grammar: each
    program is exactly one `Move`, chosen from the mover's own pseudo-legal
    trajectories on a sparse random board (two kings plus 0-8 extra
    non-king pieces, pawns excluded from the back ranks). This is enough
    surface for M1-M3 (purity, order-independence, χ-equivariance), which
    only need *some* legal transition to compare — not the full program
    grammar (reservations/castling/promotion get their own targeted
    integration tests in `tests/unit/test_phi.py`).
    """
    chosen_squares = draw(
        st.lists(st.sampled_from(_ALL_SQUARES), min_size=2, max_size=10, unique=True)
    )
    white_king_square, black_king_square, *extra_squares = chosen_squares

    white_king = Token(id=1, color=Color.WHITE, typ="k")
    black_king = Token(id=2, color=Color.BLACK, typ="k")
    board: dict[Token, Square] = {
        white_king: white_king_square,
        black_king: black_king_square,
    }

    next_id = 3
    for square in extra_squares:
        color = draw(st.sampled_from([Color.WHITE, Color.BLACK]))
        piece_type = draw(st.sampled_from(_EXTRA_TYPES))
        if piece_type == "p" and square.rank in (0, 7):
            continue  # pawns never legally live on the back ranks
        board[Token(id=next_id, color=color, typ=piece_type)] = square
        next_id += 1

    state = build_state(board)

    def program_for(color: Color) -> Program:
        candidates: list[Move] = []
        for token in state.board:
            if token.color is not color:
                continue
            for trajectory in geometry.pseudo_legal_trajectories(state, token):
                last_rank = _LAST_RANK[color]
                reaches_last_rank = (
                    token.typ == "p" and trajectory.destination.rank == last_rank
                )
                promotion = "q" if reaches_last_rank else None
                candidates.append(
                    Move(token=token, trajectory=trajectory, promotion=promotion)
                )
        assume(len(candidates) > 0)
        move = draw(st.sampled_from(candidates))
        return (move,)

    program_white = program_for(Color.WHITE)
    program_black = program_for(Color.BLACK)
    return state, program_white, program_black
