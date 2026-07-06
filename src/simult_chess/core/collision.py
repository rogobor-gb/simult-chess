"""Collision primitives (V)/(E), annihilation rank, and the color-swap
involution χ (spec §6.1, §6.3; INVARIANTS.md §1).
"""

from __future__ import annotations

from simult_chess.core.types import (
    Action,
    Bookkeeping,
    Cancel,
    Castle,
    CastlingRights,
    Move,
    Program,
    Reservation,
    Reserve,
    Square,
    State,
    Token,
    Trajectory,
)


def vertex_conflict(m1: Trajectory, m2: Trajectory) -> bool:
    """(V): the swept sets intersect (spec §6.1)."""
    return bool(m1.swept & m2.swept)


def edge_conflict(m1: Trajectory, m2: Trajectory) -> bool:
    """(E): a shared edge in opposite orientation — the head-on swap (spec §6.1).

    Not implied by (V): the swept sets of a swap are disjoint.
    """
    reversed_m2_edges = {(v, u) for u, v in m2.edges}
    return bool(m1.edges & reversed_m2_edges)


def conflicts(m1: Trajectory, m2: Trajectory) -> bool:
    """A pair conflicts iff (V) or (E) holds (spec §6.1)."""
    return vertex_conflict(m1, m2) or edge_conflict(m1, m2)


def annihilation_rank(i: int, j: int) -> tuple[int, int]:
    """:math:`r(W_i,B_j)=(\\max(i,j),\\min(i,j))`, spec §6.3."""
    return (max(i, j), min(i, j))


def mirror_square(square: Square) -> Square:
    """:math:`\\mu(c,r)=(c,7-r)`: vertical rank reflection, files fixed."""
    return Square(square.file, 7 - square.rank)


def mirror_token(token: Token) -> Token:
    """χ on a token: same id and type, opposite color."""
    return Token(id=token.id, color=token.color.opponent, typ=token.typ)


def mirror_trajectory(trajectory: Trajectory) -> Trajectory:
    """χ on a trajectory: every square mirrored under μ."""
    return Trajectory(
        path=tuple(mirror_square(square) for square in trajectory.path),
        is_jump=trajectory.is_jump,
    )


def mirror_reservation(reservation: Reservation) -> Reservation:
    """χ on a reservation: both tokens mirrored, age stamp unchanged."""
    return Reservation(
        defender=mirror_token(reservation.defender),
        protege=mirror_token(reservation.protege),
        age=reservation.age,
    )


def mirror_action(action: Action) -> Action:
    """χ on a single action: actor(s)/trajectory relabelled under μ + inversion."""
    if isinstance(action, Move):
        return Move(
            token=mirror_token(action.token),
            trajectory=mirror_trajectory(action.trajectory),
        )
    if isinstance(action, Reserve):
        return Reserve(
            defender=mirror_token(action.defender), protege=mirror_token(action.protege)
        )
    if isinstance(action, Castle):
        return Castle(side=action.side)  # files preserved; flanks are not swapped
    if isinstance(action, Cancel):
        return Cancel(reservation=mirror_reservation(action.reservation))
    raise TypeError(f"unknown action {action!r}")


def mirror_program(program: Program) -> Program:
    """χ on a program: every action relabelled (spec §0; INVARIANTS.md §1)."""
    return tuple(mirror_action(action) for action in program)


def mirror_state(state: State) -> State:
    """The color-swap involution χ on a state (INVARIANTS.md §1).

    Reflects ranks (μ), inverts every token's color, swaps :math:`R_W`/:math:`R_B`,
    and reflects castling rights kingside<->kingside / queenside<->queenside
    (flanks are not swapped, since files are preserved by μ). The repetition
    ledger and bookkeeping counters pass through unchanged: their remirrored
    semantics are a Phase 5 (M3, on Φ) concern, not this structural involution
    on State, whose own round trip (χ∘χ=id) does not depend on them.
    """
    new_board = {
        mirror_token(token): mirror_square(square)
        for token, square in state.board.items()
    }
    new_cooldown = frozenset(mirror_token(token) for token in state.cooldown)
    new_reservations_white = tuple(
        mirror_reservation(r) for r in state.reservations_black
    )
    new_reservations_black = tuple(
        mirror_reservation(r) for r in state.reservations_white
    )
    old_rights = state.bookkeeping.castling_rights
    new_rights = CastlingRights(
        white_kingside=old_rights.black_kingside,
        white_queenside=old_rights.black_queenside,
        black_kingside=old_rights.white_kingside,
        black_queenside=old_rights.white_queenside,
    )
    new_bookkeeping = Bookkeeping(
        castling_rights=new_rights,
        repetition_ledger=dict(state.bookkeeping.repetition_ledger),
        no_progress_counter=state.bookkeeping.no_progress_counter,
        phase_index=state.bookkeeping.phase_index,
    )
    return State(
        board=new_board,
        cooldown=new_cooldown,
        reservations_white=new_reservations_white,
        reservations_black=new_reservations_black,
        bookkeeping=new_bookkeeping,
    )
