from __future__ import annotations

import itertools

from conftest import build_state
from hypothesis import given, settings
from hypothesis import strategies as st

from simult_chess.core.moves import DeclaredMove
from simult_chess.core.phi import phi
from simult_chess.core.stages.defense import resolve_defense
from simult_chess.core.stages.defense_seq import resolve_defense_seq
from simult_chess.core.types import (
    Color,
    Move,
    PieceType,
    Reservation,
    Square,
    State,
    Token,
    Trajectory,
)
from simult_chess.rules.ruleset import RuleSet

RULESET = RuleSet()
RULESET_I = RuleSet(intermezzo_reading="i")


def test_m4_attacker_declaration_order_does_not_affect_defense_outcome() -> None:
    """M4 — the defensive outcome is invariant to the attacker's intra-program
    declaration order (spec §6.4 worked example, INVARIANTS.md M4).

    Fixed fixture (the d4/e3 worked example) with all k! = 2 orderings of
    the attacker's two captures, per INVARIANTS.md's own check description
    ("permute the attacker's intra-program order over all k! orderings").
    """
    white_king = Token(id=100, color=Color.WHITE, typ="k")
    black_king = Token(id=200, color=Color.BLACK, typ="k")
    d4_pawn = Token(id=1, color=Color.WHITE, typ="p")
    e3_pawn = Token(id=2, color=Color.WHITE, typ="p")
    rook1 = Token(id=3, color=Color.BLACK, typ="r")
    rook2 = Token(id=4, color=Color.BLACK, typ="r")
    state = build_state(
        {
            white_king: Square(0, 0),
            black_king: Square(7, 7),
            d4_pawn: Square(3, 3),
            e3_pawn: Square(4, 2),
            rook1: Square(3, 7),
            rook2: Square(4, 6),
        },
        reservations_white=(
            Reservation(defender=e3_pawn, protege=d4_pawn, age=(0, 0)),
        ),
    )
    rook1_path = (Square(3, 7), Square(3, 6), Square(3, 5), Square(3, 4), Square(3, 3))
    rook2_path = (Square(4, 6), Square(4, 5), Square(4, 4), Square(4, 3), Square(4, 2))
    rook1_move = Move(token=rook1, trajectory=Trajectory(path=rook1_path))
    rook2_move = Move(token=rook2, trajectory=Trajectory(path=rook2_path))
    king_trajectory = Trajectory(path=(Square(0, 0), Square(0, 1)))
    program_white = (Move(token=white_king, trajectory=king_trajectory),)

    results = [
        phi(state, program_white, ordering, RULESET).state
        for ordering in itertools.permutations((rook1_move, rook2_move))
    ]

    assert all(result == results[0] for result in results)
    board_ids = {t.id for t in results[0].board}
    assert d4_pawn.id not in board_ids  # the defense trades, it doesn't prevent
    assert rook1.id not in board_ids


# --- Hypothesis-driven M4, both readings (docs/DEVELOPMENT_addendum_v1.1.md
# Phase 11a DoD: "Both M4 branches green at >=200 hypothesis examples each").
#
# The generated pattern is the d4/e3 worked example's shape, translated
# across the board and varied over defender/protege/attacker piece types --
# always a single-step-diagonal (contact) reservation attacked by two
# straight-line movers, so admissibility and the attackers' paths stay valid
# by construction. Tested at the Stage B level directly (DeclaredMove tuples
# with explicit indices), sidestepping full-program/king-placement concerns
# that are orthogonal to what M4 is about.

_CONTACT_DEFENDER_TYPES: tuple[PieceType, ...] = ("p", "b", "q")
_PROTEGE_TYPES: tuple[PieceType, ...] = ("p", "n", "b", "r", "q")
_ATTACKER_TYPES: tuple[PieceType, ...] = ("r", "q")


def _straight_line(origin: Square, destination: Square) -> tuple[Square, ...]:
    step_file = (destination.file > origin.file) - (destination.file < origin.file)
    step_rank = (destination.rank > origin.rank) - (destination.rank < origin.rank)
    squares = [origin]
    file, rank = origin.file, origin.rank
    while (file, rank) != (destination.file, destination.rank):
        file, rank = file + step_file, rank + step_rank
        squares.append(Square(file, rank))
    return tuple(squares)


@st.composite
def defended_pair_scenarios(
    draw: st.DrawFn,
) -> tuple[State, Token, Token, DeclaredMove, DeclaredMove]:
    """A contact-defended pair (defender diagonally "behind" protege, the
    d4/e3 shape) plus two straight-line attackers, one on each square,
    translated and varied in piece type. Returns
    ``(state, defender, protege, attacker_on_protege, attacker_on_defender)``
    -- the caller assigns declaration indices to the two DeclaredMoves.
    """
    file_offset = draw(st.integers(min_value=-3, max_value=3))
    rank_offset = draw(st.integers(min_value=-2, max_value=0))
    defender_type = draw(st.sampled_from(_CONTACT_DEFENDER_TYPES))
    protege_type = draw(st.sampled_from(_PROTEGE_TYPES))
    attacker_on_protege_type = draw(st.sampled_from(_ATTACKER_TYPES))
    attacker_on_defender_type = draw(st.sampled_from(_ATTACKER_TYPES))

    def sq(file: int, rank: int) -> Square:
        return Square(file + file_offset, rank + rank_offset)

    protege_square = sq(3, 3)  # "d4"
    defender_square = sq(4, 2)  # "e3": one step northwest of the protege
    attacker_on_protege_start = sq(3, 7)
    attacker_on_defender_start = sq(4, 7)

    defender = Token(id=1, color=Color.WHITE, typ=defender_type)
    protege = Token(id=2, color=Color.WHITE, typ=protege_type)
    attacker_on_protege = Token(id=3, color=Color.BLACK, typ=attacker_on_protege_type)
    attacker_on_defender = Token(id=4, color=Color.BLACK, typ=attacker_on_defender_type)

    state = build_state(
        {
            defender: defender_square,
            protege: protege_square,
            attacker_on_protege: attacker_on_protege_start,
            attacker_on_defender: attacker_on_defender_start,
        },
        reservations_white=(
            Reservation(defender=defender, protege=protege, age=(0, 0)),
        ),
    )
    protege_attack = DeclaredMove(
        token=attacker_on_protege,
        trajectory=Trajectory(
            path=_straight_line(attacker_on_protege_start, protege_square)
        ),
        color=Color.BLACK,
        index=0,  # overwritten by the caller
        kind="move",
    )
    defender_attack = DeclaredMove(
        token=attacker_on_defender,
        trajectory=Trajectory(
            path=_straight_line(attacker_on_defender_start, defender_square)
        ),
        color=Color.BLACK,
        index=0,  # overwritten by the caller
        kind="move",
    )
    return state, defender, protege, protege_attack, defender_attack


def _reindexed(move: DeclaredMove, index: int) -> DeclaredMove:
    return DeclaredMove(
        token=move.token,
        trajectory=move.trajectory,
        color=move.color,
        index=index,
        kind=move.kind,
    )


@given(defended_pair_scenarios())
@settings(max_examples=200)
def test_m4_reading_ii_is_order_independent(
    scenario: tuple[State, Token, Token, DeclaredMove, DeclaredMove],
) -> None:
    """M4, "ii" branch: the defensive outcome does not depend on which of
    the two captures the attacker declares first (spec §6.4, the v1
    default)."""
    state, defender, protege, protege_attack, defender_attack = scenario
    protege_first = (
        _reindexed(protege_attack, 1),
        _reindexed(defender_attack, 2),
    )
    defender_first = (
        _reindexed(defender_attack, 1),
        _reindexed(protege_attack, 2),
    )

    reservations = state.reservations_white
    result_a = resolve_defense(
        protege_first, protege_first, state, reservations, (), RULESET
    )
    result_b = resolve_defense(
        defender_first, defender_first, state, reservations, (), RULESET
    )

    assert result_a.captured_tokens == result_b.captured_tokens
    assert result_a.survives(defender)
    assert not result_a.survives(protege)


@given(defended_pair_scenarios())
@settings(max_examples=200)
def test_m4_reading_i_is_order_dependent_as_specified(
    scenario: tuple[State, Token, Token, DeclaredMove, DeclaredMove],
) -> None:
    """M4, "i" branch: deliberately order-*dependent*, per spec §13.4's own
    specification -- not "anything goes". Declaring the protege-attack
    first reproduces Reading (ii)'s outcome (defense holds); declaring the
    defender-attack first strips the reservation before it can fire (both
    defended pieces fall, the attacker loses nothing)."""
    state, defender, protege, protege_attack, defender_attack = scenario
    protege_first = (
        _reindexed(protege_attack, 1),
        _reindexed(defender_attack, 2),
    )
    defender_first = (
        _reindexed(defender_attack, 1),
        _reindexed(protege_attack, 2),
    )

    reservations = state.reservations_white
    result_a = resolve_defense_seq(
        protege_first, protege_first, state, reservations, (), RULESET_I
    )
    result_b = resolve_defense_seq(
        defender_first, defender_first, state, reservations, (), RULESET_I
    )

    # Attack-protege-first: matches Reading (ii) -- defense holds.
    assert result_a.survives(defender)
    assert not result_a.survives(protege)
    assert not result_a.survives(protege_attack.token)
    assert result_a.survives(defender_attack.token)

    # Attack-defender-first: the reservation is disarmed before it can fire.
    assert not result_b.survives(defender)
    assert not result_b.survives(protege)
    assert result_b.survives(protege_attack.token)
    assert result_b.survives(defender_attack.token)

    # The two orderings must actually disagree -- this is the property that
    # is *false* for Reading (ii) and specified (not arbitrary) for (i).
    assert result_a.captured_tokens != result_b.captured_tokens
