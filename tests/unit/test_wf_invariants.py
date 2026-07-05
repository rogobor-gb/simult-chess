from __future__ import annotations

from conftest import make_bookkeeping

from simult_chess.core.types import (
    Bookkeeping,
    CastlingRights,
    Color,
    Reservation,
    Square,
    State,
    Token,
)
from simult_chess.invariants.checks import (
    check_all_state,
    check_wf1_occupancy_injectivity,
    check_wf2_domain,
    check_wf2_type_constancy,
    check_wf3_cooldown_membership,
    check_wf4_king_count,
    check_wf5_reservation_order,
    check_wf6_reservation_referential_integrity,
    check_wf7_bookkeeping_monotone,
    check_wf7_bookkeeping_ranges,
)
from simult_chess.rules.ruleset import RuleSet


def test_wf1_clean_state_has_no_violations(minimal_state: State) -> None:
    assert check_wf1_occupancy_injectivity(minimal_state) == []


def test_wf1_catches_two_tokens_on_one_square() -> None:
    a = Token(id=1, color=Color.WHITE, typ="r")
    b = Token(id=2, color=Color.BLACK, typ="n")
    king_w = Token(id=3, color=Color.WHITE, typ="k")
    king_b = Token(id=4, color=Color.BLACK, typ="k")
    state = State(
        board={
            a: Square(0, 0),
            b: Square(0, 0),  # same square as `a` — malformed
            king_w: Square(4, 0),
            king_b: Square(4, 7),
        },
        cooldown=frozenset(),
        reservations_white=(),
        reservations_black=(),
        bookkeeping=make_bookkeeping(),
    )
    violations = check_wf1_occupancy_injectivity(state)
    assert len(violations) == 1
    assert violations[0].invariant_id == "WF1"


def test_wf2_domain_catches_invalid_type() -> None:
    bad = Token(id=1, color=Color.WHITE, typ="x")  # type: ignore[arg-type]
    state = State(
        board={bad: Square(0, 0)},
        cooldown=frozenset(),
        reservations_white=(),
        reservations_black=(),
        bookkeeping=make_bookkeeping(),
    )
    violations = check_wf2_domain(state)
    assert len(violations) == 1 and violations[0].invariant_id == "WF2"


def test_wf2_type_constancy_allows_recorded_promotion() -> None:
    pawn = Token(id=1, color=Color.WHITE, typ="p")
    queen = Token(id=1, color=Color.WHITE, typ="q")
    before = State(
        board={pawn: Square(0, 6)},
        cooldown=frozenset(),
        reservations_white=(),
        reservations_black=(),
        bookkeeping=make_bookkeeping(phase_index=0),
    )
    after = State(
        board={queen: Square(0, 7)},
        cooldown=frozenset({queen}),
        reservations_white=(),
        reservations_black=(),
        bookkeeping=make_bookkeeping(phase_index=1),
    )
    assert check_wf2_type_constancy(before, after, promoted={1}) == []


def test_wf2_type_constancy_flags_unrecorded_type_change() -> None:
    pawn = Token(id=1, color=Color.WHITE, typ="p")
    queen = Token(id=1, color=Color.WHITE, typ="q")
    before = State(
        board={pawn: Square(0, 6)},
        cooldown=frozenset(),
        reservations_white=(),
        reservations_black=(),
        bookkeeping=make_bookkeeping(phase_index=0),
    )
    after = State(
        board={queen: Square(0, 7)},
        cooldown=frozenset(),
        reservations_white=(),
        reservations_black=(),
        bookkeeping=make_bookkeeping(phase_index=1),
    )
    violations = check_wf2_type_constancy(before, after)
    assert len(violations) == 1 and violations[0].invariant_id == "WF2"


def test_wf3_rejects_cooled_pawn_and_king() -> None:
    pawn = Token(id=1, color=Color.WHITE, typ="p")
    king = Token(id=2, color=Color.WHITE, typ="k")
    knight = Token(id=3, color=Color.WHITE, typ="n")
    state = State(
        board={pawn: Square(0, 1), king: Square(4, 0), knight: Square(1, 0)},
        cooldown=frozenset({pawn, king, knight}),
        reservations_white=(),
        reservations_black=(),
        bookkeeping=make_bookkeeping(),
    )
    violations = check_wf3_cooldown_membership(state)
    assert len(violations) == 2  # pawn (id=1) and king (id=2) flagged; knight is fine


def test_wf3_rejects_dead_token_in_cooldown() -> None:
    ghost = Token(id=99, color=Color.WHITE, typ="n")
    state = State(
        board={},
        cooldown=frozenset({ghost}),
        reservations_white=(),
        reservations_black=(),
        bookkeeping=make_bookkeeping(),
    )
    violations = check_wf3_cooldown_membership(state)
    assert len(violations) == 1


def test_wf4_requires_exactly_one_king_per_color(minimal_state: State) -> None:
    assert check_wf4_king_count(minimal_state) == []


def test_wf4_catches_missing_king() -> None:
    king_w = Token(id=1, color=Color.WHITE, typ="k")
    state = State(
        board={king_w: Square(4, 0)},
        cooldown=frozenset(),
        reservations_white=(),
        reservations_black=(),
        bookkeeping=make_bookkeeping(),
    )
    violations = check_wf4_king_count(state)
    assert len(violations) == 1
    assert check_wf4_king_count(state, allow_terminal=True) == []


def test_wf4_catches_extra_king() -> None:
    king_w1 = Token(id=1, color=Color.WHITE, typ="k")
    king_w2 = Token(id=2, color=Color.WHITE, typ="k")
    king_b = Token(id=3, color=Color.BLACK, typ="k")
    state = State(
        board={king_w1: Square(4, 0), king_w2: Square(3, 0), king_b: Square(4, 7)},
        cooldown=frozenset(),
        reservations_white=(),
        reservations_black=(),
        bookkeeping=make_bookkeeping(),
    )
    violations = check_wf4_king_count(state)
    assert len(violations) == 1


def test_wf5_requires_strict_age_order() -> None:
    d = Token(id=1, color=Color.WHITE, typ="p")
    q = Token(id=2, color=Color.WHITE, typ="r")
    out_of_order = (
        Reservation(defender=d, protege=q, age=(1, 1)),
        Reservation(defender=d, protege=q, age=(1, 0)),
    )
    state = State(
        board={d: Square(0, 0), q: Square(1, 0)},
        cooldown=frozenset(),
        reservations_white=out_of_order,
        reservations_black=(),
        bookkeeping=make_bookkeeping(),
    )
    violations = check_wf5_reservation_order(state)
    assert any(v.invariant_id == "WF5" for v in violations)


def test_wf5_requires_globally_unique_age_stamps() -> None:
    d1 = Token(id=1, color=Color.WHITE, typ="p")
    q1 = Token(id=2, color=Color.WHITE, typ="r")
    d2 = Token(id=3, color=Color.BLACK, typ="p")
    q2 = Token(id=4, color=Color.BLACK, typ="r")
    state = State(
        board={d1: Square(0, 0), q1: Square(1, 0), d2: Square(0, 7), q2: Square(1, 7)},
        cooldown=frozenset(),
        reservations_white=(Reservation(defender=d1, protege=q1, age=(1, 0)),),
        reservations_black=(Reservation(defender=d2, protege=q2, age=(1, 0)),),
        bookkeeping=make_bookkeeping(),
    )
    violations = check_wf5_reservation_order(state)
    assert any("unique" in v.detail for v in violations)


def test_wf6_valid_reservation_has_no_violations() -> None:
    d = Token(id=1, color=Color.WHITE, typ="p")
    q = Token(id=2, color=Color.WHITE, typ="r")
    state = State(
        board={d: Square(0, 0), q: Square(1, 0)},
        cooldown=frozenset(),
        reservations_white=(Reservation(defender=d, protege=q, age=(1, 0)),),
        reservations_black=(),
        bookkeeping=make_bookkeeping(),
    )
    assert check_wf6_reservation_referential_integrity(state) == []


def test_wf6_catches_dead_defender_color_mismatch_and_self_reservation() -> None:
    d = Token(id=1, color=Color.WHITE, typ="p")
    q = Token(id=2, color=Color.WHITE, typ="r")
    enemy = Token(id=3, color=Color.BLACK, typ="r")
    ghost = Token(id=99, color=Color.WHITE, typ="n")
    state = State(
        board={d: Square(0, 0), q: Square(1, 0), enemy: Square(2, 0)},
        cooldown=frozenset(),
        reservations_white=(
            Reservation(defender=ghost, protege=q, age=(1, 0)),  # dead defender
            Reservation(defender=d, protege=enemy, age=(1, 1)),  # color mismatch
            Reservation(defender=d, protege=d, age=(1, 2)),  # self-reservation
        ),
        reservations_black=(),
        bookkeeping=make_bookkeeping(),
    )
    violations = check_wf6_reservation_referential_integrity(state)
    assert len(violations) == 3


def test_wf7_ranges_enforced() -> None:
    ruleset = RuleSet(horizon=50)
    state = State(
        board={},
        cooldown=frozenset(),
        reservations_white=(),
        reservations_black=(),
        bookkeeping=make_bookkeeping(no_progress_counter=51, phase_index=0),
    )
    violations = check_wf7_bookkeeping_ranges(state, ruleset)
    assert len(violations) == 1


def test_wf7_monotone_castling_rights_and_phase_increment() -> None:
    before = State(
        board={},
        cooldown=frozenset(),
        reservations_white=(),
        reservations_black=(),
        bookkeeping=Bookkeeping(
            castling_rights=CastlingRights(white_kingside=False),
            repetition_ledger={},
            no_progress_counter=0,
            phase_index=5,
        ),
    )
    regained_rights = State(
        board={},
        cooldown=frozenset(),
        reservations_white=(),
        reservations_black=(),
        bookkeeping=Bookkeeping(
            castling_rights=CastlingRights(white_kingside=True),
            repetition_ledger={},
            no_progress_counter=0,
            phase_index=6,
        ),
    )
    violations = check_wf7_bookkeeping_monotone(before, regained_rights)
    assert len(violations) == 1

    skipped_phase = State(
        board={},
        cooldown=frozenset(),
        reservations_white=(),
        reservations_black=(),
        bookkeeping=Bookkeeping(
            castling_rights=CastlingRights(white_kingside=False),
            repetition_ledger={},
            no_progress_counter=0,
            phase_index=7,
        ),
    )
    violations = check_wf7_bookkeeping_monotone(before, skipped_phase)
    assert len(violations) == 1


def test_check_all_state_passes_on_minimal_legal_state(minimal_state: State) -> None:
    assert check_all_state(minimal_state, RuleSet()) == []
