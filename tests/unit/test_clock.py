"""Phase 15b: concurrent-bank time control — parsing, bonus rules, banks, flags."""

from __future__ import annotations

import pytest

from simult_chess.net.clock import (
    Banks,
    TimeControl,
    apply_phase,
    entry_hash,
    initial_banks,
    race_bonus,
)

# --- parsing / formatting -----------------------------------------------------


def test_parse_basic() -> None:
    tc = TimeControl.parse("3|0|2")
    assert tc.initial_bank == 180.0
    assert tc.increment == 0.0
    assert tc.bonus == 2.0
    assert tc.bonus_rule == "capped_difference"


def test_parse_with_increment_and_rule() -> None:
    tc = TimeControl.parse("3|2|2/dead_zone")
    assert tc.increment == 2.0
    assert tc.bonus_rule == "dead_zone"


def test_parse_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="minutes"):
        TimeControl.parse("3|2")
    with pytest.raises(ValueError, match="non-numeric"):
        TimeControl.parse("a|b|c")
    with pytest.raises(ValueError, match="unknown bonus rule"):
        TimeControl.parse("3|0|2/nonsense")


def test_format_round_trips_the_rule() -> None:
    assert TimeControl.parse("3|0|2").format() == "3|0|2"
    assert TimeControl.parse("3|2|2/winner_take_all").format() == (
        "3|2|2/winner_take_all"
    )


# --- bonus rules --------------------------------------------------------------


def test_capped_difference_refunds_only_the_saved_gap() -> None:
    tc = TimeControl(initial_bank=180, bonus=2.0)
    # Faster by 1.2s, under the 2s cap: refunded exactly the gap.
    assert race_bonus(3.0, 4.2, tc) == pytest.approx(1.2)
    # Faster by more than the cap: refunded the cap.
    assert race_bonus(1.0, 5.0, tc) == 2.0
    # Slower or tied: nothing.
    assert race_bonus(4.2, 3.0, tc) == 0.0
    assert race_bonus(3.0, 3.0, tc) == 0.0


def test_winner_take_all_is_all_or_nothing() -> None:
    tc = TimeControl(initial_bank=180, bonus=2.0, bonus_rule="winner_take_all")
    assert race_bonus(1.0, 5.0, tc) == 2.0
    assert race_bonus(4.99, 5.0, tc) == 2.0  # a hair faster still wins the lot
    assert race_bonus(5.0, 5.0, tc) == 0.0


def test_dead_zone_pays_only_a_decisive_gap() -> None:
    tc = TimeControl(
        initial_bank=180, bonus=2.0, bonus_rule="dead_zone", dead_zone=5.0
    )
    assert race_bonus(1.0, 3.0, tc) == 0.0  # 2s gap < 5s dead zone: nothing
    assert race_bonus(1.0, 7.0, tc) == 2.0  # 6s gap: capped-difference applies
    assert race_bonus(7.0, 1.0, tc) == 0.0  # slower: nothing


def test_none_rule_pays_nothing() -> None:
    tc = TimeControl(initial_bank=180, bonus=2.0, bonus_rule="none")
    assert race_bonus(1.0, 5.0, tc) == 0.0


# --- banks and flag-fall ------------------------------------------------------


def test_bank_update_applies_increment_and_bonus() -> None:
    tc = TimeControl(initial_bank=180, increment=2.0, bonus=2.0)
    banks, entry, flag = apply_phase(
        initial_banks(tc), tc, think_white=1.0, think_black=4.0, phase_index=0
    )
    assert flag is None
    # White faster by 3s -> capped 2s bonus + 2s increment - 1s spent.
    assert banks.white == pytest.approx(180 - 1 + 2 + 2)
    # Black slower -> no bonus, +2 increment - 4 spent.
    assert banks.black == pytest.approx(180 - 4 + 2)
    assert entry.bonus_white == 2.0 and entry.bonus_black == 0.0


def test_racing_to_zero_is_self_defeating_under_capped_difference() -> None:
    tc = TimeControl(initial_bank=180, bonus=2.0)
    # Both slam instantly: the gap is ~0 so neither gains a bonus.
    _banks, entry, _flag = apply_phase(
        initial_banks(tc), tc, think_white=0.01, think_black=0.01, phase_index=0
    )
    assert entry.bonus_white == 0.0 and entry.bonus_black == 0.0


def test_flag_fall_when_a_bank_would_go_nonpositive() -> None:
    tc = TimeControl(initial_bank=180, bonus=2.0)
    banks = Banks(white=3.0, black=100.0)
    _after, _entry, flag = apply_phase(
        banks, tc, think_white=3.5, think_black=2.0, phase_index=9
    )
    assert flag is not None
    assert flag.white_flagged and not flag.black_flagged


def test_double_flag_is_detected() -> None:
    tc = TimeControl(initial_bank=180)
    banks = Banks(white=1.0, black=1.0)
    _after, _entry, flag = apply_phase(
        banks, tc, think_white=2.0, think_black=2.0, phase_index=5
    )
    assert flag is not None
    assert flag.white_flagged and flag.black_flagged


def test_no_bonus_or_increment_on_a_flagging_phase() -> None:
    tc = TimeControl(initial_bank=180, increment=5.0, bonus=2.0)
    banks = Banks(white=1.0, black=100.0)
    _after, entry, flag = apply_phase(
        banks, tc, think_white=2.0, think_black=0.5, phase_index=0
    )
    assert flag is not None
    assert entry.bonus_black == 0.0  # game ended; Black earns nothing


# --- ledger hash --------------------------------------------------------------


def test_entry_hash_is_stable_and_sensitive() -> None:
    tc = TimeControl(initial_bank=180, bonus=2.0)
    _b, entry, _f = apply_phase(
        initial_banks(tc), tc, think_white=1.0, think_black=4.0, phase_index=0
    )
    # Same measured values -> identical hash (both peers agree).
    _b2, entry2, _f2 = apply_phase(
        initial_banks(tc), tc, think_white=1.0, think_black=4.0, phase_index=0
    )
    assert entry_hash(entry) == entry_hash(entry2)
    # A different think time -> different hash.
    _b3, entry3, _f3 = apply_phase(
        initial_banks(tc), tc, think_white=1.5, think_black=4.0, phase_index=0
    )
    assert entry_hash(entry) != entry_hash(entry3)
