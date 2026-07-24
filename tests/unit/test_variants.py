"""Gate 14: every rejected A/B arm value stays playable as a named variant."""

from __future__ import annotations

import pytest

from simult_chess.rules.ruleset import RuleSet
from simult_chess.rules.variants import (
    BASELINE_NAME,
    FROZEN_V1_1,
    VARIANTS,
    describe_variants,
    get_variant,
    variant_names,
)


def test_baseline_variant_is_the_frozen_default() -> None:
    assert FROZEN_V1_1 == RuleSet()
    assert get_variant(BASELINE_NAME) == RuleSet()


def test_every_campaign_ab_arm_is_reachable_as_a_named_variant() -> None:
    """The Gate 14 DoD, asserted against the campaign's own run specs."""
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from simult_chess.harness.campaign import AB_ARM_DEFS

    registered = {variant.ruleset for variant in VARIANTS.values()}
    unreachable = [
        label for label, ruleset, _ in AB_ARM_DEFS if ruleset not in registered
    ]
    assert not unreachable, f"A/B arm values with no named variant: {unreachable}"


def test_non_baseline_variants_are_distinct_rulesets_and_fingerprints() -> None:
    """A variant that fingerprints like the baseline would let a record lie."""
    others = [name for name in variant_names() if name != BASELINE_NAME]
    assert others, "the registry holds no variants"
    fingerprints = {name: get_variant(name).fingerprint() for name in others}
    assert FROZEN_V1_1.fingerprint() not in fingerprints.values()
    assert len(set(fingerprints.values())) == len(fingerprints)


def test_declined_timed_annihilation_is_deliberately_unregistered() -> None:
    """`annihilation_reading="timed"` has no Stage A implementation (spec §13.2)."""
    assert all(
        variant.ruleset.annihilation_reading == "B" for variant in VARIANTS.values()
    )


def test_unknown_variant_names_the_alternatives() -> None:
    with pytest.raises(KeyError, match="unknown variant"):
        get_variant("no_such_variant")


def test_describe_variants_lists_every_name_with_its_fingerprint() -> None:
    listing = describe_variants()
    for name in variant_names():
        assert name in listing
        assert get_variant(name).fingerprint()[:12] in listing
