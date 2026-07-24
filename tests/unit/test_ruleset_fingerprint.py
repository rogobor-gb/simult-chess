"""Gate 14: the frozen v1.1 `RuleSet` fingerprint is a stable identity.

Everything downstream of the freeze -- game records (15d), the handshake
(15a), the Zenodo deposit (17) -- binds to `RuleSet.fingerprint()`. These
tests pin the two properties that binding needs: the digest is *stable*
across anything that is not a rules change, and it *moves* whenever a rule
does.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field, fields, make_dataclass, replace
from pathlib import Path

import pytest

from simult_chess.rules.ruleset import (
    FINGERPRINT_DOMAIN,
    NOT_RULE_BEARING,
    RULE_BEARING,
    RuleSet,
    canonical_ruleset_form,
    ruleset_fingerprint,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CAMPAIGN_REPORT = _REPO_ROOT / "reports" / "campaign_v1.md"
# Every document that states the freeze. A digest printed in any of them is a
# claim about the engine, so all three are checked against the engine.
_FREEZE_DOCS = (
    _CAMPAIGN_REPORT,
    _REPO_ROOT / "docs" / "simultaneous_chess_spec_v1.md",
    _REPO_ROOT / "docs" / "INVARIANTS.md",
)

# The frozen provisional v1.1 fingerprint (maintainer rulings C1-C3,
# 2026-07-24). Spelled out rather than recomputed: a test that recomputes the
# expected value from the object under test cannot notice the object changing.
FROZEN_V1_1_FINGERPRINT = (
    "bf2bb9dab0f020b107e5cfb3d964f825f08fbcdb1a1c8c729776670f30d1491c"
)


def test_frozen_defaults_have_the_recorded_fingerprint() -> None:
    assert RuleSet().fingerprint() == FROZEN_V1_1_FINGERPRINT


def test_fingerprint_is_deterministic_within_a_process() -> None:
    assert RuleSet().fingerprint() == RuleSet().fingerprint()


def test_fingerprint_is_byte_identical_across_fresh_interpreters() -> None:
    """DoD: reproduces across two fresh interpreters.

    `hash()` would not -- str hashing is salted per process (PYTHONHASHSEED),
    which is the trap this test exists to keep the implementation out of.
    """
    program = (
        "from simult_chess.rules.ruleset import RuleSet;"
        "print(RuleSet().fingerprint())"
    )
    digests = {
        subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for _ in range(2)
    }
    assert digests == {FROZEN_V1_1_FINGERPRINT}


def test_fingerprint_survives_field_reordering() -> None:
    """DoD: reordering the dataclass's field declarations is not a rules change."""
    baseline = RuleSet()
    reordered = make_dataclass(
        "ReorderedRuleSet",
        [
            (spec.name, object, field(default=getattr(baseline, spec.name),
                                      metadata=spec.metadata))
            for spec in reversed(fields(baseline))
        ],
        frozen=True,
    )
    assert [f.name for f in fields(reordered)] != [f.name for f in fields(baseline)]
    assert ruleset_fingerprint(reordered()) == baseline.fingerprint()


def test_every_rule_bearing_field_moves_the_fingerprint() -> None:
    """A rules change is never silent. One perturbation per field."""
    baseline = RuleSet()
    perturbations = {
        "n_actions": 3,
        "horizon": 30,
        "recapture_cooldown": False,
        "cancellation_enabled": False,
        "pawn_same_square_fizzle_scope": "any_same_square",
        "annihilation_reading": "timed",
        "intermezzo_reading": "i",
    }
    assert set(perturbations) == {f.name for f in fields(baseline)}, (
        "a RuleSet field was added or renamed without deciding whether it "
        "belongs in the fingerprint"
    )
    digests = {
        name: replace(baseline, **{name: value}).fingerprint()
        for name, value in perturbations.items()
    }
    assert baseline.fingerprint() not in digests.values()
    assert len(set(digests.values())) == len(digests), "two fields collide"


def test_non_rule_bearing_fields_are_excluded() -> None:
    """A solver tolerance is not a rule; it must not re-key existing records."""

    @dataclass(frozen=True)
    class WithTolerance:
        horizon: int = field(default=50, metadata=RULE_BEARING)
        lp_atol: float = field(default=1e-9, metadata=NOT_RULE_BEARING)

    assert ruleset_fingerprint(WithTolerance()) == ruleset_fingerprint(
        WithTolerance(lp_atol=1e-6)
    )
    assert "lp_atol" not in canonical_ruleset_form(WithTolerance())


def test_undeclared_field_refuses_to_fingerprint() -> None:
    """The fingerprint guesses neither way about an unmarked field."""

    @dataclass(frozen=True)
    class Undeclared:
        horizon: int = 50

    with pytest.raises(TypeError, match="declares neither RULE_BEARING"):
        ruleset_fingerprint(Undeclared())


def test_canonical_form_is_domain_prefixed_and_name_sorted() -> None:
    lines = RuleSet().canonical_form().splitlines()
    assert lines[0] == FINGERPRINT_DOMAIN
    names = [line.split("=", 1)[0] for line in lines[1:]]
    assert names == sorted(names)
    assert names == sorted(f.name for f in fields(RuleSet()))


@pytest.mark.parametrize("doc", _FREEZE_DOCS, ids=lambda path: path.name)
def test_freeze_documents_state_the_current_fingerprint(doc: Path) -> None:
    """No document may print a digest the engine does not produce."""
    digests = set(re.findall(r"\b[0-9a-f]{64}\b", doc.read_text(encoding="utf-8")))
    assert digests == {FROZEN_V1_1_FINGERPRINT}, (
        f"{doc.name} must state exactly the current frozen fingerprint"
    )


def test_campaign_report_freeze_block_names_every_frozen_value() -> None:
    """DoD: the report's freeze block and `rules/ruleset.py` agree.

    Enforced mechanically so the report cannot drift into claiming a freeze
    the code does not implement.
    """
    report = _CAMPAIGN_REPORT.read_text(encoding="utf-8")
    for line in RuleSet().canonical_form().splitlines()[1:]:
        name, value = line.split("=", 1)
        assert f"`{name}` | `{value}`" in report, (
            f"freeze block does not name the frozen value of {name}"
        )
