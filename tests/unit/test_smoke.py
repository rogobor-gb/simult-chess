"""Smoke test for Phase 0 (DEVELOPMENT.md Gate 0): the package must be importable."""

import simult_chess


def test_package_importable() -> None:
    assert simult_chess is not None
