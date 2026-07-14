"""The stage-game solver layer (spec §8.1-8.2, docs/DEVELOPMENT_addendum_v1.1.md
Phase 10, ruling A7).

Requires the optional `solver` extra (`numpy`, `scipy` — GPL-free, but kept
quarantined behind an extra like `oracle`'s `chess` dependency, same
pattern, so core/rules stay standard-library only). Nothing here is
imported by `simult_chess.core` or `simult_chess.rules`.
"""

from __future__ import annotations
