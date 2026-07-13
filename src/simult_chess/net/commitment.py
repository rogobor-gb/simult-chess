"""Salted hash commitment for a declared program (spec §11.5, Phase 8).

Fairness without a trusted arbiter requires that neither peer can change
its program after seeing the other's -- so each phase, a peer first sends
a commitment (hash of its program plus a random salt), and only *after*
both commitments are exchanged does either side reveal the salt and
program, which the receiver re-hashes and checks against the
already-received commitment.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def commitment_hash(salt: bytes, program_json: list[dict[str, Any]]) -> str:
    """The commitment for `program_json` under `salt`: a hex SHA-256 digest.

    `program_json` is serialized with sorted keys and no incidental
    whitespace so the same program always hashes identically regardless of
    dict-ordering quirks.
    """
    canonical = json.dumps(program_json, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(salt + canonical.encode()).hexdigest()
