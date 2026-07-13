from __future__ import annotations

from simult_chess.net.commitment import commitment_hash


def test_commitment_hash_is_deterministic() -> None:
    program = [{"kind": "castle", "side": "king"}]
    assert commitment_hash(b"salt", program) == commitment_hash(b"salt", program)


def test_commitment_hash_is_key_order_independent() -> None:
    a = [{"kind": "move", "token_id": 1, "path": [[0, 0], [0, 1]]}]
    b = [{"path": [[0, 0], [0, 1]], "token_id": 1, "kind": "move"}]
    assert commitment_hash(b"salt", a) == commitment_hash(b"salt", b)


def test_commitment_hash_is_sensitive_to_salt() -> None:
    program = [{"kind": "castle", "side": "king"}]
    assert commitment_hash(b"salt-a", program) != commitment_hash(b"salt-b", program)


def test_commitment_hash_is_sensitive_to_program_content() -> None:
    salt = b"fixed-salt"
    program_a = [{"kind": "castle", "side": "king"}]
    program_b = [{"kind": "castle", "side": "queen"}]
    assert commitment_hash(salt, program_a) != commitment_hash(salt, program_b)
