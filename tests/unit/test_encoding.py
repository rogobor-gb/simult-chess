"""The pyspiel-free encoder (interop.encoding) is the source of truth for the
(21, 8, 8) planes + (7,) scalars tensor, and must produce the byte-identical
tensor the pyspiel observer serves (Phase 13b, ruling D5)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numpy")

from conftest import build_state  # noqa: E402

from simult_chess.core.types import (  # noqa: E402
    Color,
    Reservation,
    Square,
    Token,
)
from simult_chess.interop.encoding import (  # noqa: E402
    NUM_PLANES,
    NUM_SCALARS,
    encode_state,
)
from simult_chess.referee.setup import standard_starting_state  # noqa: E402
from simult_chess.rules.ruleset import RuleSet  # noqa: E402

RULESET = RuleSet()


def _reservation_state() -> object:
    white_king = Token(id=100, color=Color.WHITE, typ="k")
    black_king = Token(id=200, color=Color.BLACK, typ="k")
    d4_pawn = Token(id=1, color=Color.WHITE, typ="p")
    e3_pawn = Token(id=2, color=Color.WHITE, typ="p")
    reservation = Reservation(defender=e3_pawn, protege=d4_pawn, age=(0, 0))
    return build_state(
        {
            white_king: Square(0, 0),
            black_king: Square(7, 7),
            d4_pawn: Square(3, 3),
            e3_pawn: Square(4, 2),
        },
        reservations_white=(reservation,),
    )


def test_encode_state_shapes_and_standard_start_values() -> None:
    planes, scalars = encode_state(standard_starting_state(), RULESET)
    assert planes.shape == (NUM_PLANES, 8, 8) == (21, 8, 8)
    assert scalars.shape == (NUM_SCALARS,) == (7,)
    assert planes.dtype == np.float32
    assert planes[:12].sum() == 32  # 16 + 16 pieces
    assert planes[13:].sum() == 0.0  # no reservations at the opening
    assert list(scalars) == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 50.0]


def test_encode_state_pairing_offset() -> None:
    planes, _ = encode_state(_reservation_state(), RULESET)  # type: ignore[arg-type]
    # Defender e3 (file 4, rank 2), protege d4 (file 3, rank 3): pairing planes
    # 17 (Δfile) and 18 (Δrank) at the defender square, normalized by 7.
    assert planes[17, 2, 4] == pytest.approx((3 - 4) / 7.0)
    assert planes[18, 2, 4] == pytest.approx((3 - 2) / 7.0)


def test_encoder_matches_the_pyspiel_observer_byte_for_byte() -> None:
    pytest.importorskip("pyspiel")
    from simult_chess.interop.openspiel_adapter import SimultChessGame, SimultChessState

    for native in (standard_starting_state(), _reservation_state()):
        game = SimultChessGame()
        observer = game.make_py_observer()
        observer.set_from(SimultChessState(game, native), 0)  # type: ignore[arg-type]
        planes, scalars = encode_state(native, RULESET)  # type: ignore[arg-type]
        assert np.array_equal(observer.dict["planes"], planes)
        assert np.array_equal(observer.dict["scalars"], scalars)
