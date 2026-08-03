"""v3 18b.3: policy-head representational capacity, checked directly rather
than assumed.

18a' localized the learning pipeline's defects to the *update rule* (H4/H5 --
a biased in-tree regret estimator and a policy target contaminated by the
network prior), not the architecture. This test makes that a checked claim,
not just an inference: fit a small net's slot-1 heads, by pure supervised
cross-entropy, to the max-entropy equilibrium `solver.entropy.
solve_max_entropy_equilibrium` computes on the Matching-Pennies dodge
fixture's exact (`solve_exact`-derived) stage matrix -- the *target* this
session's solver work can finally state precisely for a real position, not a
proxy. Pre-registered prediction, stated before running it: this succeeds
(the autoregressive slot-1 head is expressive enough to represent an
arbitrary categorical distribution over a handful of legal actions at one
state) -- a negative result here would mean the representational claim in
18a' was wrong, not merely that training dynamics need work."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

import torch.nn.functional as functional  # noqa: E402

from simult_chess.core.types import (  # noqa: E402
    Bookkeeping,
    CastlingRights,
    Color,
    Square,
    State,
    Token,
)
from simult_chess.interop.encoding import encode_state  # noqa: E402
from simult_chess.learn.action_grid import SLOT_SIZE, encode_action  # noqa: E402
from simult_chess.learn.config import NetConfig  # noqa: E402
from simult_chess.learn.net import SimultChessNet  # noqa: E402
from simult_chess.rules.ruleset import RuleSet  # noqa: E402
from simult_chess.solver.entropy import solve_max_entropy_equilibrium  # noqa: E402
from simult_chess.solver.exact import solve_exact  # noqa: E402

RULESET = RuleSet(n_actions=1)

_WHITE_KING = Token(id=1, color=Color.WHITE, typ="k")
_BLACK_KING = Token(id=2, color=Color.BLACK, typ="k")
_BLOCKER_A1 = Token(id=10, color=Color.WHITE, typ="p")
_BLOCKER_A3 = Token(id=11, color=Color.WHITE, typ="p")
_BLOCKER_B2 = Token(id=12, color=Color.WHITE, typ="p")
_BLOCKER_B3 = Token(id=13, color=Color.WHITE, typ="p")
_KNIGHT = Token(id=4, color=Color.BLACK, typ="n")


def _dodge_state() -> State:
    return State(
        board={
            _WHITE_KING: Square(0, 1),
            _BLACK_KING: Square(7, 7),
            _BLOCKER_A1: Square(0, 0),
            _BLOCKER_A3: Square(0, 2),
            _BLOCKER_B2: Square(1, 1),
            _BLOCKER_B3: Square(1, 2),
            _KNIGHT: Square(2, 2),
        },
        cooldown=frozenset(),
        reservations_white=(),
        reservations_black=(),
        bookkeeping=Bookkeeping(
            castling_rights=CastlingRights(False, False, False, False),
            repetition_ledger={},
            no_progress_counter=0,
            phase_index=0,
        ),
    )


def _dense_target(
    programs: tuple, weights, state: State
) -> torch.Tensor:
    dense = torch.zeros(SLOT_SIZE, dtype=torch.float32)
    for program, weight in zip(programs, weights, strict=True):
        # RuleSet(n_actions=1): every program is exactly one action, so this
        # index is unique per program -- no aliasing to worry about.
        index = encode_action(program[0], state)
        dense[index] += float(weight)
    return dense


def test_slot1_heads_fit_the_exact_max_entropy_equilibrium() -> None:
    graph = solve_exact(_dodge_state(), RULESET, max_nodes=5000, max_depth=1)
    root = graph.root
    assert root.stage_matrix is not None
    equilibrium = solve_max_entropy_equilibrium(root.stage_matrix)

    white_target = _dense_target(
        root.white_programs, equilibrium.row_strategy, root.state
    )
    black_target = _dense_target(
        root.black_programs, equilibrium.col_strategy, root.state
    )

    planes_np, scalars_np = encode_state(root.state, RULESET)
    planes = torch.from_numpy(planes_np).unsqueeze(0)
    scalars = torch.from_numpy(scalars_np).unsqueeze(0)
    white_target_batch = white_target.unsqueeze(0)
    black_target_batch = black_target.unsqueeze(0)

    net = SimultChessNet(NetConfig(residual_blocks=1, filters=8, policy_channels=4))
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-3)

    for _ in range(300):
        slot1_white_logits, slot1_black_logits, _value, _features = net(
            planes, scalars
        )
        white_loss = -(
            white_target_batch * functional.log_softmax(slot1_white_logits, dim=1)
        ).sum(dim=1).mean()
        black_loss = -(
            black_target_batch * functional.log_softmax(slot1_black_logits, dim=1)
        ).sum(dim=1).mean()
        loss = white_loss + black_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    net.eval()
    with torch.no_grad():
        slot1_white_logits, slot1_black_logits, _value, _features = net(
            planes, scalars
        )
        white_pred = functional.softmax(slot1_white_logits, dim=1).squeeze(0)
        black_pred = functional.softmax(slot1_black_logits, dim=1).squeeze(0)

    # L1 distance between the fitted distribution and the exact max-entropy
    # target, restricted to the programs that actually exist at this state
    # (the rest of the SLOT_SIZE=9026-way head is architecturally irrelevant
    # here -- nothing trains it, nothing should be asserted about it).
    white_indices = [encode_action(p[0], root.state) for p in root.white_programs]
    black_indices = [encode_action(p[0], root.state) for p in root.black_programs]
    white_error = (
        white_pred[white_indices] - white_target[white_indices]
    ).abs().sum().item()
    black_error = (
        black_pred[black_indices] - black_target[black_indices]
    ).abs().sum().item()

    assert white_error < 0.05
    assert black_error < 0.05
