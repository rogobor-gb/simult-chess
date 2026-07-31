"""v3 18a'.5: the chi-antisymmetric/-symmetric architecture in
`learn.net.SimultChessNet` -- validated two ways, neither of which is
"trust the derivation":

1. `chi_transform` (a hand-derived tensor transform meant to equal
   `encode_state(mirror_state(s))`) is checked directly against the real
   `encode_state`/`mirror_state` over real self-play positions, not just
   asserted to be algebraically equivalent.
2. The network's forward-pass *identities* (`V(chi(s)) == -V(s)`,
   `slot1_black(chi(s)) == mirror(slot1_white(s))`) are checked to hold
   exactly (to floating-point tolerance), on a freshly-initialized
   (untrained) network -- since they are architectural, they must hold
   before any training, not just approximately after.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from simult_chess.agents.random_legal import random_legal_program
from simult_chess.core.collision import mirror_state
from simult_chess.core.phi import phi
from simult_chess.core.types import Color, State
from simult_chess.interop.encoding import encode_state
from simult_chess.learn.action_grid import MIRROR_PERMUTATION
from simult_chess.learn.config import NetConfig
from simult_chess.learn.net import SimultChessNet, chi_transform
from simult_chess.referee.setup import standard_starting_state
from simult_chess.rules.ruleset import RuleSet


def _rollout_states(n_phases: int) -> list[State]:
    """A handful of real, structurally varied positions (reservations
    included) from a short seeded self-play rollout -- not just the
    symmetric standard opening, which would leave every plane-pair swap
    trivially untested (nothing on those planes to swap)."""
    ruleset = RuleSet(n_actions=2)
    state = standard_starting_state()
    rng_white = random.Random(3)
    rng_black = random.Random(4)
    states = [state]
    for _ in range(n_phases):
        program_white = random_legal_program(state, Color.WHITE, ruleset, rng_white)
        program_black = random_legal_program(state, Color.BLACK, ruleset, rng_black)
        result = phi(state, program_white, program_black, ruleset)
        if result.outcome != "ongoing":
            break
        state = result.state
        states.append(state)
    return states


def test_chi_transform_matches_encode_state_of_mirror_state() -> None:
    ruleset = RuleSet(n_actions=2)
    states = _rollout_states(10)
    assert len(states) > 5  # sanity: the rollout didn't end immediately

    batch_planes = []
    batch_scalars = []
    expected_planes = []
    expected_scalars = []
    for state in states:
        planes, scalars = encode_state(state, ruleset)
        batch_planes.append(planes)
        batch_scalars.append(scalars)
        mirrored_planes, mirrored_scalars = encode_state(mirror_state(state), ruleset)
        expected_planes.append(mirrored_planes)
        expected_scalars.append(mirrored_scalars)

    planes_t = torch.from_numpy(np.stack(batch_planes))
    scalars_t = torch.from_numpy(np.stack(batch_scalars))
    got_planes, got_scalars = chi_transform(planes_t, scalars_t)

    np.testing.assert_allclose(
        got_planes.numpy(), np.stack(expected_planes), atol=1e-6
    )
    np.testing.assert_allclose(
        got_scalars.numpy(), np.stack(expected_scalars), atol=1e-6
    )


def test_chi_transform_is_its_own_inverse() -> None:
    ruleset = RuleSet()
    planes, scalars = encode_state(standard_starting_state(), ruleset)
    planes_t = torch.from_numpy(planes).unsqueeze(0)
    scalars_t = torch.from_numpy(scalars).unsqueeze(0)
    once_planes, once_scalars = chi_transform(planes_t, scalars_t)
    twice_planes, twice_scalars = chi_transform(once_planes, once_scalars)
    torch.testing.assert_close(twice_planes, planes_t)
    torch.testing.assert_close(twice_scalars, scalars_t)


def test_value_head_is_exactly_chi_antisymmetric_on_an_untrained_net() -> None:
    """V(chi(s)) == -V(s) must hold by construction, before any training --
    it is an architectural identity (v3 18a'.5), not a learned property."""
    torch.manual_seed(0)
    net = SimultChessNet(NetConfig(residual_blocks=2, filters=8))
    net.eval()
    ruleset = RuleSet(n_actions=2)
    states = _rollout_states(6)

    planes_list, scalars_list = [], []
    for state in states:
        planes, scalars = encode_state(state, ruleset)
        planes_list.append(planes)
        scalars_list.append(scalars)
    planes_t = torch.from_numpy(np.stack(planes_list))
    scalars_t = torch.from_numpy(np.stack(scalars_list))
    planes_chi, scalars_chi = chi_transform(planes_t, scalars_t)

    with torch.no_grad():
        _, _, value, _ = net(planes_t, scalars_t)
        _, _, value_chi, _ = net(planes_chi, scalars_chi)

    torch.testing.assert_close(value, -value_chi, atol=1e-5, rtol=1e-5)


def test_slot1_heads_are_exactly_chi_symmetrized_on_an_untrained_net() -> None:
    """slot1_black(chi(s)) == mirror(slot1_white(s)) (and symmetrically for
    white/black swapped) must hold by construction on a fresh net."""
    torch.manual_seed(0)
    net = SimultChessNet(NetConfig(residual_blocks=2, filters=8))
    net.eval()
    ruleset = RuleSet(n_actions=2)
    states = _rollout_states(6)

    planes_list, scalars_list = [], []
    for state in states:
        planes, scalars = encode_state(state, ruleset)
        planes_list.append(planes)
        scalars_list.append(scalars)
    planes_t = torch.from_numpy(np.stack(planes_list))
    scalars_t = torch.from_numpy(np.stack(scalars_list))
    planes_chi, scalars_chi = chi_transform(planes_t, scalars_t)

    with torch.no_grad():
        slot1_white, slot1_black, _, _ = net(planes_t, scalars_t)
        slot1_white_chi, slot1_black_chi, _, _ = net(planes_chi, scalars_chi)

    permutation = torch.tensor(MIRROR_PERMUTATION, dtype=torch.long)
    mirrored_white = slot1_white.index_select(1, permutation)
    mirrored_black = slot1_black.index_select(1, permutation)
    torch.testing.assert_close(slot1_black_chi, mirrored_white, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(slot1_white_chi, mirrored_black, atol=1e-5, rtol=1e-5)


def test_forward_output_shapes_and_policy_features_unchanged() -> None:
    """The external contract (shapes, policy_features == e(s) alone) that
    `NetworkEvaluator`/`slot2_logits` depend on must survive the 18a'.5
    rewrite untouched."""
    net = SimultChessNet(NetConfig(residual_blocks=2, filters=8))
    net.eval()
    ruleset = RuleSet()
    planes, scalars = encode_state(standard_starting_state(), ruleset)
    planes_t = torch.from_numpy(planes).unsqueeze(0)
    scalars_t = torch.from_numpy(scalars).unsqueeze(0)

    with torch.no_grad():
        slot1_white, slot1_black, value, policy_features = net(planes_t, scalars_t)

    from simult_chess.learn.action_grid import SLOT_SIZE

    assert slot1_white.shape == (1, SLOT_SIZE)
    assert slot1_black.shape == (1, SLOT_SIZE)
    assert value.shape == (1,)
    assert policy_features.shape[0] == 1

    # slot2_logits must still work off `policy_features` unchanged.
    a1_indices = torch.tensor([0])
    logits = net.slot2_logits(policy_features, a1_indices, Color.WHITE)
    assert logits.shape == (1, SLOT_SIZE + 1)
