"""Self-play data generation (Phase 13b, design §2.4/§2.5).

Drives the SM-MCTS search through the invariant-checked phase-resolution
path (`invariants.harness.run_phase`, lenient mode -- the same primitive
`harness.selfplay.play_one_game` uses), so every self-play game is checked
for S0-S3 violations by construction rather than by a separate audit (the
13b DoD's "invariant harness clean over all self-play games" requirement).

**One search per phase yields both colours' moves** (design §2.4: "a single
search tree yields both moves of a phase") -- this is why self-play does
*not* reuse `harness.selfplay`'s `Agent`-based `play_one_game`: two
independent `LearnedAgent` calls (one per colour) would each run their own
search, duplicating the work for no benefit, since a simultaneous-move
node's search already produces both colours' strategies from one tree.
`LearnedAgent` (`learn.agent`) exists for match play against *other* agents
(the evaluation ladder), where each side's search genuinely is independent;
this module is self-play's own loop.

**Training targets (§2.4).** Slot-1's target is the search's average
strategy `bar_sigma_omega` -- a real, regret-matching-refined mixed
strategy. Slot-2 has no such refinement (the Stage-C scope decision:
slot-2 is sampled from the network's own masked conditional prior, not
independently regret-matched, `learn.search`'s module docstring). Its
training target here is therefore the **actually-played** slot-2 index (a
hard, one-hot supervised target -- ordinary behavioural cloning of the
self-play move), not a soft distribution; `learn.train`'s loss treats the
two slots accordingly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import torch

from simult_chess.core.types import Color, State
from simult_chess.core.violation import Violation
from simult_chess.interop.encoding import encode_state
from simult_chess.invariants.harness import run_phase
from simult_chess.learn.action_grid import decode_program, sample_index
from simult_chess.learn.agent import NetworkEvaluator
from simult_chess.learn.config import SearchConfig
from simult_chess.learn.net import SimultChessNet
from simult_chess.learn.search import make_root, run_simulations
from simult_chess.referee.observe import ObservationChannel
from simult_chess.rules.ruleset import RuleSet

_OUTCOME_VALUE: dict[str, float] = {
    "white_wins": 1.0,
    "black_wins": -1.0,
    "draw": 0.0,
}


@dataclass(frozen=True, slots=True)
class PhaseRecord:
    """One phase's training example (pre-phase state, per-colour targets)."""

    planes: npt.NDArray[np.float32]
    scalars: npt.NDArray[np.float32]
    white_slot1_target: dict[int, float]
    black_slot1_target: dict[int, float]
    white_slot1_played: int
    white_slot2_played: int
    black_slot1_played: int
    black_slot2_played: int


@dataclass(frozen=True, slots=True)
class SelfPlayGame:
    """One complete self-play game: its phase records and final outcome."""

    phases: tuple[PhaseRecord, ...]
    outcome: str
    rng_seed: int
    violations: tuple[Violation, ...]


def play_one_selfplay_game(
    initial_state: State,
    net: SimultChessNet,
    ruleset: RuleSet,
    search_config: SearchConfig,
    rng_seed: int,
    *,
    max_phases: int = 500,
    device: torch.device | None = None,
) -> SelfPlayGame:
    """Play one seeded self-play game, recording a `PhaseRecord` per phase.

    Determinism: the only randomness is `rng`, seeded from `rng_seed` --
    reproduces bit-for-bit given the same net weights (dev brief Phase 6's
    determinism contract, carried over to the learning system).
    """
    evaluator = NetworkEvaluator(net, device=device)
    rng = random.Random(rng_seed)
    channel = ObservationChannel()
    state = initial_state
    outcome = "ongoing"
    violations: list[Violation] = []
    phases: list[PhaseRecord] = []

    for _ in range(max_phases):
        root = make_root(state)
        run_simulations(
            root,
            ruleset,
            evaluator,
            search_config.simulations,
            rng,
            prior_weight=search_config.prior_weight,
        )
        assert root.white is not None and root.black is not None and (
            root.context is not None
        )

        white_target = root.white.average_strategy()
        black_target = root.black.average_strategy()
        w1 = sample_index(white_target, rng)
        b1 = sample_index(black_target, rng)
        first_white = root.white.actions[w1]
        first_black = root.black.actions[b1]

        white_slot2_dist = evaluator.slot2_prior(
            root.context, Color.WHITE, state, ruleset, w1, first_white
        )
        black_slot2_dist = evaluator.slot2_prior(
            root.context, Color.BLACK, state, ruleset, b1, first_black
        )
        w2 = sample_index(white_slot2_dist, rng)
        b2 = sample_index(black_slot2_dist, rng)

        program_white = decode_program(first_white, w2, state, Color.WHITE, ruleset)
        program_black = decode_program(first_black, b2, state, Color.BLACK, ruleset)

        planes, scalars = encode_state(state, ruleset)
        phases.append(
            PhaseRecord(
                planes=planes,
                scalars=scalars,
                white_slot1_target=white_target,
                black_slot1_target=black_target,
                white_slot1_played=w1,
                white_slot2_played=w2,
                black_slot1_played=b1,
                black_slot2_played=b2,
            )
        )

        white_commitment = channel.commit(Color.WHITE, program_white)
        black_commitment = channel.commit(Color.BLACK, program_black)
        revealed_white = channel.reveal(white_commitment)
        revealed_black = channel.reveal(black_commitment)

        result = run_phase(
            state,
            revealed_white,
            revealed_black,
            ruleset,
            mode="lenient",
            rng_seed=rng_seed,
        )
        violations.extend(result.violations)
        if result.phi_result is None:
            outcome = "aborted"
            break
        state = result.phi_result.state
        outcome = result.phi_result.outcome
        if outcome != "ongoing":
            break
    else:
        outcome = "phase_limit_reached"

    return SelfPlayGame(
        phases=tuple(phases),
        outcome=outcome,
        rng_seed=rng_seed,
        violations=tuple(violations),
    )


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """A `PhaseRecord` paired with its game's terminal outcome `z` (White's
    perspective, §2.4) -- what `learn.train`'s loss consumes."""

    phase: PhaseRecord
    z: float


@dataclass
class ReplayBuffer:
    """A fixed-capacity ring buffer of `TrainingExample`s (design §2.5's
    self-play outer loop: "generate games -> append to a replay buffer ->
    sample minibatches"). Games whose outcome isn't a genuine terminal
    (`phase_limit_reached`, `aborted`) contribute `z=0.0` -- undecided, not
    a draw claim, but the same neutral value a draw gets (§2.4's z in
    {-1,0,+1})."""

    capacity: int
    examples: list[TrainingExample] = field(default_factory=list)
    _next: int = 0

    def add_game(self, game: SelfPlayGame) -> None:
        z = _OUTCOME_VALUE.get(game.outcome, 0.0)
        for phase in game.phases:
            self._add(TrainingExample(phase=phase, z=z))

    def _add(self, example: TrainingExample) -> None:
        if len(self.examples) < self.capacity:
            self.examples.append(example)
        else:
            self.examples[self._next] = example
            self._next = (self._next + 1) % self.capacity

    def sample(self, batch_size: int, rng: random.Random) -> list[TrainingExample]:
        n = min(batch_size, len(self.examples))
        return rng.sample(self.examples, n)

    def __len__(self) -> int:
        return len(self.examples)
