"""pyspiel simultaneous-move adapter (Phase 12, docs/DEVELOPMENT_addendum_
v1.1.md).

Registers this game with OpenSpiel as `simult_chess.interop.openspiel_
adapter.SimultChessGame`/`SimultChessState`, `short_name="simult_chess"`.
The referee/Φ acquire no OpenSpiel dependency: this module only *calls*
`simult_chess.core.phi.phi` and the legality/candidate-generation helpers
that already exist for the stdlib-only agents.

Program-indexing scheme (spec §4.4's L(s,pi), the addendum's own phrase)
----------------------------------------------------------------------
pyspiel requires a fixed-size discrete action space per player
(`Game.num_distinct_actions()`), but our real "action" per decision phase
is an entire declared `Program` (up to `RuleSet.n_actions`, drawn from a
combinatorially large and state-dependent set). Following the same pattern
OpenSpiel's own custom Python games use for large/variable action sets
(e.g. its C++ chess implementation): a pyspiel action integer is an index
into `enumerate_legal_programs(state, color, ruleset)`, computed fresh from
the *current* state -- the integer's meaning is state-dependent (index 5
might be a rook move in one state and a reservation in another), which is
standard practice here and exactly what "delegating to L(s,pi) via a
program-indexing scheme" means: `_legal_actions` is `range(len(programs))`,
`_apply_actions` re-enumerates and looks the chosen indices up.

`enumerate_legal_programs` is deliberately *exhaustive*, unlike the solver
layer's pruned/sampled `solver/supports.py` (a different consumer with a
different requirement: an LP support can drop candidates, a search
algorithm driven through this adapter must see the true legal set). It also
expands every promotion choice explicitly, unlike `agents/candidates.py`'s
`move_candidates` (which samples one random promotion per pawn-to-last-rank
trajectory for agent use) -- full fidelity to L(s,pi) requires all of them.

`_MAX_DISTINCT_ACTIONS` is an empirically-set ceiling, not a real
architectural limit: at the standard start and after a handful of random
plies, |legal programs| for one color is in the ~1000-1600 range (<100ms to
enumerate); a deliberately reservation-dense contrived stress position (32
same-color queens) reaches ~32,000 (~2.5s). `_MAX_DISTINCT_ACTIONS` gives
headroom over that and `_apply_actions` asserts against it rather than
silently truncating, so any true violation is a loud, investigable error,
not a silent correctness gap.

Tensor observation encoding (reused by Phase 13's `docs/LEARNING_DESIGN.md`
once written -- this module is the source of truth until then)
----------------------------------------------------------------------
A flat `float32` vector, `dict`-viewed as:
- `"planes"`, shape `(17, 8, 8)`: 12 board-occupancy planes (one per
  (color, type) pair, spec §1.1's `col`/`typ`), 1 cooldown plane (spec §7),
  and 4 reservation-actor planes -- `(white_defenders, white_proteges,
  black_defenders, black_proteges)`, one square marked per live token that
  currently plays that role in >=1 active reservation. This is a
  simplification flagged for Phase 13 to revisit if needed: it marks
  *where* reservation actors stand, not *which* defender-protege pairing,
  since a full pairwise (square, square) relation does not fit a per-square
  plane encoding.
- `"scalars"`, shape `(7,)`: the four `CastlingRights` booleans (spec
  §1.2's eta), the no-progress counter nu and horizon H (spec §10, T4), and
  phase-index parity (`phase_index % 2`).
"""

from __future__ import annotations

import numpy as np
import pyspiel

from simult_chess.core import geometry, legality
from simult_chess.core.phi import phi
from simult_chess.core.types import (
    Action,
    Castle,
    CastleSide,
    Color,
    Move,
    PieceType,
    Program,
    Reserve,
    Square,
    State,
)
from simult_chess.referee.setup import standard_starting_state
from simult_chess.rules.ruleset import RuleSet

_NUM_PLAYERS = 2
_MAX_DISTINCT_ACTIONS = 50_000
_MAX_GAME_LENGTH = 500  # matches harness/selfplay.py's own default cap

_PIECE_TYPES: tuple[PieceType, ...] = ("p", "n", "b", "r", "q", "k")
_PROMOTABLE: tuple[PieceType, ...] = ("n", "b", "r", "q")
_LAST_RANK = {Color.WHITE: 7, Color.BLACK: 0}
_CASTLE_SIDES: tuple[CastleSide, CastleSide] = ("king", "queen")

_TERMINAL_RETURNS = {
    "white_wins": (1.0, -1.0),
    "black_wins": (-1.0, 1.0),
    "draw": (0.0, 0.0),
}

_GAME_TYPE = pyspiel.GameType(
    short_name="simult_chess",
    long_name="Simultaneous Chess",
    dynamics=pyspiel.GameType.Dynamics.SIMULTANEOUS,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=False,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={},
)


def _exhaustive_move_and_castle_candidates(state: State, color: Color) -> list[Action]:
    """Like `agents.candidates.move_candidates`, but expands *every*
    promotion choice for a pawn reaching the last rank instead of sampling
    one -- full fidelity to L(s,pi) (spec §4.2, §6.5), needed since a
    pyspiel action must faithfully index the true legal-program set."""
    candidates: list[Action] = []
    for token in state.board:
        if token.color is not color or token in state.cooldown:
            continue
        for trajectory in geometry.pseudo_legal_trajectories(state, token):
            if token.typ == "p" and trajectory.destination.rank == _LAST_RANK[color]:
                for promotion in _PROMOTABLE:
                    candidates.append(
                        Move(token=token, trajectory=trajectory, promotion=promotion)
                    )
            else:
                candidates.append(
                    Move(token=token, trajectory=trajectory, promotion=None)
                )
    for side in _CASTLE_SIDES:
        if geometry.castle_move(state, color, side) is not None:
            candidates.append(Castle(side=side))
    return candidates


def _exhaustive_reserve_candidates(state: State, color: Color) -> list[Action]:
    """Every individually-admissible Reserve action for `color` (spec §4.3).
    Identical to `agents.candidates.reserve_candidates`; re-derived here
    (not imported) so this module's enumeration is self-contained and
    reviewable as one exhaustive-by-construction unit."""
    candidates: list[Action] = []
    for defender in state.board:
        if defender.color is not color or defender in state.cooldown:
            continue
        for protege in state.board:
            if protege.color is not color or protege is defender:
                continue
            target = state.board[protege]
            pattern = geometry.capturing_pattern_trajectory(state, defender, target)
            if pattern is not None:
                candidates.append(Reserve(defender=defender, protege=protege))
    return candidates


def enumerate_legal_programs(
    state: State, color: Color, ruleset: RuleSet
) -> list[Program]:
    """Every legal program for `color` at `state` (spec §4.4's L(s,pi)),
    exhaustively -- see the module docstring for why this differs from
    both `agents/candidates.py` (samples one promotion) and
    `solver/supports.py` (deliberately pruned)."""
    pool = [
        *_exhaustive_move_and_castle_candidates(state, color),
        *_exhaustive_reserve_candidates(state, color),
    ]
    programs: list[Program] = []
    for action in pool:
        single: Program = (action,)
        if legality.is_legal_program(state, single, color, ruleset):
            programs.append(single)
    if ruleset.n_actions >= 2:
        for i, first in enumerate(pool):
            for j, second in enumerate(pool):
                if i == j:
                    continue
                pair: Program = (first, second)
                if legality.is_legal_program(state, pair, color, ruleset):
                    programs.append(pair)
    assert len(programs) <= _MAX_DISTINCT_ACTIONS, (
        f"{len(programs)} legal programs exceeds _MAX_DISTINCT_ACTIONS "
        f"({_MAX_DISTINCT_ACTIONS}) -- raise the ceiling, this is an "
        "empirical bound, not an architectural one (see module docstring)"
    )
    return programs


class SimultChessGame(pyspiel.Game):  # type: ignore[misc]
    """The game, from which states and observers can be made."""

    def __init__(self, params: dict[str, object] | None = None) -> None:
        self._ruleset = RuleSet()
        super().__init__(
            _GAME_TYPE,
            pyspiel.GameInfo(
                num_distinct_actions=_MAX_DISTINCT_ACTIONS,
                max_chance_outcomes=0,
                num_players=_NUM_PLAYERS,
                min_utility=-1.0,
                max_utility=1.0,
                utility_sum=0.0,
                max_game_length=_MAX_GAME_LENGTH,
            ),
            params or {},
        )

    @property
    def ruleset(self) -> RuleSet:
        return self._ruleset

    def new_initial_state(self) -> SimultChessState:
        return SimultChessState(self, standard_starting_state())

    def make_py_observer(
        self, iig_obs_type: object = None, params: object = None
    ) -> SimultChessObserver:
        del iig_obs_type
        return SimultChessObserver(params)


class SimultChessState(pyspiel.State):  # type: ignore[misc]
    """Current state of the game: a thin wrapper around the native `State`
    plus the per-phase legal-program enumerations pyspiel's action integers
    index into."""

    def __init__(self, game: SimultChessGame, state: State) -> None:
        super().__init__(game)
        self._ruleset = game.ruleset
        self._state = state
        self._outcome = "ongoing"
        self._returns = [0.0, 0.0]
        self._cached_programs: dict[Color, list[Program]] = {}

    def _legal_programs(self, color: Color) -> list[Program]:
        cached = self._cached_programs.get(color)
        if cached is None:
            cached = enumerate_legal_programs(self._state, color, self._ruleset)
            self._cached_programs[color] = cached
        return cached

    @staticmethod
    def _color_of(player: int) -> Color:
        return Color.WHITE if player == 0 else Color.BLACK

    def current_player(self) -> int:
        if self._outcome != "ongoing":
            return int(pyspiel.PlayerId.TERMINAL)
        return int(pyspiel.PlayerId.SIMULTANEOUS)

    def _legal_actions(self, player: int) -> list[int]:
        assert player >= 0
        return list(range(len(self._legal_programs(self._color_of(player)))))

    def _apply_actions(self, actions: list[int]) -> None:
        assert self._outcome == "ongoing"
        program_white = self._legal_programs(Color.WHITE)[actions[0]]
        program_black = self._legal_programs(Color.BLACK)[actions[1]]
        result = phi(self._state, program_white, program_black, self._ruleset)
        self._state = result.state
        self._outcome = result.outcome
        self._cached_programs = {}
        if result.outcome in _TERMINAL_RETURNS:
            self._returns = list(_TERMINAL_RETURNS[result.outcome])

    def _action_to_string(self, player: int, action: int) -> str:
        program = self._legal_programs(self._color_of(player))[action]
        return repr(program)

    def is_terminal(self) -> bool:
        return self._outcome != "ongoing"

    def returns(self) -> list[float]:
        return list(self._returns)

    def clone(self) -> SimultChessState:
        """Override pyspiel's default clone (Python `copy.deepcopy`, which
        cannot handle the `mappingproxy`-backed immutable fields of
        `simult_chess.core.types.State` -- an existing Phase 1 design choice,
        out of scope to change here). Our wrapped native `State` is already
        fully immutable (every `_apply_actions` call *replaces* `self._state`
        wholesale via `phi`, never mutates it in place), so a correct clone
        needs no deep copy at all -- just a fresh wrapper sharing it. Needed
        for MCTSBot and any other algorithm that forks the search tree via
        `State.clone()`."""
        cloned = SimultChessState(self.get_game(), self._state)
        cloned._outcome = self._outcome
        cloned._returns = list(self._returns)
        cloned._cached_programs = dict(self._cached_programs)
        return cloned

    def __deepcopy__(self, memo: dict[int, object]) -> SimultChessState:
        """Some OpenSpiel wrappers (e.g. `load_game_as_turn_based`'s
        `TurnBasedSimultaneousState`) reach for plain `copy.deepcopy` on a
        nested Python state rather than calling its `.clone()`; intercepting
        here gives the same safe, non-deep-copying behavior either way."""
        return self.clone()

    @property
    def state(self) -> State:
        """The wrapped native `simult_chess.core.types.State` (not part of
        the pyspiel API; used by the conformance test and by callers that
        want the underlying board)."""
        return self._state

    @property
    def ruleset(self) -> RuleSet:
        """The `RuleSet` in effect (not part of the pyspiel API; used by
        `SimultChessObserver` to fill the horizon scalar)."""
        return self._ruleset

    def __str__(self) -> str:
        occupant = geometry.occupant_lookup(self._state.board)
        rows = []
        for rank in range(7, -1, -1):
            row = []
            for file in range(8):
                token = occupant(Square(file, rank))
                if token is None:
                    row.append(".")
                else:
                    is_white = token.color is Color.WHITE
                    row.append(token.typ.upper() if is_white else token.typ)
            rows.append(" ".join(row))
        return "\n".join(rows)


class SimultChessObserver:
    """Observer, conforming to the PyObserver interface (see
    `open_spiel.python.observation`). Builds the tensor encoding documented
    in this module's docstring."""

    _NUM_PLANES = len(_PIECE_TYPES) * _NUM_PLAYERS + 1 + 4
    _NUM_SCALARS = 7

    def __init__(self, params: object) -> None:
        if params:
            raise ValueError(f"Observation parameters not supported; passed {params}")
        plane_size = self._NUM_PLANES * 8 * 8
        self.tensor = np.zeros(plane_size + self._NUM_SCALARS, np.float32)
        self.dict = {
            "planes": np.reshape(self.tensor[:plane_size], (self._NUM_PLANES, 8, 8)),
            "scalars": self.tensor[plane_size:],
        }

    def set_from(self, state: SimultChessState, player: int) -> None:
        del player  # perfect information: the tensor is player-independent
        planes = self.dict["planes"]
        scalars = self.dict["scalars"]
        planes.fill(0)
        native = state.state

        plane_of: dict[tuple[Color, PieceType], int] = {}
        index = 0
        for color in (Color.WHITE, Color.BLACK):
            for piece_type in _PIECE_TYPES:
                plane_of[(color, piece_type)] = index
                index += 1
        cooldown_plane = index
        index += 1
        white_defender_plane, white_protege_plane = index, index + 1
        black_defender_plane, black_protege_plane = index + 2, index + 3

        for token, square in native.board.items():
            plane = plane_of[(token.color, token.typ)]
            planes[plane, square.rank, square.file] = 1.0
            if token in native.cooldown:
                planes[cooldown_plane, square.rank, square.file] = 1.0

        for reservation in (*native.reservations_white, *native.reservations_black):
            defender_square = native.board.get(reservation.defender)
            protege_square = native.board.get(reservation.protege)
            is_white = reservation.defender.color is Color.WHITE
            defender_plane = white_defender_plane if is_white else black_defender_plane
            protege_plane = white_protege_plane if is_white else black_protege_plane
            if defender_square is not None:
                planes[defender_plane, defender_square.rank, defender_square.file] = 1.0
            if protege_square is not None:
                planes[protege_plane, protege_square.rank, protege_square.file] = 1.0

        rights = native.bookkeeping.castling_rights
        scalars[0] = float(rights.white_kingside)
        scalars[1] = float(rights.white_queenside)
        scalars[2] = float(rights.black_kingside)
        scalars[3] = float(rights.black_queenside)
        scalars[4] = float(native.bookkeeping.no_progress_counter)
        scalars[5] = float(native.bookkeeping.phase_index % 2)
        scalars[6] = float(state.ruleset.horizon)

    def string_from(self, state: SimultChessState, player: int) -> str:
        del player
        return str(state)


# Register the game with the OpenSpiel library
pyspiel.register_game(_GAME_TYPE, SimultChessGame)
