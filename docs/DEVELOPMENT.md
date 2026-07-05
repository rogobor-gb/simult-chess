# Simultaneous Chess — Development Brief for the Coding Agent, v1.0

*Audience: an autonomous coding agent. Ground truth: `simultaneous_chess_spec_v1.md`
(**spec**) and `INVARIANTS.md` (**inv**). This document is the build plan: the order in
which to construct the system, the typed contract of each module, the definition of
done per phase, and the **commit gates** at which you must stop and return control to
the maintainer.*

*Repository directory: `simult-chess/`. Importable package: `simult_chess/` (hyphens are
illegal in module names; the distribution name `simult-chess` normalizes to the import
name `simult_chess`).*

---

## 0. Prime directives (non-negotiable)

1. **The spec is ground truth.** Do not invent, "improve", or silently reinterpret a
   rule. Every rule you implement cites its spec section in the docstring; every
   validation check cites its `inv` ID.
2. **`[OPEN]` / `[C]` items become `RuleSet` fields — never hardcoded constants.** The two
   still-open items (`cancellation_enabled`, `pawn_same_square_fizzle_scope`) and every
   `[K]`-flagged convention in `inv §8` are read from `RuleSet` at runtime. A variant is a
   new `RuleSet` or a swapped stage implementation, never a fork.
3. **$\Phi$ is a pure function.** No wall-clock, no global mutable state, no unseeded
   randomness inside resolution. Same $(s,\pi_\mathrm W,\pi_\mathrm B,\text{RuleSet})$ ⇒
   bit-identical $s'$ and trace (`inv M1`).
4. **You do not commit.** Never run `git commit`, `git push`, `git tag`, or rewrite
   history. At each **commit gate** you: (a) ensure the gate's Definition of Done is green,
   (b) print a concise change summary and a *suggested* conventional-commit message,
   (c) **STOP** and wait for the maintainer to commit and explicitly authorize the next
   phase. Staging (`git add -n` dry-runs, showing `git status`) is fine; the commit is the
   maintainer's.
5. **Style.** PEP 8 enforced by `ruff`; full type annotations checked by `mypy --strict`;
   **NumPy-style docstrings on every public function and class**, stating symbols and
   assumptions. No code merges a gate with failing `ruff`, `mypy`, or `pytest`.
6. **Dependency hygiene.** `simult_chess.core` and `simult_chess.rules` import **only** the
   standard library. Third-party packages (`chess`, `numpy`, `hypothesis`, networking) live
   strictly in the layers named for them (§3, §5). Rationale in §3.4.

---

## 1. Formal objects to encode (symbol table)

These are restated compactly so the contract is inline; consult the spec for proofs.

- **State.** $s=(\beta,\,C,\,R_\mathrm W,\,R_\mathrm B,\,\eta)$, with occupancy
  $\beta:\mathcal P^{\text{live}}\hookrightarrow\mathcal S$ (injective partial map from live
  tokens to squares $\mathcal S=\{0,\dots,7\}^2$), cooldown set $C\subseteq\mathcal
  P^{\text{live}}$, age-ordered reservation lists $R_\omega$, bookkeeping
  $\eta=(\text{castling rights},\ \text{repetition ledger},\ \nu,\ t)$ with no-progress
  counter $\nu$ and phase index $t$ (spec §1.2).
- **Token.** Identity-carrying; $\mathrm{col}:\mathcal P\to\{\mathrm W,\mathrm B\}$ fixed,
  $\mathrm{typ}:\mathcal P\to\{\mathsf p,\mathsf n,\mathsf b,\mathsf r,\mathsf q,\mathsf k\}$
  mutable only by promotion (spec §1.1).
- **Trajectory.** $\tau=(q_0,\dots,q_\ell)$. **Swept set**
  $\sigma(\tau)=\{q_1,\dots,q_\ell\}$ (**origin excluded** — this exclusion is what makes the
  vacated-square theorem `inv R6` hold). **Directed edges**
  $\varepsilon(\tau)=\{(q_j,q_{j+1})\}$. Knights: $\sigma=\{q_1\}$, $\varepsilon=\varnothing$
  (spec §1.3).
- **Collision primitives** on opposing executing moves (spec §6.1):
  - **(V)** vertex conflict: $\sigma(m_1)\cap\sigma(m_2)\ne\varnothing$.
  - **(E)** edge conflict: $\exists (u,v)\in\varepsilon(m_1)$ with $(v,u)\in\varepsilon(m_2)$
    (head-on swap). **(E) is not implied by (V).**
- **Annihilation rank** (spec §6.3): index each player's moves by declaration order
  $1,\dots,k$; for candidate edge $\{W_i,B_j\}$,
  $r(W_i,B_j)=(\max(i,j),\ \min(i,j))$, ordered lexicographically. Greedy match in
  increasing rank.
- **Precedence relation** for the capture cascade (spec §6.4): $c\prec c'$ iff executing
  capture event $c$ fires a reservation whose recapture vacates the square targeted by
  $c'$ (thereby voiding $c'$). Resolve in any topological order; mutual-defense cycles
  (SCCs) resolve under base semantics.
- **Cooldown update** (spec §6.7):
  $C'=\{\text{tokens that displaced this phase}\}\setminus(\text{pawns}\cup\text{kings})$.
- **Color-swap involution** $\chi$ (inv §1): $\mu(c,r)=(c,7-r)$ (files fixed) composed with
  color inversion; relabels tokens, reflects $C$, maps each reservation to the opposite
  player's list, preserves flank of castling rights. Equivariance target
  (`inv M3`): $\chi(\Phi(s,\pi_\mathrm W,\pi_\mathrm B))=\Phi(\chi(s),\chi(\pi_\mathrm
  B),\chi(\pi_\mathrm W))$.

---

## 2. Repository layout

```
simult-chess/
├── pyproject.toml            # build, deps, ruff/mypy/pytest config
├── README.md
├── LICENSE                   # see §3.4 (GPL note if `chess` becomes non-optional)
├── docs/
│   ├── simultaneous_chess_spec_v1.md
│   ├── INVARIANTS.md
│   └── DEVELOPMENT.md        # this file
├── src/simult_chess/
│   ├── __init__.py
│   ├── core/                 # PURE. stdlib only.
│   │   ├── types.py          # Square, Color, Token, Trajectory, State, Action, Program
│   │   ├── geometry.py       # pseudo-legal trajectory generation on β; σ, ε
│   │   ├── legality.py       # L(s, π): L1–L6
│   │   ├── collision.py      # (V), (E), rank r, χ involution
│   │   ├── phi.py            # Φ pipeline assembly (pure)
│   │   └── stages/
│   │       ├── fizzle.py     # Stage F  (spec §6.2)
│   │       ├── annihilate.py # Stage A  (spec §6.3)
│   │       ├── defense.py    # Stage B  (spec §6.4)
│   │       └── closure.py    # Stage C/D (spec §6.5–6.7)
│   ├── rules/                # PURE. stdlib only.
│   │   ├── ruleset.py        # RuleSet dataclass (all [K]/[OPEN] knobs)
│   │   └── registry.py       # stage-implementation strategy registry
│   ├── invariants/
│   │   ├── checks.py         # executable predicates, one per inv ID
│   │   ├── harness.py        # strict/lenient runner; violation aggregation
│   │   └── repro.py          # repro-dump schema (inv §9), serialization
│   ├── referee/
│   │   ├── serialize.py      # full-state + public-position key K(β,C)
│   │   ├── observe.py        # observation channels; commit–reveal interface
│   │   └── match.py          # phase loop; terminal detection; event log
│   ├── agents/
│   │   ├── base.py           # Agent Protocol
│   │   ├── random_legal.py   # uniform over legal programs (fuzzing workhorse)
│   │   └── greedy.py         # material heuristic over single-action programs
│   ├── harness/
│   │   └── selfplay.py       # headless engine-vs-engine sweep, seeded, reproducible
│   ├── ui/                   # deferred to Phase 7
│   └── net/                  # deferred to Phase 8
└── tests/
    ├── unit/                 # deterministic fixtures per rule
    ├── property/             # hypothesis: M1–M4 metamorphic
    └── fixtures/             # hand-built positions (spec worked examples)
```

---

## 3. Cross-cutting contracts (build these interfaces before the stages that use them)

### 3.1 `RuleSet` (rules/ruleset.py)

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

FizzleScope = Literal["both_pawns", "any_same_square"]
AnnihilationReading = Literal["B", "timed"]
IntermezzoReading = Literal["ii", "i"]


@dataclass(frozen=True, slots=True)
class RuleSet:
    """Immutable parameterization of the transition operator.

    Every field whose truth couples to an invariant is flagged ``[K]`` in
    ``INVARIANTS.md §8``. Changing a field for a variant requires editing the
    coupled invariant(s) in lockstep; the checker reads this object, never a
    literal.

    Parameters
    ----------
    n_actions : int
        Actions per decision phase, :math:`N`. v1: ``2``. Couples: ``L1``.
    horizon : int
        No-progress draw horizon, :math:`H`. v1: ``50``. Couples: ``WF7``, ``T4``.
    recapture_cooldown : bool
        Whether a recapturer enters :math:`C'`. v1: ``True``. Couples: ``R13``.
    cancellation_enabled : bool
        Whether ``Cancel`` is admissible. v1: ``True`` (spec §9 [OPEN]).
        Couples: ``R17``, ``L6``.
    pawn_same_square_fizzle_scope : FizzleScope
        Scope of the same-square fizzle. v1: ``"both_pawns"`` (spec §13 [C,confirm]).
        Couples: ``R2``.
    annihilation_reading : AnnihilationReading
        Mid-path collision semantics. v1: ``"B"`` (declaration-priority pairing).
        Couples: ``R4``.
    intermezzo_reading : IntermezzoReading
        Defensive-precedence semantics. v1: ``"ii"`` (unconditional). Couples:
        ``R7``, ``M4`` (flips ``M4`` from *true* to *order-dependent* under ``"i"``).
    """

    n_actions: int = 2
    horizon: int = 50
    recapture_cooldown: bool = True
    cancellation_enabled: bool = True
    pawn_same_square_fizzle_scope: FizzleScope = "both_pawns"
    annihilation_reading: AnnihilationReading = "B"
    intermezzo_reading: IntermezzoReading = "ii"
```

### 3.2 Stage strategy interfaces (rules/registry.py)

Each ordered stage is a `Protocol`; v1 default implementations register under the
`RuleSet` key they satisfy. Variant levers (spec §13) become alternative registrations,
not edits to `phi.py`.

```python
from typing import Protocol

class FizzleResolver(Protocol):
    def __call__(self, declared, ruleset, *, tie_break=None): ...

class AnnihilationMatcher(Protocol):
    def __call__(self, executing, ruleset, *, tie_break=None): ...

class DefenseResolver(Protocol):
    def __call__(self, survivors, reservations, ruleset, *, tie_break=None): ...
```

**The `tie_break` hook is mandatory and load-bearing.** It injects the processing order
of *provably-commuting* choices (equal-rank annihilation edges; topological order of
$\prec$; fizzle back-induction order). Default is the canonical deterministic order;
`inv M2` drives it through many admissible permutations and asserts identical output. If
you cannot expose an order hook without changing the result, that is itself a bug in the
stage (it means the order is not actually free).

### 3.3 Serialization & repro (referee/serialize.py, invariants/repro.py)

- **Public-position key** $K(s)=\text{hash}(\beta,\,C)$ **excludes reservations** — this
  is the repetition/draw coordinate for `inv T3` and matches spec §10.
- **Full serialization** includes $R_\mathrm W,R_\mathrm B,\eta$ and is the payload of the
  repro dump (`inv §9`), sufficient to replay a $\Phi$ call deterministically from
  `rng_seed + ruleset + state_pre + programs`.

### 3.4 The `chess` library boundary (justification)

`chess` (PyPI, current release **1.11.2**, Production/Stable, Python ≥3.8) is admitted
**only** as (a) a test-only cross-validator of hand-rolled geometry, (b) UI board
rendering (Phase 7), and (c) a future single-move suggestion oracle (Phase 9). It is
**never** imported by `core`/`rules`. Two reasons, on the merits rather than by
convention:

1. **Semantic.** `chess` models *alternating* play with check/checkmate and king-safety
   legality; this game abolishes all three (spec §10). Using it in resolution would import
   the exact semantics the design rejects.
2. **License.** `chess` is GPL-3.0-or-later. Keeping it an **optional extra**
   (`pip install "simult-chess[oracle]"`) and out of the core prevents the copyleft
   obligation from attaching to the engine itself. Declare it under
   `[project.optional-dependencies]`, not `dependencies`.

Hand-rolling ray/step generation on an $8\times8$ board is ~150 lines and removes the
coupling entirely; cross-validate it against `chess` pseudo-legal generation in tests.

### 3.5 The simultaneity contract (referee/observe.py) — why online is "thin" later

Every mode with two deciders (local hot-seat **and** online) obeys one loop:
**collect both programs → reveal → resolve**. Neither side's program is visible to the
other before both commit. Build this **commit–reveal interface** in Phase 6 (referee),
even though the perfect-information base "reveals everything". Networking (Phase 8) is
then only a *transport* for the same handshake — which is precisely why online play lands
as an add-on rather than a redesign (spec §11.5, §14).

---

## 4. Phased build plan with commit gates

Each phase lists **Goal → Deliverables → Definition of Done (DoD)**, then a
**⛔ COMMIT GATE**. Gates are where you stop. Do not cross a gate until the maintainer
confirms the commit and authorizes the next phase.

---

### Phase 0 — Scaffold & toolchain

**Goal.** A lintable, testable, empty package.
**Deliverables.** `pyproject.toml` (`ruff`, `mypy --strict`, `pytest`, `hypothesis` as
dev deps; `chess` as optional extra); `src/` layout; `docs/` populated with spec, inv,
this file; a trivial `test_smoke.py`.
**DoD.** `ruff check`, `mypy`, `pytest` all green on the empty scaffold; `python -c
"import simult_chess"` works.

> #### ⛔ COMMIT GATE 0
> **Suggested message:** `chore: scaffold simult-chess package, tooling, and docs`
> **STOP.** Await maintainer commit + authorization.

---

### Phase 1 — Core state algebra (no $\Phi$)

**Goal.** Immutable, identity-carrying state types + well-formedness.
**Deliverables.** `core/types.py`: frozen dataclasses for `Square`, `Color`, `Token`,
`Trajectory` (computing $\sigma,\varepsilon$), `Reservation`, `State`, `Action`
(`Move`/`Reserve`/`Castle`/`Cancel`), `Program`. `invariants/checks.py`: `WF1–WF7`.
Serialization stubs (`referee/serialize.py`): full-state + public key $K(\beta,C)$.
**DoD.** Unit tests build hand-crafted states and assert `WF1–WF7`; a malformed state
(e.g. two tokens on one square) is caught by `WF1`. `mypy --strict` clean.

> #### ⛔ COMMIT GATE 1
> **Suggested message:** `feat(core): state algebra + WF1–WF7 well-formedness [spec §1]`
> **STOP.**

---

### Phase 2 — Geometry oracle & declaration legality $L(s,\pi)$

**Goal.** Per-piece pseudo-legal trajectories on $\beta$ only, and the legality predicate.
**Deliverables.** `core/geometry.py` (stepwise + slider ray generation; knight jumper
with empty $\varepsilon$; pawn push/diagonal-capture split; **no look-ahead, no
check**). `core/legality.py`: `L1–L6`, including the degenerate no-legal-displacement
exception (`L2`) and the actor/protégé distinction (`L3`, `L4`: cooled *actors*
forbidden, cooled *protégés* allowed — spec §7). `core/collision.py`: `(V)`, `(E)`,
rank $r$, and $\chi$.
**DoD.** Unit tests for `L1–L6` incl. bipartite-conflict consequence of `L5`; a
cross-validation test comparing `geometry.py` against `chess` pseudo-legal generation on
random boards. Property test: $\chi$ is an involution ($\chi^2=\mathrm{id}$).

> #### ⛔ COMMIT GATE 2
> **Suggested message:** `feat(core): geometry oracle, L(s,π) legality, (V)/(E), χ [spec §4,§6.1]`
> **STOP.**

---

### Phase 3 — The transition operator $\Phi$ (four sub-gates)

Build the pipeline stage by stage; each sub-phase is independently testable and gets its
own gate. Assemble `phi.py` last, wiring `assert L(s, π)` preconditions.

**3a — Stage F (fizzle).** `stages/fizzle.py`: `F1` by backward induction on the
functional dependency digraph (assert out-degree ≤ 1; 2-cycles exported to `(E)`); `F2`
pawn convergence, scoped by `RuleSet.pawn_same_square_fizzle_scope`. Tests: `R1`, `R2`,
`R16`.

> #### ⛔ COMMIT GATE 3a
> **Suggested message:** `feat(core): Stage F fizzle resolution (F1/F2) [spec §6.2]`
> **STOP.**

**3b — Stage A (annihilation matching).** `stages/annihilate.py`: bipartite conflict
graph over $M^\ast$; greedy match in increasing rank $r$; `tie_break` hook over
equal-rank edges. Tests: `R3` (edge-swap not subsumed by (V)), `R4`, `R5`, `R6`
(vacated-square theorem), plus the spec §6.3 worked cases (i) and (ii).

> #### ⛔ COMMIT GATE 3b
> **Suggested message:** `feat(core): Stage A annihilation matching by rank [spec §6.3]`
> **STOP.**

**3c — Stage B (defense-precedence cascade).** `stages/defense.py`: pending captures;
precedence DAG $\prec$; topological resolution with `tie_break`; oldest-valid reservation
(`R-multi-in`); defender fires ≤ once/phase (`R-multi-out`); mutual-defense SCC → base
semantics; termination by strict token descent. Reproduce the spec §6.4 worked example
(`d4`/`e3` defense) and assert the outcome is **independent of Black's declaration order**.
Tests: `R7–R12`.

> #### ⛔ COMMIT GATE 3c
> **Suggested message:** `feat(core): Stage B intermezzo defense-precedence cascade [spec §6.4]`
> **STOP.**

**3d — Stage C/D (closure) + $\Phi$ assembly.** `stages/closure.py`: arrivals,
promotion→cooldown, `C'` update (incl. recapturers/promoted, gated by
`recapture_cooldown`), reservation invalidation, bookkeeping, event log, terminal
detection. `core/phi.py`: pure assembly `Φ(s, π_W, π_B, ruleset) -> (s', trace)`.
Tests: `R13–R18`, `T1–T4`.

> #### ⛔ COMMIT GATE 3d
> **Suggested message:** `feat(core): Stage C/D closure and pure Φ assembly [spec §6.5–6.7,§10]`
> **STOP.**

---

### Phase 4 — Invariant harness & repro dumps

**Goal.** Turn `INVARIANTS.md` into a runnable checker wired around $\Phi$.
**Deliverables.** `invariants/harness.py` (strict/lenient modes; runs `PRE` before $\Phi$,
`STATE` on $s,s'$, `TRACE` on the trace); `invariants/repro.py` (the `inv §9` schema;
deterministic replay from a dump).
**DoD.** Every `WF/L/R/T` check is invoked by the harness; a deliberately corrupted $\Phi$
(injected bug) is caught and produces a valid, replayable repro dump. Lenient mode
aggregates multiple violations without halting.

> #### ⛔ COMMIT GATE 4
> **Suggested message:** `feat(invariants): executable harness + repro-dump replay [inv §0,§9]`
> **STOP.**

---

### Phase 5 — Metamorphic property tests (the soundness crown jewels)

**Goal.** Encode the operator's soundness lemmas as `hypothesis` properties.
**Deliverables.** `tests/property/`: a generator of random **legal** $(s,\pi_\mathrm
W,\pi_\mathrm B)$ (respecting $L$); `M1` purity, `M2` internal-order independence (drives
the `tie_break` hooks), `M3` $\chi$-equivariance (with the program swap), `M4`
reservation-order independence (contingent on `intermezzo_reading="ii"`).
**DoD.** All four properties pass on ≥10⁴ generated cases per property in CI. `M3` is the
single most important test; treat any failure as S0 and halt.

> #### ⛔ COMMIT GATE 5
> **Suggested message:** `test(property): metamorphic M1–M4 (determinism, order-indep, χ) [inv §7]`
> **STOP.**

---

### Phase 6 — Referee, agents, headless self-play — *the "solid & tested" milestone*

**Goal.** Play whole games engine-vs-engine and empirically hunt for strange behavior.
**Deliverables.** `referee/match.py` (phase loop over the **commit→reveal→resolve**
contract of §3.5; draw/terminal detection incl. synchronous double-king draw);
`referee/observe.py` (commit–reveal interface, perfect-info = reveal-everything);
`agents/random_legal.py`, `agents/greedy.py`; `harness/selfplay.py` (seeded, reproducible
$K$-game sweep with the invariant harness in **lenient** mode, emitting an aggregated
violation report keyed by `inv` ID and `rng_seed`).
**DoD.** A 10⁴-game random-vs-random sweep completes with **zero S0/S1 violations**; any
S2/S3 findings are written to the report for maintainer review (this is the mechanism you
asked for: a broken rule during play is captured, localized to an `inv` ID, and
replayable). Determinism: re-running a seed reproduces the game exactly.

> #### ⛔ COMMIT GATE 6
> **Suggested message:** `feat(referee,agents,harness): headless self-play + invariant sweep`
> **STOP.** *This is the milestone after which online play is unlocked.*

---

### Phase 7 — Local play: TUI, split-screen hot-seat, human-vs-PC

**Goal.** A human can play, locally, against an agent or a second human.
**Deliverables.** `ui/` TUI rendering **from public state only**; a small textual action
DSL for entering a program of up to $N$ actions (e.g. `Nf3; e3 def Kf4`); hot-seat that
**hides the first mover's committed program** until both are in (reusing the §3.5
commit–reveal interface); human-vs-`Agent` mode.
**DoD.** Two humans can play a full local game; a human can play an agent; illegal program
entry is rejected with the failing `L`-clause named. Optional: ASCII/`chess`-SVG board
snapshots for debugging.

> #### ⛔ COMMIT GATE 7
> **Suggested message:** `feat(ui): TUI, split-screen hot-seat, human-vs-agent [spec §11.5]`
> **STOP.**

---

### Phase 8 — Online direct-connection play *(deferred; specified now)*

**Goal.** Play your brother/friends over a direct connection. Because Phases 6–7 already
route all two-decider play through the commit–reveal contract, this phase adds a
**transport**, not new game logic.

**Deliverables.** `net/`: an `asyncio` TCP transport (one peer hosts on `host:port`, the
other connects — no lobby/matchmaking in v1); a JSON message schema for
`commit`/`reveal`/`resolve`/`event-log`; a **commit–reveal handshake** (salted hash
commitment at declaration, selective reveal of executed/fired actions at resolution) so
simultaneity is fair without a trusted arbiter — this is exactly the structure spec §11.5
prescribes, and it makes the perfect-information base the trivial "reveal everything"
case. Minimal resync/timeout handling.

**DoD.** Two processes on different machines on a LAN (or across the internet with a
forwarded port) play a full game; the post-phase public event logs on both peers are
byte-identical; a dropped/garbled commitment is detected, not silently accepted.

**Scope notes (be explicit with the maintainer).** NAT traversal and a hosted relay are
**out of scope** for v1 direct connection. If "with friends over the internet" later needs
to avoid port-forwarding, a thin WebSocket relay is the natural follow-up — flag it, do
not build it unasked.

> #### ⛔ COMMIT GATE 8
> **Suggested message:** `feat(net): direct-connection online play via commit–reveal [spec §11.5]`
> **STOP.**

---

### Phase 9 — Future roadmap (do not build without explicit instruction)

Sketch only, so the architecture leaves room:
1. **Hidden-information variant** (spec Ch. 11): private reservations; the observation
   function already lives in `referee/observe.py`; solution concept shifts to CFR/MCCFR.
2. **Stockfish-advised agent**: a single-move suggestion oracle for *one* action slot,
   clearly labelled heuristic — never the solver (spec's alternation semantics do not
   apply here).
3. **OpenSpiel integration**: register `simult_chess` as a `pyspiel` game; use
   `turn_based_simultaneous_game` for CFR and SM-MCTS for the base game (spec §8.4). Keep
   the referee/$\Phi$ free of any OpenSpiel dependency; expose a thin adapter only.

---

## 5. Testing strategy (summary)

- **Unit** (`tests/unit/`): one deterministic fixture per rule and per spec worked
  example; the ground-truth cases in spec §6.3–6.4 and §12.1 are mandatory fixtures.
- **Property** (`tests/property/`): `M1–M4` via `hypothesis`; these are the highest-value
  bug finders and gate CI.
- **Fuzz** (`harness/selfplay.py`): seeded self-play as the *empirical* validation
  instrument — the statistically robust check that the operator behaves across the
  reachable state space, not merely on hand-picked positions. Report violations by `inv`
  ID with replayable seeds.

Coverage is a diagnostic, not a target; prefer one decisive metamorphic property over
many shallow line-coverage assertions.

---

## 6. Milestone map to maintainer goals

| Goal | Delivered at |
|---|---|
| Correct, spec-faithful rules engine | Gate 3d |
| Broken-rule detection during play, localized & replayable | Gate 4 + Gate 6 |
| "Solid and somewhat tested" | Gate 6 (10⁴-game clean sweep) |
| Local play (vs PC, split-screen) | Gate 7 |
| **Online play with brother/friends** | **Gate 8** (thin transport on the Gate 6 contract) |
| Variants without forking | Continuous (RuleSet + stage registry, from Gate 3) |

---

## 7. Commit-gate protocol (reference)

At every gate, emit:

```
CHANGE SUMMARY
  - files added/modified: ...
  - DoD status: ruff ✔  mypy ✔  pytest ✔ (N passed)  invariants ✔
SUGGESTED COMMIT
  <conventional-commit message from the gate>
STATUS: HALTED — awaiting maintainer commit and authorization to proceed to Phase <k+1>.
```

Then **stop**. Do not stage the next phase's work, do not commit, do not push. Resume only
on explicit instruction.
