# DEVELOPMENT — Roadmap v2 (Phases 14–17)

*Third roadmap document. Agent-facing, same status as `DEVELOPMENT.md` (v1.0)
and `DEVELOPMENT_addendum_v1.1.md` (v1.1): gitignored, not project
documentation, but **ground truth for phase planning**.*

*Ground truth for rules remains `docs/simultaneous_chess_spec_v1.md` +
`docs/INVARIANTS.md`. Ground truth for the learning track is the separate
`docs/LEARNING_ROADMAP_v2.md` (Phases 18a–18d); nothing in this document
touches the learning system.*

**Scope.** Everything needed to take the engine from "correct and headless"
to "two people can actually play it, and a web developer can build on it."
Phases 14–17. Same discipline as v1.0/v1.1: Goal → Deliverables →
Definition of Done → ⛔ COMMIT GATE → STOP. All commits by the maintainer.

---

## 0. Why these phases, in this order

The engine is correct (Phase 6 sweep, zero violations over 2.26×10⁵ states;
Phase 11b campaign, zero violations of any severity over 4.5×10⁴ games) and
online-playable in principle (Phase 8). It is not yet *usable*, for four
reasons that are independent of each other:

1. **The defaults are not frozen.** Gate 11b's freeze decision is still
   open. Every game record produced before the freeze is invalidated by it.
2. **Three concrete defects make an online human game fail** (§B1–B3 below).
3. **There is no clock**, and the game's simultaneity makes a clock a
   genuine design object, not a UI widget.
4. **There is no game record format**, so nothing played can be replayed,
   cited, or turned into a regression fixture.

Freeze first (Phase 14) because everything downstream binds to the frozen
`RuleSet`; then make the session robust (Phase 15); then, optionally, a
window (Phase 16); then collaboration and publication hygiene (Phase 17).

---

## A. Defects and gaps found by audit (2026-07-23)

Recorded here so they are fixed deliberately rather than rediscovered.

| # | Location | Finding | Severity |
|---|---|---|---|
| **B1** | `net/session.py:run_online_match` | `message_timeout: float = 30.0`, never overridden by `net/cli.py`. Each peer decides locally, commits, then `recv(timeout=30)`. If the opponent thinks longer than 30 s **after** you have committed, your side raises `ProtocolError` and the game dies. A human online game currently has an undocumented 30-second thinking limit implemented as a crash. | **blocker** |
| **B2** | `net/cli.py:_human_program_source` | `prompt_program` calls blocking `input()` inside the asyncio event loop. No keepalive, countdown, or concurrent I/O is possible until entry moves to `asyncio.to_thread` (or an equivalent). Prerequisite for any clock. | **blocker** |
| **B3** | `net/session.py` | No handshake. Colours are asserted independently by each side's `--color` flag with no cross-check; protocol version, spec version, `RuleSet` fingerprint, `max_phases`, and initial-state hash are all assumed to agree. Two `--color white` peers produce a `ValueError` out of Φ's legality assertion instead of "colour assignment conflict". | **blocker** |
| B4 | `net/session.py` | An illegal *remote* program is correctly caught (Φ asserts `L(s,π)`, `core/phi.py:125–128`) but surfaces as an uncaught `ValueError`. Should be a `ProtocolError` with an explicit adjudication (abort, or forfeit against the sender). | high |
| B5 | `net/transport.py` | A clean disconnect is detected (`readline` → `b""`); a silent network partition is not. No keepalive/heartbeat. With B1 fixed by simply raising the timeout, a partition hangs forever. | high |
| B6 | `net/`, `ui/` | No resign, draw offer, rematch, or abort. `SessionOutcome`/`OnlineOutcome` have no vocabulary for a non-Φ termination. | high |
| B7 | repo-wide | No game record format. `ui/notation.py` formats individual programs; nothing serialises a whole game with its `RuleSet` fingerprint and initial state. | high |
| B8 | `ui/notation.py` | Short move form has no file-only disambiguation (`exd5` → must write `e4d5`). Documented Phase-7 scope cut. Cosmetic for a TUI, irrelevant once entry is click-based. | low |
| B9 | `core/legality.py:check_l6_geometric_legality` | Rebuilds the occupant lookup from `state.board` on every call; the dominant remaining cost per Phase-13b cProfile. Deliberately left alone because the fix changes `check_legal_program`/`is_legal_program` signatures used repo-wide. Now also the throughput lever for the learning track. | see `LEARNING_ROADMAP_v2.md` §1 |
| B10 | repo-wide | No hosted CI. Ruling A11 deferred a GitHub Actions gate "until a second contributor starts pushing". That contributor is now arriving. | medium |

---

## B. Rulings requested from the maintainer

These are decisions, not tasks. The agent must not pick a side.

| # | Question | Options | Default if unruled |
|---|---|---|---|
| **C1** | `pawn_same_square_fizzle_scope` freeze value | `both_pawns_only` (current default, draw rate 0.382) vs `any_same_square` (0.521, Δ = +0.140, the largest effect any A/B arm produced) | keep current default; record the alternative as a named variant |
| **C2** | The other four A/B arms (`cancellation_enabled`, `intermezzo_reading`, `recapture_cooldown`, `H`) | freeze at baseline vs adopt an arm | freeze at baseline |
| **C3** | Whether the freeze is announced as **provisional** (A5, revisable post-learning) or **definitive for v1.1** | provisional / definitive | provisional, with the record format binding the `RuleSet` hash so a later change is detectable rather than silent |
| **C4** | Time-control semantics (§15b) — the race-bonus rule | winner-take-all `b` / capped-difference `min(b, \|t_W − t_B\|)` / Fischer increment to both / none | capped-difference (see the analysis; winner-take-all is degenerate at the proposed calibration) |
| **C5** | Flag-fall without an arbiter | self-adjudicating clock ledger (both peers compute it) vs advisory-only clock | self-adjudicating; it is derivable identically on both sides |
| **C6** | Phase 16 (local window) — build or skip | Tk window / skip and go straight to the web client | build only if the maintainer wants a demo artifact; it is throwaway relative to the web app |

### Rulings record (binding maintainer decisions)

| # | Decided | Ruling |
|---|---|---|
| **C1** | 2026-07-24 | **`both_pawns`** (freeze at current default). `any_same_square` kept as the named variant `any_same_square_fizzle`. |
| **C2** | 2026-07-24 | **Freeze all four arms at baseline.** Each rejected value kept as a named variant (`rules/variants.py`). |
| **C3** | 2026-07-24 | **Provisional** (A5). The `.scn` record binds the `RuleSet` fingerprint, so a later change is detectable, not silent. |
| **B4** | 2026-07-24 | Illegal remote program **forfeits its sender** (both peers run the same pure `L` check, so they agree without an arbiter). |
| **C4** | 2026-07-25 | **Capped-difference** race bonus, `β = min(b, |t_W − t_B|)`. Implemented as a registry so the rule is a parameter, not a rewrite. The maintainer notes this is a *starting* calibration to be revisited during playtest: (a) the "reflex-race" degeneracy is a limiting-case argument, not a claim that intermediate humans will actually rush a 3-minute game — good time use beats a fast opponent; (b) `b = 2 s` may simply be too large; (c) a candidate refinement to register alongside capped-difference is a **dead-zone**: award no bonus at all when the two commit times are within some threshold (e.g. 5 s) of each other, so the bonus only rewards a *decisive* speed gap. Register it as a selectable bonus rule; do not make it the default yet. |
| **C5** | 2026-07-25 | **Self-adjudicating** clock ledger (both peers compute it identically). Explicitly a testing-grade solution, not a tournament-grade one; a tamper-proof arbiter is out of scope for now. |
| **C6** | 2026-07-25 | **Build the Tk window** as throwaway, *on the condition* that it is cleanly separated from the core effort (the game's definition/testing and the learning system). The maintainer is not the app developer and the window must never entangle the engine. The two engine-side pieces (16.1 partial legality, 16.2 affordance/threat API) are not throwaway and are built regardless. |

---

## Phase 14 — Provisional parameter freeze v1.1 (closes Gate 11b)

**Goal.** Turn `reports/campaign_v1.md` into a frozen `RuleSet` so that
everything downstream — game records, the app, the paper — binds to a
stated, fingerprinted set of defaults.

**Why now and not after the learning re-run.** A5 made the freeze
provisional pending a post-Phase-13 re-run under learned agents. Phase 13b
did not produce agents strong enough for that re-run to mean anything, and
waiting blocks four downstream phases. The correct move is to freeze on the
evidence available and be explicit about its epistemic status, which the
campaign report already states: every balance statistic is a functional of
the state distribution induced by *those* agents, not of equilibrium play.
Freezing does not assert equilibrium balance; it asserts a versioned
default.

**Deliverables.**
- Maintainer ruling on C1–C3; `rules/ruleset.py` defaults updated
  accordingly (possibly a no-op — a no-op freeze is still a freeze).
- `RuleSet.fingerprint()` — a stable hash over all rule-bearing fields
  (canonical field ordering, explicitly excluding any non-rule field such
  as solver tolerances). Unit-tested for stability across process
  restarts and field reordering.
- `reports/campaign_v1.md` committed, with a freeze block naming the frozen
  values, the fingerprint, and the provisional/definitive status.
- Spec + `INVARIANTS.md` §8 rows updated: each parameter's status moves
  from `[OPEN]`/`[C]` to `[FROZEN v1.1, fingerprint …]`.
- A named-variant registry entry for every rejected arm value, so the
  alternatives remain playable without forking (`rules/` registry pattern,
  as with `intermezzo_reading`).

**DoD.** ruff/mypy/pytest green; `RuleSet().fingerprint()` reproduces
byte-identically across two fresh interpreters; the campaign report's freeze
block and `rules/ruleset.py` agree; every A/B arm value is reachable as a
named variant.

> #### ⛔ COMMIT GATE 14
> **Suggested message:** `feat(rules): freeze provisional v1.1 defaults with RuleSet fingerprint [Gate 11b]`
> **STOP.**

---

## Phase 15 — Session hardening, time control, transport, records

Four sub-phases, each with its own gate. 15a is a prerequisite for 15b;
15c and 15d are independent of each other.

### 15a — Handshake, robustness, match services

**Goal.** Make an online game between two humans survive contact with
reality.

**Deliverables.**
- **Handshake message** exchanged once before the phase loop, both
  directions: `protocol_version`, `spec_version`, `ruleset_fingerprint`
  (Phase 14), `initial_state_hash`, `max_phases`, proposed colour, and a
  random tiebreak nonce. Colour is **negotiated**, not asserted: each peer
  proposes, and a mismatch is resolved by the nonce comparison or rejected
  with a named error. Any field mismatch aborts with a specific message
  (fixes **B3**).
- **Timeout separation** (fixes **B1**): distinguish *transport* liveness
  from *decision* time. Transport timeout applies only to messages the peer
  should already have sent; decision time is governed by 15b's clock (or,
  with no clock, unbounded). Expose both on the CLI.
- **Non-blocking entry** (fixes **B2**): program entry runs off the event
  loop via `asyncio.to_thread`, so a heartbeat and countdown can run
  concurrently.
- **Keepalive** (fixes **B5**): periodic `ping`/`pong` while waiting for a
  peer's commit, with a liveness deadline independent of decision time.
- **Illegal-remote-program adjudication** (fixes **B4**): catch Φ's
  `ValueError`, re-raise as `ProtocolError` carrying the failing `L`-clause,
  and terminate with a `termination_reason` of `illegal_program`.
- **Match services** (fixes **B6**): `resign`, `offer_draw`,
  `accept_draw`, `decline_draw`, `abort`, `rematch` message types.
- **Outcome refactor.** Keep the outcome codomain exactly
  `{white_wins, black_wins, draw}` (payoff $u \in \{-1,0,+1\}$ unchanged,
  spec §10) and add a separate `termination_reason ∈ {regicide, repetition,
  no_progress, resignation, timeout, illegal_program, abort,
  phase_limit}`. **Rationale:** a resignation or a flag-fall is not a
  property of Φ; it must not enter `State`, `RuleSet`, or the payoff
  functional. Inv M1 (no wall-clock in the operator) stays literally true.

**DoD.** ruff/mypy/pytest green; a two-OS-process game survives a 10-minute
think on one side; mismatched colours, mismatched `RuleSet` fingerprints,
and mismatched spec versions each abort with their own named error before
phase 0; a `kill -STOP` on one peer is detected by keepalive within the
liveness deadline; resignation and draw-offer round trips are tested.

> #### ⛔ COMMIT GATE 15a
> **Suggested message:** `feat(net): session handshake, keepalive, match services, termination reasons`
> **STOP.**

### 15b — Time control: concurrent banks with a race bonus

**Goal.** Implement the maintainer's proposed control: a per-player bank,
plus a bonus awarded to whichever player commits first in a phase.

#### The formal object

A `TimeControl` lives in a session-layer `MatchConfig`. It is **not** part of
`RuleSet` and **not** part of `State`: Φ remains a pure function of
$(s, \pi_\mathrm W, \pi_\mathrm B)$ with no wall-clock (inv M1), and the
clock ledger is match metadata carried alongside the state, hashed
separately.

Per phase $k$, each player $\omega$ has a remaining bank $B_\omega^k$ and
chooses a *thinking time* $t_\omega^k \ge 0$ and a program. Clocks run
**concurrently** — both players are on the clock for the same phase, unlike
alternating chess. Update:

$$
B_\omega^{k+1} \;=\; B_\omega^{k} \;-\; t_\omega^{k} \;+\; \iota \;+\; \beta_\omega^{k},
$$

with $\iota$ a Fischer increment paid to both (optionally 0) and
$\beta_\omega^k$ the race bonus. A player with $B_\omega^k - t_\omega^k \le 0$
before committing loses on time.

**This design solves the null-program problem.** L1 requires
$1 \le |\pi_\omega| \le N$ and L2 requires at least one Move/Castle whenever
a legal displacement exists, so there is no "pass" available as a timeout
fallback. Forfeit-on-time needs no default program at all — the game simply
ends. That is a strictly better answer than a canonical default program, and
it is the reason to prefer this control over a per-phase byoyomi.

#### Why winner-take-all is degenerate at 3|2, and the fix

Let $V_\omega(t)$ be the expected stage payoff from thinking $t$ (increasing,
concave), and $\lambda$ the shadow price of one banked second — the marginal
continuation value of the clock. Ignoring the bank constraint, the phase
objective under a winner-take-all bonus $b$ is

$$
\max_{t \ge 0} \;\; V(t) \;-\; \lambda t \;+\; \lambda b \,\Pr\!\left(t < t_{-\omega}\right).
$$

Without the bonus the first-order condition is $V'(t^\ast) = \lambda$: think
until the marginal decision value equals the price of time. With the bonus,
the game is a **preemption game**. In a symmetric atomless equilibrium with
c.d.f. $F$ on the support, indifference requires
$V(t) - \lambda t + \lambda b\,(1 - F(t)) = \text{const}$, hence

$$
f(t) \;=\; \frac{V'(t) - \lambda}{\lambda b},
\qquad \operatorname{supp} F \subseteq [0, t^\ast),
$$

since $f \ge 0$ requires $V'(t) \ge \lambda$. Two consequences:

1. **Equilibrium thinking is strictly shorter than optimal**, on all of the
   support. Some rent dissipation is intended — that is the "flavour" the
   maintainer wants.
2. **Normalisation gives a degeneracy threshold.** $\int_0^{t^\ast} f = 1$
   forces $\lambda b = V(t^\ast) - V(0) - \lambda t^\ast \equiv \Delta$, the
   total value of a phase's thought. If $\lambda b > \Delta$ no atomless
   equilibrium exists and mass piles at $t = 0$: **both players race to
   instant, unconsidered programs with positive probability.** The bonus is
   then worth more than thinking is.

Calibration for the proposed 3|2: a 180 s bank over an expected ~40 phases
(Phase 11b median phase counts) gives an average allocation of ~4.5 s per
phase. A 2 s bonus is ~44% of that. That is deep in the degenerate regime —
the predicted outcome is that both players slam in fast garbage and the game
becomes a reflex contest rather than a decision contest.

**Recommended fix (ruling C4): capped-difference bonus.**

$$
\beta_\omega^{k} \;=\;
\begin{cases}
\min\!\big(b, \; |t_\mathrm W^k - t_\mathrm B^k|\big) & \text{if } t_\omega^k < t_{-\omega}^k,\\[2pt]
0 & \text{otherwise.}
\end{cases}
$$

You are refunded only what you actually saved relative to your opponent,
capped at $b$. This is continuous at ties, removes the discontinuous prize
that drives preemption, and makes racing to $t=0$ self-defeating (if both
race, the difference is ~0 and nobody gains). It preserves exactly the
intended flavour — the faster decider gains — without the degenerate
equilibrium. With this rule, $b = 2$ s on a 180 s bank is perfectly sound.

**Notation.** Standard chess `3|2` already means "3 min + 2 s Fischer
increment to both". To avoid collision, write **`3|ι|b`** — e.g. `3|0|2`
for the maintainer's proposal (3 min bank, no Fischer increment, 2 s race
bonus), `3|2|2` for both. Record it in full in the match header rather
than relying on the shorthand.

#### The arbiter problem, and its solution

With no trusted third party, two peers cannot agree on "who was first" from
message arrival times: one-way delay $d$ biases the comparison by $d$, so a
20 ms peer beats a 150 ms peer on essentially every close race. Solution:

- Each peer measures $t_\omega$ **locally**, from its own phase start (the
  instant it finishes resolving the previous phase) to its own commit.
- The measured $t_\omega$ is **included inside the committed payload**, so
  the salted hash binds it and it cannot be revised after seeing the
  opponent's time.
- On reveal, each peer **cross-checks** the claimed time against its own
  observed arrival: under a bounded-delay assumption
  $d \le d_{\max}$, a claim is admissible iff
  $|t_\text{claimed} - (\tau_\text{arrival} - \tau_\text{phase start})| \le d_{\max} + \varepsilon$.
  Out-of-tolerance claims raise a `ProtocolError` naming the discrepancy.
- $d_{\max}$ is measured during the handshake (a few ping/pong round trips)
  and fixed for the match, with a floor so that a lossy link cannot be
  gamed by inflating the tolerance.

This makes the clock **self-adjudicating** (ruling C5): both peers derive
identical banks from identical revealed data, so a flag-fall is computed
independently and agreed by construction. The existing post-phase state-hash
divergence check is extended with a **clock-ledger hash** covering
$(B_\mathrm W, B_\mathrm B, t_\mathrm W, t_\mathrm B, \beta_\mathrm W, \beta_\mathrm B)$
per phase — the same instrument, applied to the clock.

#### Two properties worth stating to players

- **Wall-clock length.** Because clocks run concurrently, a game lasts
  roughly $\max(\text{banks})$, not their sum. A `3|0|2` game here runs
  ~3–4 minutes, where chess `3|2` runs ~7. Same numbers, half the game.
- **The design invariant survives.** "Simultaneity eliminates first-mover
  advantage" is untouched: Φ is unchanged and there is still no mover order.
  What the bonus adds is a *first-decider* resource advantage, strictly
  outside the operator. This should be said explicitly in the spec so the
  invariant is not later misread as broken.
- **Timing side channel.** Commit *arrival times* are observable, and the
  bonus rewards making them early. A fast commit correlates with a prepared
  or forced reply, so the control trades a small amount of informational
  simultaneity for flavour. If a tournament-clean mode is ever wanted,
  quantise commit transmission to a fixed grid; not in scope here, but flag
  it in the spec's variant list (§13) rather than discovering it later.

**Deliverables.** `net/clock.py` (`TimeControl`, `ClockLedger`, bank
arithmetic, race-bonus rules as a registry so C4 is a parameter not a
rewrite); committed-payload extension carrying the self-reported elapsed
time; handshake-time $d_{\max}$ estimation; clock-ledger hash in the
per-phase divergence check; flag-fall detection and adjudication;
`--time-control 3|0|2` CLI flag; countdown display in the TUI.

**DoD.** ruff/mypy/pytest green; unit tests for bank arithmetic including
both race-bonus rules and exact-tie behaviour; a property test that both
peers' ledgers are byte-identical over a simulated match with injected
asymmetric latency; an out-of-tolerance time claim is rejected with a named
error; a flag-fall is detected independently and identically by both peers;
a two-OS-process `3|0|2` game runs to a genuine loss on time.

> #### ⛔ COMMIT GATE 15b
> **Suggested message:** `feat(net): concurrent-bank time control with capped-difference race bonus`
> **STOP.**

### 15c — Rendezvous relay and protocol document *(deliberately minimal)*

**Goal.** Let two people who cannot port-forward play each other, and give
the web developer a written protocol — with the **least possible code**, all
of it disposable.

**Design constraint, per the maintainer: this is throwaway work and should
look like it.** The relay must contain **zero game logic**. It does not
parse programs, does not know what a phase is, does not run Φ, and never
becomes a game server. It is a byte pipe with a room code. When a real
backend is built it is deleted in one commit, and nothing else changes.

This also means 15c makes **no commitment** on the server-authoritative
question. Both peers keep running Φ; the relay just moves their bytes. The
web developer can later choose thin-client or peer-authoritative
architecture freely, without being boxed in by anything built here.

**Deliverables.**
- `net/relay.py` — target **≈60–80 lines**, stdlib `asyncio` only, no new
  dependency. One `asyncio.start_server`; a dict of room code → first
  waiting peer; on the second connection with the same code, pipe both
  directions with `asyncio.gather` until either closes. Nothing else. No
  persistence, no accounts, no TLS (see DoD note), no reconnection.
- `net/cli.py` gains `--via HOST:PORT --room CODE` alongside the existing
  `host`/`connect`. The existing `Peer` is unchanged — it is already just a
  socket, so the relay is transparent to it.
- `docs/PROTOCOL.md` — **the actual deliverable of this phase.** The message
  schema (`hello`, `commit`, `reveal`, `ack`, `ping`/`pong`, `resign`,
  `offer_draw`, `rematch`, `abort`), field types, the ordering contract, the
  commitment construction, the clock-claim rules, the divergence checks, and
  a worked byte-level transcript of one full phase. Versioned
  (`protocol_version`), with an explicit compatibility policy.

**DoD.** ruff/mypy/pytest green; the relay is under 100 lines and imports
nothing from `simult_chess.core`/`rules`/`referee` (asserted by a test, the
same quarantine pattern as the `oracle`/`solver`/`learn` extras); two peers
on different machines behind different NATs complete a full game through a
relay on a cheap VPS; `docs/PROTOCOL.md`'s worked transcript is generated
from a real captured session, not written by hand.

**Scope notes.** No TLS, no authentication, no abuse controls — the room
code is the only secret and traffic is plaintext on the wire. Acceptable for
playing your brother; state this in the README in one sentence, and do not
publish a relay address anywhere public. If the relay is ever exposed
beyond family use, it needs TLS termination and rate limiting, which is the
point at which it should be replaced rather than extended.

> #### ⛔ COMMIT GATE 15c
> **Suggested message:** `feat(net): stdlib rendezvous relay (disposable) + docs(protocol): wire schema v1`
> **STOP.**

### 15d — Game record format

**Goal.** Make a played game a first-class, replayable, citable object.

**Why it matters more than it looks.** One artifact does four jobs:
replay and review; regression fixtures harvested from real human play (a
qualitatively different distribution from self-play sweeps); worked examples
for the paper; and a citable object for the Zenodo deposit.

**Deliverables.**
- `referee/record.py`: a text format (`.scn`) with a header block —
  `spec_version`, `protocol_version`, `ruleset_fingerprint` (Phase 14),
  full `RuleSet` field dump, initial state, player labels, `TimeControl`,
  seed if any — followed by one line per phase carrying both programs in
  `ui/notation.py` form, the resolved outcome, and the clock ledger entry.
- Writer wired into `ui/session.py`, `net/session.py`, and
  `referee/match.py`; reader with a `--replay FILE` CLI that re-derives
  every state through Φ and **verifies** the recorded resolutions rather
  than trusting them.
- A `--to-fixture` mode emitting a pytest fixture from any recorded phase.

**DoD.** ruff/mypy/pytest green; round-trip property test (write → read →
re-resolve → byte-identical states) over ≥10³ seeded self-play games; a
record whose `ruleset_fingerprint` does not match the current `RuleSet`
refuses to replay with a named error rather than silently producing
different results; replaying a real two-human game reproduces it exactly.

> #### ⛔ COMMIT GATE 15d
> **Suggested message:** `feat(referee): .scn game record format with fingerprint-checked replay`
> **STOP.**

---

## Phase 16 — Local window *(optional; ruling C6)*

**Goal.** A window a non-technical player can use, with no server and no
new dependency.

**Honest scoping.** If the web app is going to be built, most of this is
throwaway. Build it only as a demo artifact or if the maintainer wants to
play without a terminal. The two engine-side deliverables (16.1, 16.2) are
**not** throwaway — the web client needs exactly the same primitives — and
are worth building even if the window is skipped.

**Deliverables.**

*16.1 — `core/legality.py`: `check_partial_program`.* Runs L3–L6 only,
skipping the L1 budget and L2 mandatory-displacement clauses, so a program
under construction can be validated live. Full `L` still runs on submit.
Must be implemented by factoring the existing clause functions, not by
duplicating them.

*16.2 — `ui/affordances.py`.* From `(state, colour, ruleset, partial
program)`: legal destinations per own token, legal reservation pairings
(including the aggressive-dual future-destination case), cancellable
reservations, and a **threat overlay** — which of the player's tokens are
reachable by an opponent capture next phase. The overlay must be labelled
as *declaration-time attack potential, not safety*: under simultaneity an
attack may fizzle, be annihilated, or be pre-empted by a defensive
recapture. Serves the Tk window and the web client identically.

*16.3 — `ui/window.py`.* Tkinter (ships with CPython — no new dependency).
Board, click-to-build program with live legality feedback, submit, hot-seat
concealment, human-vs-agent, countdown display from 15b, and — the highest
value item — a **resolution view rendering `PhiTrace`**: which moves
fizzled and under which clause, which pairs annihilated, which reservations
fired or were invalidated. Without it a new player cannot learn why
anything happened. There is no check in this game; kings are simply
captured, so the trace view plus the threat overlay are what separate
"learnable" from "arbitrary".

**DoD.** ruff/mypy/pytest green (affordances and partial legality unit
tested; the window itself smoke-tested headless); a person who has not read
the spec can complete a game against `greedy` without typing notation; every
`PhiTrace` category has a rendering.

> #### ⛔ COMMIT GATE 16
> **Suggested message:** `feat(ui): partial legality, affordance API, Tk window with trace and threat views`
> **STOP.**

---

## Phase 17 — Collaboration and publication package

**Goal.** Make the repository safe for a second contributor and the work
citable.

**Deliverables.**
- **CI (closes B10, activates A11).** GitHub Actions running
  `scripts/check.sh` on PRs — the same script, not a reimplementation.
  Matrix over the supported Python versions; extras (`solver`, `oracle`,
  `openspiel`, `learn`) tested in separate jobs so the quarantine is
  enforced by CI, not just by a local test.
- `CONTRIBUTING.md`: the gate protocol, the spec-first rule (no
  implementation before the formal ground truth is settled), the extras
  quarantine, the commit convention, and an explicit statement that rule
  changes go through the spec and `INVARIANTS.md` first.
- **Boundary document** for the collaboration: what is the engine (owned by
  the maintainer, Apache-2.0 code / CC BY 4.0 docs) and what is the
  application layer. Put the scope in writing before handover — not because
  anything is expected to go wrong, but because "who owns the rules" is
  much easier to answer in advance than after an app has users. *This is
  organisational hygiene, not legal advice; a lawyer should see anything
  with money attached.*
- **Publication package**: arXiv (cs.GT) preprint built from
  `simultaneous_chess_spec_v1.md` + `INVARIANTS.md`, with the Zenodo DOI
  deposit covering the tagged engine release and the campaign report.
  **Timing argument:** the contribution is the formal specification, the
  proven-correct operator (order-independence M2, χ-symmetry M3,
  well-definedness Lemmas 6.4a/6.4c/13.4), and the executable
  invariant methodology — none of which depends on the learning system
  working. Publishing now also timestamps the design ahead of the
  collaboration. Do not wait for Phase 18.
- Community distribution (chessvariants.com, BoardGameGeek) stays deferred
  until a playable client exists, per the existing decision.

**DoD.** CI green on a test PR and required for merge; `CONTRIBUTING.md`
reviewed; preprint compiles and the Zenodo deposit resolves; the tagged
release's `RuleSet.fingerprint()` is recorded in the deposit metadata.

> #### ⛔ COMMIT GATE 17
> **Suggested message:** `ci: local gate on PRs; docs: contributing, collaboration scope, publication package`
> **STOP.**

---

## Milestone map (extends v1.0 §6 and v1.1)

| Goal | Delivered at |
|---|---|
| Frozen, fingerprinted v1.1 defaults | Gate 14 |
| An online human game that survives a long think | Gate 15a |
| Clock with a first-decider bonus, self-adjudicating | Gate 15b |
| Play through NAT; written wire protocol for the app | Gate 15c |
| Replayable, citable game records | Gate 15d |
| A window a non-technical player can use | Gate 16 |
| Repository safe for a second contributor; work citable | Gate 17 |

## Recommended sequencing

**14 → 15a → 15d → 15b → 15c → 17 → 16.**

Rationale: freeze before anything binds to the defaults; 15a because the
game currently breaks; 15d early so that every subsequent test game becomes
a durable artifact rather than being lost; 15b before 15c because the clock
is the interesting design work and the relay is trivial; 17 before 16
because CI should exist before a second contributor pushes; 16 last and
optional.

The minimum viable handover to the collaborator is **14 + 15a + 15c** —
frozen rules, a game that works, and a written protocol. That is a stronger
pitch than a rough GUI.
