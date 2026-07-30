# LEARNING — Roadmap v3 (Phases 18a′–18f) and study guide

*Supersedes `LEARNING_ROADMAP_v2.md`. Companion to `docs/LEARNING_DESIGN.md`
(Phase 13a) and to the mathematical treatise
`docs/learning_algorithms_vol1.pdf` (hereafter **Vol. I**), which contains the
proofs of every claim marked (†) below. Agent-facing roadmap document, same
status as `DEVELOPMENT_roadmap_v2.md`.*

**What changed from v2, in one paragraph.** Three of v2's open empirical
questions turned out to be analytically decidable, and deciding them changes the
phase ordering. The slot-2 policy update has *identically zero expected
gradient* (†Vol. I Prop. 8.1), so H2 is a derived defect rather than a
hypothesis; the in-tree regret estimator is biased and violates the hypothesis of
the SM-MCTS convergence theorem (†Prop. 8.5), which is a fourth candidate cause
v2 did not list; and the sole ground-truth fixture is symmetric, so it is
*provably incapable* of detecting either defect (†Cor. 8.7). Separately, the
unregularised stage equilibrium is a discontinuous set-valued function of the
payoff matrix (†Prop. 2.6), so v2's Phase 18c policy target is ill-posed as a
regression target and the regularisation machinery of v2's Phase 18d is a
*prerequisite* for it, not a successor arm. v3 therefore opens with a new
correctness phase, merges 18c and 18d, and adds two phases that v2 lacked
entirely: a measurement of strategic depth, and a population-based evaluation to
replace the fixed ladder.

---

## 0. What 13b did and did not establish, restated

**Established.** Unchanged from v2, and worth restating because it is the real
asset: the pipeline runs end to end; self-play produces zero invariant
violations; the evaluation harness exists; and the Matching-Pennies convergence
proof found and fixed three real algorithmic bugs. That is a working instrument.

**Not established, and now partly explained.** v2 listed three hypotheses. The
list is revised as follows.

| # | Claim | Status in v3 |
|---|---|---|
| **H1** | Compute starvation: ~20 ms/sim measured against a ~2.2 ms design estimate, a ~10× shortfall. | **Still open.** Unchanged; Phase 18a addresses it. |
| **H2** | Structural bias in the policy factorisation. | **Reformulated and settled in part.** The *architecture* is not the constraint: an autoregressive head `p(a₁)p(a₂|a₁)` is a universal parameterisation of joint distributions over the slot pair, and the two slots belong to one player, so there is no game-theoretic obstruction to correlating them. What is defective is the *update*: slot 2 is sampled from the network's own masked conditional prior and trained by hard cross-entropy against that sample, so its expected gradient is exactly zero (†Prop. 8.1) and, in the tabular limit, it undergoes neutral drift to fixation on an arbitrary action with fixation probability equal to its initial probability (†Prop. 8.3). Slot 2 is running no learning algorithm at all. |
| **H3** | Wrong backup operator: a search average where the object with standing is the Shapley operator. | **Still open, and sharpened.** The Monte-Carlo target has unit variance and each state is visited approximately once per run, so the value head regresses on single Bernoulli draws (†Prop. 8.11). The remedy is a TD(λ) sweep toward the stage `val`, not a discontinuous switch. |
| **H4** | *(new)* The in-tree regret estimator is biased and not Hannan consistent. | **Established.** `learn/search.py` uses expected-value substitution: the counterfactual value of an unplayed action is a running mean over the rounds at which it happened to be sampled, which estimates its payoff against the *average historical* opponent, not the current one. Against a learning opponent the per-round bias does not vanish, it accumulates linearly while the signal accumulates like √T, and the subsequence over which the mean is taken is itself selected by the regret (†Prop. 8.5). Hypothesis (H2) of the SM-MCTS convergence theorem fails. |
| **H5** | *(new)* The reported policy target is 13.5 % raw network prior. | **Established, quantitatively.** `_regret_matching_strategy` returns a prior-blended strategy and `_simulate` accumulates *that blend* into `strategy_sum`. With `prior_weight = 1` and √-fade, the prior's share of the stored average is `(1/M)·Σ_{t≤M}(√t+1)⁻¹` = 17.7 % at M=64, **13.5 % at the LIGHT default M=128**, 10.2 % at M=256. That average is then the policy training target, so the prior is partly trained toward itself (†Prop. 8.9). |

**The methodological point, revised.** v2 said the diagnosis was unfalsifiable
because there was no rung between a 2×2 matrix and 8×8 chess. That is true and
Phase 18b still builds the rung. But there is a second, sharper problem v2 did
not name: **a symmetric fixture cannot falsify a symmetric defect.** Regret
matching, RM⁺, Hedge and expected-value substitution are all equivariant under
relabelling of actions, so on a fixture whose equilibrium is protected by
symmetry — Matching Pennies, and every χ-symmetric M5 fixture — the equilibrium
is a fixed point of the dynamics *whether or not the estimator is biased*
(†Cor. 8.7). The suite is systematically blind to an entire class of error, and
the cheapest fix in this whole roadmap is a synthetic fixture with a unique,
fully mixed, **non-uniform** equilibrium.

---

## 1. Phase 18a′ — Correctness before comparison *(new, blocking)*

**Goal.** Repair the four established defects and close the fixture blind spot,
so that every later comparison measures what it claims to measure. Nothing in
this phase requires a training run; most of it is unit tests and small diffs.

**Why this comes first.** Every deliverable of v2's Phase 18c changes the
training targets. If the regret estimator that produces the search statistics is
still biased and the policy target is still 13.5 % prior, an 18c-versus-13b
comparison is confounded by three factors at once and a null result is
uninterpretable. This phase costs days; the runs it protects cost weeks.

**Deliverables.**

*18a′.1 — The asymmetric ground-truth fixture.* Extend the synthetic-evaluator
unit test (`tests/unit/test_search_matching_pennies.py`) with a weighted
rock–paper–scissors stage game

```
U = [[ 0, -1,  2],
     [ 1,  0, -1],
     [-2,  1,  0]]      x* = y* = (1/4, 1/2, 1/4),   val(U) = 0
```

which is antisymmetric — so M5's `val = 0` anchor still applies — but whose
equilibrium is non-uniform and whose relabelling symmetry group is trivial. Add
one more with `val ≠ 0` and unequal support sizes. Assert convergence of the
*average* strategy to `x*` in total variation, across ≥ 30 seeds, at the same
significance discipline as the Matching-Pennies test. **This test should fail on
the current estimator.** If it passes, that is itself a finding and must be
reported, because Prop. 8.5 predicts otherwise.

*18a′.2 — Unbiased in-tree regret.* Replace expected-value substitution with one
of, in order of preference:
  - **(E3) explicit row evaluation.** At each expanded node maintain a restricted
    pool `A'` of k ≈ 16 *programs* per colour; on each simulation evaluate
    `u(a, b_t)` for every `a ∈ A'` against the sampled opponent program `b_t`, at
    a cost of k Φ calls ≈ 2.2 ms at the measured 0.139 ms/Φ. Exact instantaneous
    regret on `A'`; hypothesis (H2) of the convergence theorem holds.
  - **(E2) outcome sampling.** `ĝ_t(a) = 1{a_t = a}·v_t / σ_t(a)` with an ε-floor.
    Unbiased at zero extra Φ cost, at the price of variance `O(1/min σ)`.

(E3) is preferred because the same evaluations are reused by 18c: one set of Φ
calls serves the regret update, the value target and the policy target.

*18a′.3 — Separate exploration from the reported policy.* Sample from
`σ̃_t = (1−ε_t)·RM⁺(R_t) + ε_t·prior`, but accumulate **`RM⁺(R_t)`**, not the
blend, into `strategy_sum`. Add an explicit ε-floor so that hypothesis (H3) of
the convergence theorem (every action visited infinitely often) is a guarantee
rather than an empirical observation — the √-fade raised the tail but does not
provide a floor. Adopt **linear averaging** (`σ̄ ∝ Σ t·σ_t`), standard in CFR⁺,
which additionally halves the residual prior share.

*18a′.4 — Optimistic variants.* Add a one-step prediction `m_t = g_{t−1}` to the
regret update (predictive RM⁺) and expose Hedge and optimistic Hedge as
configurable alternatives. Against another optimistic learner the per-node
NashConv rate improves from `O(1/√M)` to `O(log M / M)` — at M = 128 that is
roughly 0.09 → 0.04. This is the cheapest strength improvement available
anywhere in this roadmap and v2 did not contain it. Note also that regret
matching's `√|A|` regret constant is worse than Hedge's `√log|A|`, and at the
measured pool size |A| ≈ 52 that is a factor of ~3.6 in regret, i.e. ~13× in
simulations required — so the v1 choice of RM over Exp3 "because it has no
temperature to tune" should be revisited rather than inherited.

*18a′.5 — χ-antisymmetric network.* Impose `V(χs) = −V(s)` architecturally
rather than learning it: encode both `s` and `χs`, share all weights, and output
`V̂(s) = ½(g(e(s)) − g(e(χs)))`, with the analogous symmetrisation of the two
policy heads. This makes M5's `val = 0` anchor and the colour-symmetry
spot-check *identities* rather than tests, and it halves the effective sample
complexity of the value head. Cost: one extra forward pass, batched.

*18a′.6 — Retrospective diagnostics on the archived 13b checkpoints.* Zero new
compute. (a) Plot mean slot-2 conditional entropy against checkpoint index, per
seed. Prop. 8.3 predicts monotone collapse toward determinism with a
seed-dependent, strategically arbitrary limit; observing it closes H2 without a
single new game. (b) Plot the slot-1 average strategy's total-variation distance
from the raw network prior; Prop. 8.9 predicts a floor near 0.135.

**DoD.** ruff/mypy/pytest green including the slow suite; 18a′.1 passes on the
new estimator and its pre-change failure is recorded in the report; the
Matching-Pennies test still passes (no regression); 18a′.6 reported with figures;
a written statement of which of H2, H4, H5 the retrospective diagnostics confirm.

> #### ⛔ COMMIT GATE 18a′
> **Suggested message:** `fix(learn): unbiased in-tree regret, unblended strategy average, optimistic RM+, χ-antisymmetric net; test(learn): asymmetric equilibrium fixtures`
> **STOP.**

---

## 2. Phase 18a — Triage and throughput *(retained, re-scoped)*

**Goal.** Remove the throughput ceiling and measure the *magnitude* of the
slot-2 deficit. The existence of that deficit is no longer in question, so
18a.1's estimand changes.

**Deliverables.**

*18a.1 — Magnitude of the coordination deficit (re-scoped).* Unchanged
mechanically: relative Elo of the learned agent against the ladder at N = 1
versus N = 2. Re-scoped in interpretation: this is no longer a test of whether
H2 holds but an estimate of how much strength the empty slot-2 update costs.
Pre-register the estimand, test statistic and sample size, and **use a paired
design with common random numbers** (same seeds, same opponent draws, same
initial-state sequence across arms); the paired variance is typically an order
of magnitude below the unpaired variance, so the same conclusion is reachable at
a fraction of the game count.

*18a.2 — Coordination probe (re-scoped, and deferred behind 18b).* v2 proposes
measuring the learned joint policy's mass on coordinated programs against the
product of its marginals. Do this, but **only after 18b computes the
factorisation gap φ(s)** — the exact value a player loses by being restricted to
product-form mixtures over its own two slots (†Def. 4.6). If φ ≡ 0 on the
fixtures, the probe is uninformative by construction and should not be run.

*18a.3 — Throughput.* Unchanged from v2 and still correct: fix **B9**
(`check_l6_geometric_legality` rebuilding the occupant lookup per call) as a
mechanical, behaviour-preserving refactor gated by the full slow suite, then
implement leaf-batched evaluation. Note that 18a′.2's (E3) adds k Φ calls per
simulation, so the throughput target should be restated against the *new*
per-simulation cost model, not the 13a one.

*18a.4 — Profiling report.* Re-measure ms/simulation, positions/s and games/hour,
and restate the LIGHT budget against measured throughput.

**DoD.** ruff/mypy/pytest green including the slow suite; 18a.3 verified
behaviour-preserving against the 100-game conformance and 500-game minimatch
sweeps; ≥ 3× measured improvement in simulations/second or a written explanation
of why not; 18a.1 reported with pre-registered, paired statistics; a written
verdict on H1.

> #### ⛔ COMMIT GATE 18a
> **Suggested message:** `perf(core,learn): batched leaf evaluation, occupant-lookup hoisting; report(learn): 13b triage`
> **STOP.**

---

## 3. Phase 18b — Ground truth, and the measurement of strategic depth

**Goal.** Build problems whose equilibrium is known exactly, and — new in v3 —
**use them to measure how strategically deep this game actually is.**

**Why this is still the load-bearing phase.** Unchanged from v2, and reinforced:
between "converges on a 2×2 matrix" and "plays 8×8 simultaneous chess well"
there is no intermediate rung, and §0 now adds that the existing rung is
symmetry-blind.

**Deliverables.**

*18b.1 — Board-size generalisation (feasibility first).* Unchanged from v2,
including the instruction to stop and report if `board_size` is not a contained
change.

*18b.2 — Exactly-solvable endgame suite.* Unchanged from v2 in construction:
K+R vs K, K+P vs K, K+N+N vs K, and two-to-four-phase constructed positions,
solved by value iteration with `solver/lp.py` on the *exhaustive* legal-program
sets, memoised over the reachable state graph. **One change to the fixture
format:** the equilibrium *value* is unique and safe to store, but the
equilibrium *strategies* are not — the optimal face generically has positive
dimension in a game with payoff ties, and which vertex HiGHS returns depends on
pivot order (†Prop. 2.6). Store either the optimal face as a system of linear
equalities and inequalities (from complementary slackness) or, preferably, the
**maximum-entropy equilibrium**, which is the canonical τ ↓ 0 limit of the
regularised solve and is stable under solver upgrades.

*18b.3 — Component tests against truth.* Unchanged from v2: value-head error
against the exact value; search-derived NashConv against the exact equilibrium;
policy-head representational capacity by direct supervised fitting.
**Pre-registered prediction:** the supervised-fitting test will *succeed*,
because the autoregressive head is a universal parameterisation and the defect is
in the update rule, not the architecture. If it fails, that is a genuine
capacity finding and changes the diagnosis; recording the prediction in advance
is what makes the test informative either way.

*18b.4 — The depth profile (new).* On every solved fixture compute and report
the four functionals of †Vol. I §2.7 and §4.5:

| Symbol | Definition | Reads as |
|---|---|---|
| `γ(s)` | `min_j max_i U_ij − max_i min_j U_ij` | how much simultaneity matters — the value of the move order, identically 0 in any alternating game |
| `μ(s)` | `val(U) − max_i min_j U_ij` | the value of being allowed to randomise |
| `k*(s)`, `H*(s)` | equilibrium support size; entropy of the max-entropy equilibrium | how rich the required mixture is |
| `φ(s)` | `val(U) − max over product-form x of min_y xᵀUy` | how much intra-program slot coordination is worth |
| `δ(s)` | `val(U) − val(U with White's Reserve programs deleted)` | the shadow price of the commitment mechanic |
| `ρ_res`, `ρ_can` | equilibrium mass on programs containing `Reserve` / `Cancel` | whether the mechanics are used at equilibrium at all |

Each costs at most one extra LP once the exact solver exists. `γ` is the game's
strategic-depth signature and it is, as far as this project has ever measured,
entirely unknown. `δ` and `ρ_can` are the first quantitative statement about
whether the reservation mechanic — described in the spec as a Schelling-style
deterrence device — carries any strategic weight; recall that the Phase 11b
campaign reports a structural cancellation rate of exactly 0.000 *by
construction*, so nothing whatever is currently known.

**DoD.** ruff/mypy/pytest green; the exact solver reproduces M5's proven
`val = 0` on the existing symmetric fixtures; every solved position replays
through the invariant harness with zero violations at any severity; the three
component tests report against every checkpoint of the 13b run,
retrospectively; the depth profile is reported with distributions, not just
means, and `φ` is reported before 18a.2 is run.

> #### ⛔ COMMIT GATE 18b
> **Suggested message:** `feat(learn): exactly-solved endgame suite, canonical max-entropy equilibrium fixtures, depth-profile functionals`
> **STOP.**

---

## 4. Phase 18c — The unified node solver *(merges v2's 18c and 18d.1)*

**Goal.** Replace the search-average value backup with a regularised stage-game
equilibrium, give slot 2 a genuine target, and make the policy target
well-posed — as **one change**, not three.

**Why one change and not three.** v2 sequences 18c.1 (LP value backup), 18c.2
(slot-2 refinement) and 18c.3 (double oracle) as separate deliverables. They are
all consequences of a single structural move: *materialise a small stage game at
each expanded node over a pool of **programs**, not of slot-1 actions.* Once that
exists, the value target, the policy target, the unbiased regret estimator of
18a′.2 and the slot-2 conditional all fall out of the same Φ calls. Sequencing
them separately invites partial implementation, in which the value target changes
while the regret estimator remains biased and the comparison is confounded.

**The formal object.** Unchanged from v2 and correct. For state s with
restricted supports and stage matrix `U_ij = r + V(Φ(s,π_i,π_j))`, the Shapley
operator is `(𝒯V)(s) = val[U]`, and Shapley (1953) establishes that it
characterises the value; the T4 no-progress horizon supplies a finite effective
horizon, so backward induction applies and no discount is needed (†Prop. 1.6).
**One correction to v2's plan:** do not train the policy head against the LP's
equilibrium mixed strategy. The equilibrium correspondence `U ↦ (x*,y*)` is
set-valued, discontinuous, and admits no continuous selection (†Prop. 2.6) — an
arbitrarily small perturbation of one payoff entry can move the optimal strategy
by the full diameter of the simplex — so it is not a well-posed regression
target, and the bootstrapping loop amplifies the instability. Use the
**entropy-regularised (quantal-response) equilibrium** `x*_τ` instead: it is
unique, interior, Lipschitz in U with constant `O(1/τ)`, and satisfies
`|val_τ(U) − val(U)| ≤ τ·log max(m,n)` — so at k = 16 and τ = 0.02 the value bias
is under 0.06, and annealing τ ↓ 0 recovers the exact equilibrium (†Thm. 2.9).

**Deliverables.**

*18c.1 — The row-sketch node solver.* At each expanded node maintain a pool of
k ≈ 8–16 **programs** per colour, seeded by the network prior and grown by
18c.3. Per simulation: sample the opponent program `b_t` from its current
strategy, evaluate the column `u(a, b_t)` for all a in the pool (k Φ calls,
≈ 2.2 ms), and accumulate the evaluated rows into a running sketch of the stage
matrix. This gives, from one set of Φ calls:
  - the **exact instantaneous regret vector** on the pool (18a′.2's (E3));
  - a stage-matrix estimate whose regularised value `val_τ` is the **value
    target** — at `O(k)` per simulation rather than the `O(k²)` a full stage
    matrix would cost;
  - the **policy target** `x*_τ` over programs, which factorises into a slot-1
    marginal and a slot-2 conditional `σ̄(a₂|a₁)` — this is the H2 fix, and it is
    free.

*18c.2 — TD(λ) sweep on the value target.* Do not switch discontinuously from
the Monte-Carlo return `z` to `val_τ`. Sweep
`target = (1−λ)·val_τ(Û(s)) + λ·z` over λ ∈ {0, 0.25, 0.5, 0.75, 1} on the 18b
fixtures. λ = 1 is the 13b behaviour and λ = 0 is v2's proposal; the intermediate
settings are usually best, and a sweep makes a null result interpretable. Bound
the replay-buffer age explicitly: bootstrapped targets computed under an old
network are stale in a way Monte-Carlo targets are not.

*18c.3 — Support growth and the exploitability bracket.* Top-k-by-prior is a
heuristic restriction and it carries **no certificate**: the doubly-restricted
value is neither an upper nor a lower bound on the true value (†Cor. 3.7).
Double oracle supplies the certificate — solve the restricted game, compute a
best response outside the support, add it, repeat; termination with no improving
best response certifies an equilibrium of the full game, and the best-response
gap *is* an exploitability bound. Two tiers:
  - **Required:** full double oracle at the stage level on the 18b fixtures,
    where best responses are exactly computable. This produces the only rigorous
    exploitability numbers the project will ever have.
  - **Diagnostic:** on a random subsample of self-play nodes, compute the
    one-sided bracket over the *single-action pool* only — `O(|pool|)` Φ calls,
    ≈ 7 ms at |pool| ≈ 52 — and report its width as a running measure of how
    binding the top-k restriction is. Any strictly positive width is a *proof*
    of suboptimality with a witness, and the witness is a concrete program the
    designer can inspect; expect the witnesses to be coordination and reservation
    programs.

**DoD.** ruff/mypy/pytest green; on the 18b suite the value head's error against
exact values decreases monotonically over checkpoints; the search-derived
NashConv against the *exact* equilibrium decreases across ≥ 3 checkpoints;
the factorisation probe shows joint mass strictly exceeding the product of
marginals wherever φ > 0; the double-oracle bracket on the 18b fixtures is
reported as a number, not a trend; the TD(λ) sweep is reported in full and the
selected λ is justified against it; the 13b strength gate (defeats `matrix_1ply`
at one-sided exact binomial p < 0.01) passes — but see Phase 18f on why that gate
is a weak proxy.

> #### ⛔ COMMIT GATE 18c
> **Suggested message:** `feat(learn): row-sketch node solver — regularised Shapley backup, program-level regret matching, TD(λ) value targets, double-oracle bracket`
> **STOP.**

---

## 5. Phase 18d — Regularised dynamics as the default

**Goal.** Make the policy update itself regularised, so that convergence is in
*last iterate* rather than in average, and so that τ becomes a usable design
parameter rather than an internal constant.

**Motivation.** Unregularised no-regret dynamics in zero-sum games do not
converge in last iterate; in continuous time they conserve the Bregman
divergence to any interior equilibrium and are Poincaré recurrent, and in
discrete time with constant step they spiral outward. This is why CFR-family
methods report the *average* strategy. Adding a strictly convex penalty against
a slowly-moving magnet policy makes the regularised problem strongly monotone
with modulus τ, which converts `O(1/√T)` average-iterate guarantees into
**linear last-iterate** rates; iterating the magnet then converges to the true
equilibrium. This is the R-NaD idea and, in cleaner first-order form, Magnetic
Mirror Descent.

**Why this is now a default and not an arm.** v2 framed 18d as an alternative to
18c, to be compared at the end. Three arguments say it belongs inside 18c
instead. (i) The policy target of 18c *must* be regularised to be well-posed at
all (†Prop. 2.6). (ii) Last-iterate convergence removes the need to store and
normalise a strategy sum, and makes the node's output the same object at every
iteration, which simplifies tree reuse between phases. (iii) MMD's inputs are
exactly what the row-sketch already produces — a payoff vector over a restricted
program pool — so the marginal implementation cost is one closed-form update.

**Deliverables.**

*18d.1 — MMD as the in-tree and outer-loop update.* With negative entropy the
update has the closed form

```
x_{t+1}(a) ∝ x_t(a)^{1/(1+ηα)} · μ(a)^{ηα/(1+ηα)} · exp( η·g_t(a) / (1+ηα) )
```

— a geometric interpolation of the current policy, the magnet μ, and the
exponentiated payoff. One line, given `g_t`. Implement the magnet schedule
`μ ← x` on an outer loop and the α (equivalently τ) annealing schedule.

*18d.2 — Comparison report.* `reports/learning_v2.md`: last-iterate NashConv,
entropy trajectory and value calibration for {RM⁺, optimistic RM⁺, Hedge, MMD}
on identical seeds, budgets and fixtures, with pre-registered estimands and a
paired design. Conclude with a recommendation for the HEAVY profile and — if no
arm reaches the strength bar — an explicit statement of what would be needed,
rather than an open-ended scaling promise.

*18d.3 — The τ-indexed agent family (new).* The regularised equilibrium *is* a
logit quantal response equilibrium with precision 1/τ. A τ-sweep therefore
produces a family of agents whose errors are game-theoretically coherent — they
mix proportionally to exponentiated advantage — rather than an engine with
artificially injected blunders. Ship this as a **difficulty ladder** for the
eventual online client, and calibrate τ against human play once game records
exist (Phase 15/17 of the development roadmap). This is the cleanest bridge
between the learning system and the playable product, and it costs nothing
beyond exposing a parameter already in the loop.

*18d.4 — Hidden-information readiness note.* Unchanged from v2, with one
addition: state explicitly that MMD (or R-NaD) is the component that *transfers*
to Chapter 11, whereas a search-centric solution leaning on the Shapley operator
does not, because the operator has no direct analogue over information sets.
Φ, the plane layout, the trunk and the action space are unchanged; the
observation function becomes player-dependent and the search becomes an
information-set search. No implementation.

**DoD.** ruff/mypy/pytest green; all arms reported on identical fixtures with
paired statistics; last-iterate NashConv reported for MMD alongside
average-iterate NashConv for the others; invariant harness clean over all
self-play at every severity; the τ-ladder demonstrated on ≥ 3 settings with a
monotone strength ordering.

> #### ⛔ COMMIT GATE 18d
> **Suggested message:** `feat(learn): magnetic mirror descent, τ-annealed regularised equilibria, QRE difficulty ladder; report(learn): update-rule comparison`
> **STOP.**

---

## 6. Phase 18e — Depth as a design objective *(new)*

**Goal.** Use the learning system to inform the definitive `RuleSet` freeze on a
game-theoretic criterion, rather than on descriptive balance statistics alone.

**Why.** Ruling A5 schedules the definitive parameter freeze after the learning
phases, and the Phase 11b campaign already runs five A/B arms
(`cancellation_enabled`, `intermezzo_reading`, `pawn_same_square_fizzle_scope`,
`recapture_cooldown`, `H`) at 5×10³ games each. What it reports is draw rate,
phase-count distribution, horizon attribution, mechanic usage rates and material
volatility — all descriptive, none of them a statement about how *deep* the
resulting game is. Once 18b's exact solver exists, the depth profile
(γ, μ, k*, φ, δ, ρ) can be computed per arm at the cost of one LP per fixture,
and the open rulings become comparable on a common, game-theoretically
meaningful scale.

**Deliverables.**

*18e.1 — Depth profile per arm.* Re-run the 11b arms and report the depth
profile alongside the descriptive statistics. Report distributions, not means:
a `RuleSet` whose γ is large in a few sharp positions and zero elsewhere is a
different game from one whose γ is uniformly moderate.

*18e.2 — The depth functional and its sensitivity.* Define

```
D(RuleSet) = E_{s∼fixtures}[ w₁·γ(s) + w₂·log k*(s) + w₃·φ(s) + w₄·δ(s) ]
```

subject to balance constraints (draw rate below a stated bound; median phase
count within a stated interval; colour symmetry, which M3 guarantees for the
rules and which therefore diagnoses agent asymmetry only). The weights `w` are
not estimable — they encode what kind of game the designer wants — so report the
profile per arm and the aggregate under several weightings as a sensitivity
analysis. **This is a design instrument, not an optimisation:** the ruling
remains the maintainer's.

*18e.3 — Candidate depth-increasing arms (for the designer's consideration).*
Each is a measurable arm, not a proposal to change the rules:
  - **`n_actions = 3`.** Directly increases φ's ceiling — more slots means more
    intra-program coordination to be worth something — at a combinatorial cost
    the autoregressive head absorbs linearly. Measure φ and the throughput hit.
  - **Cancellation cost.** Spec §9 makes `Cancel` free and slot-less on the
    grounds that L2 already prevents covert passing. Free withdrawal weakens the
    commitment: a reservation that can be abandoned at no cost is a cheaper
    Schelling device. An arm in which `Cancel` consumes a slot would raise δ.
  - **Reservation lifetime.** A reservation that expires after a fixed number of
    phases converts a standing commitment into a timed one and interacts with
    the intermezzo reading; measure δ and ρ_res.
  - **Horizon `H`.** Already an arm; the depth reading is that a short `H` caps
    pursuit endgames and mechanically reduces k* in the late game.

*18e.4 — The smallest exactly-solved variant (new, and the publishable object).*
v2's closing question 3 asks what the smallest game in this family is whose
equilibrium is exactly computable end to end, and observes that such a game "may
also be the most publishable object, since an exactly-solved simultaneous chess
variant is a cleaner contribution than a moderately strong neural agent". v3
promotes that from a question to a deliverable. Identify the variant — reduced
board if 18b.1 succeeds, otherwise a piece-restricted opening position — solve it
exactly, and report its full depth profile. **An exactly-solved simultaneous-move
chess variant with a measured strategic-depth signature is a contribution
independent of how strong the neural agent turns out to be**, and it is the
result most likely to interest a game-theory audience rather than only a
chess-variant audience.

**DoD.** ruff/mypy/pytest green; depth profile reported for every 11b arm with
pre-registered primary estimand and explicit FDR control across the secondary
family; sensitivity analysis over ≥ 3 weightings; `reports/depth_v1.md` written;
the maintainer's freeze ruling recorded with its stated objective.

> #### ⛔ COMMIT GATE 18e
> **Suggested message:** `report(rules): strategic-depth profile across RuleSet arms; feat(solver): exactly-solved micro-variant`
> **STOP.** *This gate is the definitive parameter freeze (ruling A5).*

---

## 7. Phase 18f — Population evaluation *(new)*

**Goal.** Replace the fixed three-agent ladder as the primary strength metric.

**Why.** Relative strength in a zero-sum game is not a total order: the "beats"
relation admits cycles, and a scalar Elo fitted to a tournament matrix is a
projection onto the matrix's *transitive* component, discarding the cyclic
component entirely (the combinatorial Hodge decomposition
`P = grad r + rot`). In a game whose design claim is that it *requires* mixing —
i.e. whose stage games have γ > 0 — the cyclic component is precisely what the
game is about, and reporting only Elo against `{random_legal, greedy,
matrix_1ply}` measures the part of performance the game is least about. Worse,
a learned agent can raise its win rate against all three while becoming *more*
exploitable, by specialising to the cyclic structure of a three-point
population; the 13b DoD gate is therefore a proxy the training procedure can, and
under self-play pressure will, overfit.

**Deliverables.**

*18f.1 — The empirical game.* Maintain the population of all checkpoints plus the
three baselines; compute the full tournament matrix using the existing match-play
code; report (i) the Nash equilibrium of the meta-game, (ii) the mass it places
on each checkpoint — a checkpoint with zero meta-Nash mass is dominated and its
Elo is misleading — and (iii) the ratio `‖rot‖/‖P‖` as a measure of how cyclic
the population is.

*18f.2 — Elo, relabelled.* Continue reporting relative Elo with bootstrap CIs,
but label it explicitly as a summary of the transitive component only.

*18f.3 — Statistical protocol (project-wide).* Three amendments, applying to
every phase from 18a onward:
  - **Paired designs with common random numbers** for all arm comparisons.
  - **Always-valid sequential tests** (confidence sequences / e-values) in place
    of fixed-n exact binomials, so that expensive runs can be monitored
    continuously at controlled type-I error and stopped early for futility. Keep
    α = 0.01 for continuity with the Phase 10 and 13b gates.
  - **Explicit multiplicity control.** One pre-registered primary estimand per
    phase; everything else is exploratory or FDR-controlled. v3 adds many
    reported quantities and this is what keeps them honest.

**DoD.** ruff/mypy/pytest green; meta-game Nash and Hodge ratio reported for the
13b population retrospectively and for every subsequent run; the strength gate
restated in population terms; `reports/evaluation_v2.md` written.

> #### ⛔ COMMIT GATE 18f
> **Suggested message:** `feat(learn): empirical-game evaluation, meta-Nash and Hodge decomposition; test: always-valid sequential gates`
> **STOP.**

---

## 8. Study guide, revised

*Ordered by dependency. Each tier states the object to understand, not just the
paper to read. Changes from v2 are marked* **(new)** *or* **(moved)**.

### Tier 0 — The setting, precisely

- **Zero-sum simultaneous-move Markov game, perfect information.** Shapley
  (1953), *Stochastic Games*, PNAS 39(10). Defines 𝒯 and proves existence of a
  value. Short and readable. **(new)** Note that the *undiscounted* case here
  needs no contraction argument: the no-progress rule gives a lexicographic
  progress potential and hence a finite effective horizon, so uniqueness of the
  fixed point comes from a depth induction, not from Banach (†Vol. I §1.3).
- **Why simultaneity changes everything.** In an alternating game the stage
  "game" is a maximisation; here it is a matrix game with a mixed-strategy
  value. **(new)** The quantitative form: the commitment gap
  `γ(U) = min_j max_i U_ij − max_i min_j U_ij` is the exact error of a pure
  backup, it is identically zero in any alternating game, and it is the game's
  strategic-depth signature. An engine built on a `max` backup is worst exactly
  where the game is most itself.
- **Minimax-Q.** Littman (1994), ICML. Value iteration with an LP backup: the
  direct ancestor of Phase 18c.
- **Exploitability / NashConv.** Understand why this, not win rate, is the metric
  with standing, and why a *learned* best response gives only a lower bound.
  **(new)** Add Balduzzi et al. (2018), *Re-evaluating Evaluation*, NeurIPS, for
  the Hodge decomposition and why Elo discards the cyclic component.

### Tier 1 — Regret minimisation

- **Mirror descent / FTRL first** **(moved up from Tier 3)**. The single scheme
  `x_{t+1} = argmax {⟨x, Σg⟩ − ψ(x)/η}`; with ψ = negative entropy it is
  multiplicative weights. Regret matching, Hedge, CFR and MMD are all choices of
  (regulariser, payoff estimator, anchor). Learning this *first* rather than last
  is the reordering v3 recommends most strongly: it makes every later algorithm a
  special case rather than a new thing.
- **Blackwell approachability** **(new)**. Blackwell (1956), Pacific J. Math.
  Regret matching *is* approachability of the nonpositive orthant, and the
  potential-function proof of its `Δ√(|A|T)` bound is two lines. Worth having
  because the `√|A|` versus Hedge's `√log|A|` gap is a factor of ~3.6 at this
  project's pool sizes.
- **Regret matching.** Hart & Mas-Colell (2000), Econometrica 68(5).
- **CFR and CFR⁺.** Zinkevich et al. (2007); Tammelin (2014). Understand the
  counterfactual decomposition, why the *average* strategy converges, and — new
  here — alternating updates and linear averaging, both of which are cheap and
  both of which are absent from the current implementation.
- **Optimistic / predictive variants** **(new)**. Regret `O(√(Σ‖g_t − m_t‖²))`,
  giving `O(log T)` in self-play against another optimistic learner. Cheapest
  available improvement; see 18a′.4.
- **MCCFR.** Lanctot et al. (2009). The outcome-sampling estimator is the
  unbiased alternative (E2) of 18a′.2 and its variance analysis is what makes the
  ε-floor necessary.
- **Simultaneous-move MCTS.** Lisý et al. (2013), NIPS — the DUCT
  non-convergence result. Then Bošanský et al. (2016), *Algorithms for computing
  strategies in two-player simultaneous move games*, AI 237 — the survey, and
  still the single most directly relevant paper. **Read the convergence theorem
  for its hypotheses, not its conclusion**: the project's implementation violates
  two of the four (†Vol. I §7.3).

### Tier 2 — Support growth and best responses

- **Double oracle.** McMahan, Gordon, Blum (2003), ICML. The termination
  certificate is what turns a restricted-support LP into a statement about the
  full game — and it is the *only* upper bound on exploitability available to
  this project. **(new)** The Dantzig–Wolfe reading: double oracle is column
  generation on the minimax LP, and the best-response oracle is the pricing
  subproblem.
- **PSRO.** Lanctot et al. (2017), NIPS. **(new)** With ε-approximate oracles the
  terminated profile is a 2ε-Nash equilibrium, so approximate oracles give
  approximate certificates — far better than none. Also the meta-strategy-solver
  design space (Nash, uniform ⇒ fictitious play, rectified Nash, α-rank).

### Tier 3 — Regularised dynamics and last-iterate convergence

*This tier is now a prerequisite for Phase 18c, not a successor to it.*

- **Why unregularised dynamics cycle.** Mertikopoulos, Papadimitriou, Piliouras
  (2018), SODA. In continuous time FTRL conserves the Bregman divergence to any
  interior equilibrium and is Poincaré recurrent; in discrete time with constant
  step it spirals outward. This is the precise reason averaging is needed and
  the precise thing regularisation fixes.
- **QRE** **(new, and the bridge to the economics side)**. McKelvey & Palfrey
  (1995), GEB 10(1). The entropy-regularised equilibrium *is* a logit QRE with
  precision 1/τ; the principal branch as τ ↓ 0 is an equilibrium refinement. This
  is also what makes the τ-indexed difficulty ladder of 18d.3 principled rather
  than cosmetic.
- **Why the regularised target is well-posed** **(new)**. Strong monotonicity
  with modulus τ gives uniqueness, interiority, and a Lipschitz map `U ↦ x*_τ`
  with constant `O(1/τ)`, against a *discontinuous, set-valued* unregularised
  correspondence. This single fact is why v3 merges 18c and 18d.
- **R-NaD.** Perolat et al. (2021), ICML — the theory; Perolat et al. (2022),
  Science 378 — the system. Reward transformation `−η log(π/μ)` against a magnet.
  The right reference for the *hidden-information* variant, where search is hard;
  the wrong choice for the base game, where search is cheap and perfect
  information makes the Shapley operator available.
- **Magnetic Mirror Descent.** Sokota et al. (2023), ICLR. The same idea in
  first-order form with a closed-form update — a geometric interpolation of the
  current policy, the magnet, and the exponentiated advantage. The natural
  default for 18d.1.
- **NeuRD** **(new)**. Hennes et al. (2020), AAMAS. The policy-gradient
  realisation that behaves like replicator dynamics rather than softmax policy
  gradient, avoiding the vanishing-gradient pathology at the simplex boundary.

### Tier 4 — Systems, for context only

- Silver et al. (2018), AlphaZero, Science 362 — the pipeline 13b imitated, and
  specifically an *alternating*-game pipeline. Read it as a source of defaults to
  question, not to inherit: the visit-count policy target, the Monte-Carlo value
  target and the `max` backup are all alternating-game choices.
- Brown & Sandholm (2018), Libratus; (2019), Pluribus.
- Schmid et al., *Student of Games* — search plus CFR unified across perfect and
  imperfect information; the closest thing to a general recipe.

---

## 9. Questions worth holding open through the reading

1. Does the perfect-information structure of the base game make the whole
   imperfect-information apparatus (CFR, R-NaD) unnecessary until Chapter 11 —
   with regularised Shapley/LP backups the right answer until then? *v3's
   working answer is yes for the search and no for the policy update: MMD is
   worth adopting now precisely because it is the component that transfers.*
2. **(revised)** Is the bottleneck representational or algorithmic? v2 posed this
   as open. It is now largely settled on the algorithmic side: the slot-2 update
   has zero expected gradient, so no amount of capacity or compute helps it.
   What remains open is whether the *slot-1* head can express an equilibrium
   mixed strategy over ~9,000 factored logits, and 18b.3 answers that directly by
   supervised fitting. The pre-registered prediction is that it can.
3. **(revised, promoted to a deliverable)** What is the smallest game in this
   family whose equilibrium is *exactly* computable end to end? See 18e.4. That
   game, not 8×8, is where the algorithm should be validated, and it is probably
   the most publishable object the project will produce.
4. **(new)** How much of this game's strategic content actually lives in the
   reservation mechanic? Nothing is currently known: the campaign's cancellation
   rate is 0.000 by construction and δ has never been computed. If δ and ρ_can
   come back near zero on the 18b fixtures under equilibrium play, the mechanic
   is decoration and the design should be reconsidered; if they come back large,
   that is the paper's most interesting figure.
5. **(new)** Is γ large enough, often enough, to justify the whole edifice? The
   design's central claim is that simultaneity creates depth. γ measures exactly
   that, it has never been measured, and 18b.4 measures it for the first time.
   A distribution of γ concentrated near zero would be the most important
   negative result this project could produce — and it is better to know it
   before the definitive freeze than after.
