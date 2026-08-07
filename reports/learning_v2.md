# Learning v2 — Phase 18d.2 selection-rule comparison report

Generated 2026-08-07. Scope: `docs/LEARNING_ROADMAP_v3.md` §5, Phase 18d.2
("Comparison report... last-iterate NashConv, entropy trajectory and value
calibration for {RM⁺, optimistic RM⁺, Hedge, MMD} on identical seeds,
budgets and fixtures, with pre-registered estimands and a paired design").

> **Verdict up front.** No arm reaches an exploitability bar cleanly on
> every fixture, but the four arms separate clearly and consistently: the
> two RM+ variants converge fastest and most reliably wherever a fixture
> has a dominant (near-pure) equilibrium, MMD is consistently the
> **slowest and least converged arm within this budget** on all three
> fixtures, and Hedge sits in between. **Recommendation for the HEAVY
> profile: `selection="rm_plus"` (plain RM+), not MMD.** This echoes
> 18c.2's own honest finding (MMD-family regularized dynamics show real
> promise but are unstable/slow to settle at a fixed, un-annealed budget)
> from an independent angle (in-tree convergence rather than a training
> value target). See §4 for the explicit statement of what MMD would need
> to close the gap.

## 1. Pre-registered estimands (stated before running)

1. **Final-checkpoint exact NashConv** — `learn.nashconv.
   row_sketch_nashconv_exact` (new this pass, §5): best response searched
   over the **full exhaustive** program pool (`interop.openspiel_adapter.
   enumerate_legal_programs`), not just the arm's own seeded pool. Lower is
   better; `0` at a true equilibrium.
2. **Value-calibration trajectory** — `|read_out().value − solve_exact
   value|` against increasing simulation budget.
3. **Entropy trajectory** — Shannon entropy (bits) of the reported policy
   (`average_strategy()`: MMD's last iterate `x`, or the linearly-averaged
   `strategy_sum` for the other three) against increasing simulation
   budget.

## 2. Run configuration (for regeneration)

| constant | value |
|---|---|
| arms | `rm_plus`, `regret_matching` (optimistic RM+, the production default), `hedge`, `mmd` |
| fixtures | `matching_pennies_dodge`, `three_way_dodge`, `dominant_strategy_contrast` (18b's own exact fixtures) |
| simulation checkpoints | 50, 100, 200, 300, 400 (cumulative) |
| seeds | `{0, 1, 2, 3, 4}`, identical across all four arms per fixture (paired design) |
| `pool_size`/`pool_seed_size` | 24 (covers every fixture's full legal-action count — max observed 19, `three_way_dodge`'s black pool — so NashConv measures selection quality, not a pool-coverage artifact; see §5) |
| `epsilon_floor` | 0.02 |
| MMD params | `eta=1.0, alpha0=0.5, magnet_period=50` (18d.1's own validated defaults) |
| evaluator | uniform-prior synthetic evaluator (empty prior → `seed_program_pool` falls back to uniform-over-every-legal-action) — the same construction `test_row_sketch_mmd.py` validated 18d.1 with, isolating the selection rule's own convergence from a real network's prior-seeding luck |

The run script (`comparison_18d2.py`, scratchpad, not committed — matches
this session's established convention for report-generating drivers) is a
thin harness over the committed `learn.row_sketch`/`learn.nashconv`
package: `run_simulations`, `read_out`, `row_sketch_nashconv_exact`,
`solver.exact.solve_exact`.

## 3. Results

### 3.1 Final checkpoint (400 simulations), mean over 5 seeds

| fixture | arm | NashConv | value error | entropy (bits) |
|---|---|---|---|---|
| matching_pennies_dodge | rm_plus | **0.0494** | 0.000 | 1.38 |
| matching_pennies_dodge | regret_matching (opt. RM+) | 0.2329 | 0.100 | 1.34 |
| matching_pennies_dodge | hedge | 0.1166 | 0.000 | 1.34 |
| matching_pennies_dodge | mmd | 0.3877 | 0.000 | 1.21 |
| three_way_dodge | rm_plus | 0.1037 | 0.033 | 1.87 |
| three_way_dodge | regret_matching (opt. RM+) | 0.1216 | 0.033 | 1.88 |
| three_way_dodge | hedge | 0.1188 | 0.033 | 1.86 |
| three_way_dodge | mmd | **0.3302** | 0.067 | 1.86 |
| dominant_strategy_contrast | rm_plus | **0.0001** | 0.000 | 0.23 |
| dominant_strategy_contrast | regret_matching (opt. RM+) | 0.0001 | 0.000 | 0.23 |
| dominant_strategy_contrast | hedge | 0.0042 | 0.000 | 1.92 |
| dominant_strategy_contrast | mmd | 0.0237 | 0.000 | 1.89 |

Bold marks the best and worst NashConv per fixture. MMD is the worst arm
on two of three fixtures and second-worst on the third.

### 3.2 The dominant-strategy fixture is the clearest separator

`dominant_strategy_contrast` has a known near-pure equilibrium (18b). The
two RM+ variants collapse entropy from ~0.80 bits (checkpoint 50) to
**0.23 bits** by checkpoint 400 — correctly converging toward the
dominant pure action — while Hedge and MMD's entropy barely moves (2.10 →
1.92 for Hedge, 2.34 → 1.89 for MMD): **both stay far more diffuse than
the fixture's own known equilibrium calls for**, within this budget. This
is the single most legible finding in the whole comparison: RM+'s clipped
regret update is a *much* faster collapse mechanism onto a dominant
action than either Hedge's decaying-temperature softmax or MMD's
fixed-strength (`alpha0=0.5`) pull toward its own (initially uniform)
magnet.

### 3.3 A genuine surprise: optimistic RM+ underperforms plain RM+ on
`matching_pennies_dodge`

18a′.4's own theory (Syrgkanis et al. 2015) predicts optimistic RM+
*tightens* the convergence rate specifically **against another optimistic
learner** — exactly this setup, since both colours use the same
`selection` value. Instead, on this one fixture, optimistic RM+ finishes
with **higher** NashConv (0.2329) and nonzero value error (0.100 at
checkpoints 200–400, where plain RM+ is already exact) than plain RM+
(0.0494, value error 0.000 throughout). Plausible reading, not confirmed:
at this small pool size and short horizon, the one-step prediction
(`last_regret_increment`) is itself noisy early on, and that noise
appears to be actively destabilizing rather than accelerating convergence
before it has enough rounds to average out — the predicted benefit is an
asymptotic rate, not a small-sample guarantee. `three_way_dodge` and
`dominant_strategy_contrast` show no such reversal (the two RM+ variants
are statistically indistinguishable there), so this looks fixture-
specific rather than a general defect in the optimistic variant.

### 3.4 Trajectories (mean value error / entropy over 5 seeds; full table
in `comparison_18d2.log`, scratchpad)

MMD's value error is often *lowest at the earliest checkpoint* on
`matching_pennies_dodge` and `three_way_dodge` (e.g. 0.000 at checkpoint
50 on both, when every other arm still has nonzero error) but is not the
best arm by the final checkpoint on either — consistent with 18c.2's own
finding that MMD-style dynamics can show fast *early* progress that
doesn't hold up over a longer horizon at a fixed, un-annealed strength.

## 4. What MMD would need (per the roadmap's own DoD: "if no arm reaches
the strength bar, an explicit statement of what would be needed")

Two candidates, neither implemented this pass:

1. **More magnet updates within budget.** `magnet_period=50` means only 8
   magnet resets across 400 simulations; a shorter period (or a budget-
   scaled schedule) would let the magnet actually walk toward the true
   equilibrium within these short fixture horizons, rather than pulling
   the whole run toward a magnet that barely moves from its uniform
   initialization.
2. **Annealing `alpha0` toward 0** rather than holding it fixed — the
   roadmap's own §5 text notes convergence to the *regularized* equilibrium
   is what a fixed `alpha0` guarantees; walking `alpha0` down over the run
   (not just relying on repeated magnet updates at a fixed strength) is
   the more direct way to close the gap to the *true* equilibrium.

Both are concrete, testable follow-ups, not implemented here (matching
this pass's own scope: compare the arms as they exist today, not re-tune
MMD to win).

## 5. A methodology note worth recording

The first version of `row_sketch_nashconv_exact`'s own cross-validation
(a real, untrained `NetworkEvaluator` at `pool_size=8`) produced NashConv
≈ 1.0 for *every* arm — not a bug, but `three_way_dodge`-style fixtures
having up to 19 legal black programs while an undertrained network's
prior seeds a pool of only 4–8: best response searched over the *full*
pool (by design, §1 above) then trivially finds one of the excluded
programs. `pool_size=24` (covering every fixture's full legal-action
count) and the uniform-prior evaluator (§2) remove this confound, which
is why this report's numbers differ sharply from that first probe. Left
as a documented pitfall for whoever tunes `pool_size` next: a "row_sketch
policy is highly exploitable" reading is meaningless unless the pool
actually had a chance to contain the winning response in the first place.
