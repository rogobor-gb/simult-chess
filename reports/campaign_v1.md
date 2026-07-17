# Campaign v1 — Phase 11b pre-registered empirical report

Generated 2026-07-17T20:21:14.936269+00:00. 45000 games total. Estimands, statistics, and sample sizes were declared before any run (`docs/DEVELOPMENT_addendum_v1.1.md` §11b); every table below is traceable to the run-spec constants in `harness/campaign.py` (agent pair, `RuleSet` diff from baseline, seed range).

> **Interpretive caveat.** All balance statistics here are functionals of the state distributions induced by *these agents*, not of equilibrium play. The freeze this report supports is therefore provisional by construction (ruling A5) and is re-estimated after Phase 13 under learned agents.

## 1–2. Tournament matrix: draw rate & phase-count distribution

| pairing | n | white wins | black wins | draws | other | draw rate (95% Wilson) | phase median (IQR) |
|---|---|---|---|---|---|---|---|
| random_legal vs random_legal | 2223 | 1025 | 1097 | 101 | 0 | 0.045 [0.038, 0.055] | 40 (27–57) |
| random_legal vs greedy | 2223 | 508 | 1561 | 154 | 0 | 0.069 [0.059, 0.081] | 45 (32–68) |
| random_legal vs matrix_1ply | 2222 | 443 | 1335 | 443 | 1 | 0.199 [0.183, 0.216] | 55 (31–102) |
| greedy vs random_legal | 2222 | 1582 | 488 | 152 | 0 | 0.068 [0.059, 0.080] | 44 (30–65) |
| greedy vs greedy | 2222 | 547 | 609 | 1066 | 0 | 0.480 [0.459, 0.501] | 59 (39–93) |
| greedy vs matrix_1ply | 2222 | 408 | 814 | 1000 | 0 | 0.450 [0.429, 0.471] | 48 (29–94) |
| matrix_1ply vs random_legal | 2222 | 1340 | 427 | 454 | 1 | 0.204 [0.188, 0.222] | 57 (32–104) |
| matrix_1ply vs greedy | 2222 | 794 | 396 | 1032 | 0 | 0.464 [0.444, 0.485] | 48 (30–93) |
| matrix_1ply vs matrix_1ply | 2222 | 694 | 680 | 848 | 0 | 0.382 [0.362, 0.402] | 36 (23–72) |

## 3. T4-horizon attribution (fraction of draws by cause)

| pairing | draws | mutual_king_loss | repetition | horizon | other |
|---|---|---|---|---|---|
| random_legal vs random_legal | 101 | 0.891 | 0.010 | 0.099 | 0.000 |
| random_legal vs greedy | 154 | 0.935 | 0.013 | 0.052 | 0.000 |
| random_legal vs matrix_1ply | 443 | 0.553 | 0.000 | 0.447 | 0.000 |
| greedy vs random_legal | 152 | 0.914 | 0.000 | 0.086 | 0.000 |
| greedy vs greedy | 1066 | 0.626 | 0.059 | 0.315 | 0.000 |
| greedy vs matrix_1ply | 1000 | 0.562 | 0.037 | 0.401 | 0.000 |
| matrix_1ply vs random_legal | 454 | 0.581 | 0.000 | 0.419 | 0.000 |
| matrix_1ply vs greedy | 1032 | 0.565 | 0.060 | 0.375 | 0.000 |
| matrix_1ply vs matrix_1ply | 848 | 0.636 | 0.053 | 0.311 | 0.000 |

## 4–5. Reservation / cancellation / cooldown usage & material volatility

| pairing | reservation rate | cancellation rate | cooldown occupancy | material volatility (stdev of increments) |
|---|---|---|---|---|
| random_legal vs random_legal | 0.415 | 0.000 | 0.063 | 1.610 |
| random_legal vs greedy | 0.137 | 0.000 | 0.052 | 2.190 |
| random_legal vs matrix_1ply | 0.387 | 0.000 | 0.087 | 2.074 |
| greedy vs random_legal | 0.138 | 0.000 | 0.051 | 2.200 |
| greedy vs greedy | 0.000 | 0.000 | 0.053 | 2.122 |
| greedy vs matrix_1ply | 0.232 | 0.000 | 0.078 | 2.267 |
| matrix_1ply vs random_legal | 0.386 | 0.000 | 0.088 | 2.058 |
| matrix_1ply vs greedy | 0.232 | 0.000 | 0.078 | 2.264 |
| matrix_1ply vs matrix_1ply | 0.405 | 0.000 | 0.094 | 2.384 |

> **Cancellation-usage caveat, found during the campaign pilot.** `agents/candidates.py` (shared by `random_legal`, `greedy`, and `matrix_1ply` via `solver/supports.py`) only enumerates Move/Castle/Reserve candidates -- no agent in this roster can ever declare a `Cancel` action. Cancellation-usage rate is therefore structurally 0.000 everywhere in this campaign, and the `cancellation_enabled` A/B arm below is a confirmed null by construction, not evidence the rule has no effect. Revisit under Phase 13's learned agents, which aren't limited to this candidate set.

## 6. Color-symmetry audit (baseline config, pooled decisive games)

H0: p_White = 1/2. Pooled across all 9 baseline pairings: 7341/14748 decisive games White, p = 0.4978 [0.4897, 0.5058], two-sided exact binomial p-value = 0.5925. Power target was ~4.9×10³ decisive games at δ=0.02, α=.05, 1-β=.8 (§11b); since M3 proves operator-level symmetry exactly, any rejection here localizes to agent asymmetry, not a rules bug.

## A/B arms (matrix_1ply self-play; control = baseline mm slice)

Control: `tournament:matrix_1ply_vs_matrix_1ply` — n=2222, draw rate 0.382 [0.362, 0.402].

| arm | RuleSet diff | n | draw rate (95% Wilson) | Δ draw rate vs control | MDE at this n (α=.05, 1-β=.8) |
|---|---|---|---|---|---|
| cancellation_enabled=off | cancellation_enabled=False | 5000 | 0.390 [0.377, 0.404] | +0.009 | ±0.019 |
| intermezzo_reading=i | intermezzo_reading=i | 5000 | 0.409 [0.395, 0.423] | +0.027 | ±0.019 |
| pawn_fizzle=any_same_square | pawn_same_square_fizzle_scope=any_same_square | 5000 | 0.521 [0.507, 0.535] | +0.140 | ±0.019 |
| recapture_cooldown=off | recapture_cooldown=False | 5000 | 0.402 [0.388, 0.415] | +0.020 | ±0.019 |
| horizon=30 | horizon=30 | 2500 | 0.410 [0.391, 0.429] | +0.028 | ±0.027 |
| horizon=80 | horizon=80 | 2500 | 0.398 [0.379, 0.418] | +0.017 | ±0.027 |

### A/B arms: estimands 3–5 detail (same breakdown as the tournament matrix, for interpreting *why* a draw-rate delta moved)

| arm | mutual_king_loss | repetition | horizon | reservation rate | cooldown occupancy | volatility |
|---|---|---|---|---|---|---|
| *(control) mm baseline* | 0.636 | 0.053 | 0.311 | 0.405 | 0.094 | 2.384 |
| cancellation_enabled=off | 0.633 | 0.032 | 0.335 | 0.403 | 0.094 | 2.362 |
| intermezzo_reading=i | 0.616 | 0.040 | 0.344 | 0.394 | 0.096 | 2.359 |
| pawn_fizzle=any_same_square | 0.155 | 0.126 | 0.719 | 0.335 | 0.089 | 2.180 |
| recapture_cooldown=off | 0.631 | 0.038 | 0.331 | 0.399 | 0.094 | 2.370 |
| horizon=30 | 0.457 | 0.011 | 0.533 | 0.407 | 0.097 | 2.399 |
| horizon=80 | 0.762 | 0.061 | 0.177 | 0.392 | 0.092 | 2.329 |

## Violations summary (DoD: zero S0/S1 across all arms)

S0=0, S1=0, S2=0, S3=0 across 45000 games.

None found.

## Seed reference appendix

| run spec | agent pair | RuleSet diff | base seed | n games |
|---|---|---|---|---|
| tournament:random_legal_vs_random_legal | random_legal vs random_legal | baseline | 0 | 2223 |
| tournament:random_legal_vs_greedy | random_legal vs greedy | baseline | 1000000 | 2223 |
| tournament:random_legal_vs_matrix_1ply | random_legal vs matrix_1ply | baseline | 2000000 | 2222 |
| tournament:greedy_vs_random_legal | greedy vs random_legal | baseline | 3000000 | 2222 |
| tournament:greedy_vs_greedy | greedy vs greedy | baseline | 4000000 | 2222 |
| tournament:greedy_vs_matrix_1ply | greedy vs matrix_1ply | baseline | 5000000 | 2222 |
| tournament:matrix_1ply_vs_random_legal | matrix_1ply vs random_legal | baseline | 6000000 | 2222 |
| tournament:matrix_1ply_vs_greedy | matrix_1ply vs greedy | baseline | 7000000 | 2222 |
| tournament:matrix_1ply_vs_matrix_1ply | matrix_1ply vs matrix_1ply | baseline | 8000000 | 2222 |
| arm:cancellation_enabled=off | matrix_1ply vs matrix_1ply | cancellation_enabled=False | 100000000 | 5000 |
| arm:intermezzo_reading=i | matrix_1ply vs matrix_1ply | intermezzo_reading=i | 101000000 | 5000 |
| arm:pawn_fizzle=any_same_square | matrix_1ply vs matrix_1ply | pawn_same_square_fizzle_scope=any_same_square | 102000000 | 5000 |
| arm:recapture_cooldown=off | matrix_1ply vs matrix_1ply | recapture_cooldown=False | 103000000 | 5000 |
| arm:horizon=30 | matrix_1ply vs matrix_1ply | horizon=30 | 104000000 | 2500 |
| arm:horizon=80 | matrix_1ply vs matrix_1ply | horizon=80 | 105000000 | 2500 |

