# Learning v1 — Phase 13b LIGHT-profile training report

Generated 2026-07-23. Scope: `docs/LEARNING_DESIGN.md` (algorithm/evaluation
design) and `docs/DEVELOPMENT_addendum_v1.1.md` §Phase 13b (deliverables and
DoD). One LIGHT-profile run of 2000 self-play games, 5 checkpoints (every
400 games), each checkpoint run through the full evaluation protocol of
design §6.1/§6.2/§6.4 (first bullet of §6.3 only — see §5 below).

> **Verdict up front.** The 13b DoD is **not met**. Code quality and
> correctness deliverables are complete and green; the learned agent did
> not clear the strength gate at any checkpoint, and the exploitability
> estimate did not show a decreasing trend. See §6 for the full checklist.
> This report documents that outcome honestly rather than presenting a
> partial pass as a success.

## 1. Run configuration (for regeneration)

| constant | value |
|---|---|
| total self-play games | 2000 |
| batch size | 40 games/batch (54 batches recorded, including 2 resumes) |
| checkpoint cadence | every 400 games (5 checkpoints) |
| search budget | M=128 simulations/move (LIGHT default, design §4.3) |
| SGD steps per batch | 200 |
| max phases/game | 60 |
| self-play parallelism | 4 worker processes |
| network | `SimultChessNet`, B=6 residual blocks / F=64 filters, 75,314,317 parameters |
| device | Apple M4, PyTorch MPS backend |
| ladder games/checkpoint | 12 vs `random_legal`/`greedy`, 8 vs `matrix_1ply` (design §6.1 budget caveat) |
| ladder seeds | `base_seed = games_so_far * 7 + {1, 2, 3}` for {random_legal, greedy, matrix_1ply} |
| NashConv seed | `random.Random(games_so_far)` |
| M5 fixtures | standard start, midgame (knight+pawn), midgame (rook+pawn) — all χ-symmetric, `solved_value = 0` on each by M5 (Phase 10) |

The run script (`run_full.py`, not committed — an ephemeral driver kept
outside the repo per this session's scratch convention) is a thin harness
over the committed `learn/` package: `generate_self_play_games`,
`ReplayBuffer`, `train_step`, `save_checkpoint`/`load_checkpoint`,
`play_ladder_match`, `restricted_support_nashconv`,
`stage_policy_entropy`, `color_symmetry_spot_check`. All of `learn/` is
committed, tested, and covered by the seeds above, so the run is
reproducible in the sense that matters for the codebase — the specific
model weights from this run are not (see §5's artifact-loss note), but a
fresh run from the same code and seed scheme will exercise identical
logic paths.

## 2. Ladder results across checkpoints (design §6.1/§6.2)

| games | vs random_legal (win rate / Elo) | vs greedy (win rate / Elo) | vs matrix_1ply (win rate / Elo / gate p) |
|---|---|---|---|
| 400  | 0.583 / +58  | 0.750 / +191 | 0.375 / -89  / p=0.9375 |
| 800  | 0.583 / +58  | 0.375 / -89  | 0.438 / -44  / p=0.8125 |
| 1200 | 0.750 / +191 | 0.333 / -120 | 0.438 / -44  / p=0.8125 |
| 1600 | 0.833 / +280 | 0.542 / +29  | 0.750 / +191 / p=0.1094 |
| 2000 | 0.458 / -29  | 0.458 / -29  | 0.438 / -44  / p=0.875  |

**Primary strength gate (§6.2, 13b DoD)**: final checkpoint must beat
`matrix_1ply` at one-sided exact binomial p<0.01. **Not met at any
checkpoint** — the closest approach was p=0.1094 at 1600 games, and the
final (2000-game) checkpoint regressed to p=0.875, one of the weakest
results in the run. No checkpoint shows the strength gate trending toward
significance; 1600 games looks like a local high point that did not hold.

## 3. Exploitability — exact restricted-support NashConv (design §6.3, first bullet)

| games | standard_start | midgame_knight_pawn | midgame_rook_pawn |
|---|---|---|---|
| 400  | 2.662 | 0.587 | 8.453 |
| 800  | 0.000 | 3.885 | 7.789 |
| 1200 | 3.642 | 3.764 | 1.968 |
| 1600 | 4.540 | 5.596 | 9.987 |
| 2000 | 1.886 | 1.388 | 9.9999 |

`solved_value = 0` (M5's proven value) on all three fixtures at every
checkpoint — the sanity anchor holds throughout, so the network's value
head is at least reading the right ballpark on these small fixtures even
though the derived NashConv is not shrinking.

**13b DoD asks for a decreasing exploitability estimate across ≥3
checkpoints. Not met on any of the three fixtures.** None is monotonic:
`standard_start` and `midgame_knight_pawn` both rise then partially fall
with no consistent direction, and `midgame_rook_pawn` is essentially
pinned near its practical ceiling (~10) for the last two checkpoints —
if anything the trend on that fixture is worsening, not improving.

## 4. Learning-diagnostic figures (design §6.4)

| games | entropy (white / black) | color symmetry (white wins / decisive, p) |
|---|---|---|
| 400  | 3.089 / 2.733 | 160/361, p=0.0351 |
| 800  | 3.234 / 2.752 | 140/361, p=2.37e-05 |
| 1200 | 3.281 / 2.715 | 159/343, p=0.1949 |
| 1600 | 3.174 / 2.862 | 141/345, p=0.000819 |
| 2000 | 3.021 / 2.984 | 159/341, p=0.233 |

Entropy stays well above zero for both colors throughout (no collapse to
pure strategies), which is the healthy sign design §6.4 asks this figure
to check for. Color symmetry oscillates between non-significant and
strongly significant deviations from 1/2 checkpoint to checkpoint, with no
consistent direction — since M3 proves the rules operator itself is
exactly χ-symmetric (Phase 9), any deviation here localizes to the learned
agent's own asymmetric behavior on a given checkpoint's snapshot, not a
rules defect. By the final checkpoint the deviation happens to be
non-significant again (p=0.233), but given the oscillation across the
other four checkpoints this reads as noise around a genuinely unstable
policy rather than a resolved issue.

## 5. Diagnostics specified in the design but not run this cycle

Design §6.3's second bullet (**approximate exploitability via
best-response training**, i.e. training a from-scratch challenger against
each frozen checkpoint and reporting its win rate as an exploitability
lower bound) and §6.4's second bullet (**value-head calibration curves**,
binning predicted value against Monte-Carlo return) are both implemented
in the committed code (`learn/exploit.py::train_best_responder`,
`learn/diagnostics.py::value_calibration`) but were **not invoked** by this
run's evaluation harness. The calibration import was present but unused;
best-response training was never wired in. Given the strength-gate and
NashConv results above already show the DoD is not met, and given the
explicit decision to pause further algorithm-side work after this run,
neither gap was closed retroactively.

Note for any future attempt to backfill these: this would require the
saved checkpoints, and (per the artifact-loss note below) `checkpoint_02000.pt`
and its four predecessors from this specific run no longer exist. Both
diagnostics would need a fresh training run to produce checkpoints against
which to compute them.

**Artifact-loss note.** `run.log`, `results.json`, and all five checkpoint
files from this run lived only in this session's ephemeral scratch
directory (`/private/tmp/...`), which was cleared between conversation
sessions. The tables in this report were transcribed directly from
`results.json`/`run.log` at the time each checkpoint was produced (read
and reported in-conversation during monitoring), so the figures above are
accurate to the source, but the underlying files and trained weights
themselves are gone and cannot be independently re-inspected. **Process
lesson for future long runs**: persist `results.json` and at minimum the
final checkpoint into a location outside ephemeral `/private/tmp` (e.g.
a gitignored path inside the repo, or copy off-box) as soon as a run
completes, rather than leaving the only copy in session scratch.

## 6. Invariant violations (DoD: zero S0/S1 across all self-play games)

**Zero violations across all 2000 games and all 54 recorded batches** —
confirmed both by the per-batch `violations=0` log lines (every one of the
54 batches) and by an explicit scan for any nonzero `violations=` entry in
the full run log (none found). This held true across both operational
incidents below, including the resumed portions of the run.

## 7. Operational incidents during this run

Two network-mount-drop stalls occurred during the run (this is a
network-mounted repo; see the global environment note on SMB mounts and
compiled extensions — a related but distinct issue from what happened
here, which was `ProcessPoolExecutor` workers losing their editable-install
import path mid-sleep):

1. **Stall before the run's own log window began.** An overnight sleep
   cycle dropped the network mount backing the venv's editable install;
   self-play workers crashed with `ModuleNotFoundError`, and the run hung
   indefinitely (no timeout on `ProcessPoolExecutor.map`). Recovered by
   killing the stuck process and resuming from `checkpoint_00400.pt`.
2. **Second stall, 2026-07-22 14:27 to 2026-07-23 00:18 (~10 hours).**
   Same failure mode: the mount dropped, all 4 workers crashed with
   `ModuleNotFoundError`, and the run sat idle overnight with zero CPU
   usage and no exception surfaced to the log (the crash was only visible
   in the workers' own stdout capture). No games or checkpoint data were
   lost — `checkpoint_01600.pt` was intact — but the batch in progress at
   the time (toward 1760 games) had to be regenerated. Recovered the same
   way: killed the stuck process, updated the resume point to
   `checkpoint_01600.pt`, relaunched under `caffeinate -i` plus an
   additional `caffeinate -s -w <pid>` (`PreventSystemSleep`) assertion
   layered on to reduce the chance of a repeat.

A third, milder incident (batches around games 640–720) saw per-game
throughput degrade to 300–400s/game (vs. a normal 40–55s/game) due to CPU
contention from another process on the machine; this self-resolved without
intervention and did not stall the run or corrupt any data.

None of these incidents affected correctness (S0/S1 stayed at zero
throughout) — they were pure availability/infrastructure issues, not
algorithm bugs, and are unrelated to the DoD-not-met verdict above.

## 8. DoD checklist (`docs/DEVELOPMENT_addendum_v1.1.md` §Phase 13b)

| criterion | result |
|---|---|
| ruff/mypy/pytest green (network code included) | **met** — ruff and mypy clean; full pytest suite green |
| final checkpoint beats `matrix_1ply` at one-sided exact binomial p<0.01 | **not met** — best p=0.1094 (1600 games); final checkpoint p=0.875 |
| decreasing exploitability estimate across ≥3 checkpoints | **not met** — no fixture shows a monotonic decrease; `midgame_rook_pawn` is pinned near its ceiling by the end |
| invariant harness clean over all self-play games (S0/S1 = 0) | **met** — zero violations across 2000 games / 54 batches |
| report committed | this document |

**Overall: DoD not met.** The infrastructure, correctness, and tooling
side of Phase 13b is solid and fully delivered — the `learn/` package
(network, SM-MCTS with regret matching, self-play, training,
checkpointing, the full evaluation harness) works end-to-end, survived two
real infrastructure incidents without any data loss or correctness
regression, and every invariant held throughout. What did not happen is
convergence to a strong or low-exploitability policy within this run's
budget (2000 games, M=128 simulations, B=6/F=64 net). The oscillating,
non-monotonic strength and exploitability numbers across checkpoints point
more toward the training configuration (learning rate, search budget,
self-play data volume relative to a 75M-parameter network) than toward a
simple "needs more games" story, since there is no visible trend even in
the direction of convergence.

## 9. Conclusion and next steps

Per `docs/DEVELOPMENT_addendum_v1.1.md`, Phase 13b is the last phase in
this addendum (v1.1 covers Phases 9–13), and Gate 13b's stated next action
is: *"re-run the Phase 11 campaign under learned agents and revisit the
provisional freeze (A5)."* **That next action is not triggered by this
run** — re-running the balance campaign under a learned agent that has not
demonstrated equilibrium-adjacent play would not produce a meaningful
re-estimation of the provisional freeze. This is deliberately left as
future roadmap work, to be scoped separately (e.g. hyperparameter tuning,
a larger search budget or smaller network, more self-play games, or a
revised training schedule) rather than folded into this report or gate.

This report, together with the `learn/` package and its test suite,
concludes the current roadmap (`docs/DEVELOPMENT_addendum_v1.1.md`,
Phases 9–13) as specified. Ruling A5's provisional freeze remains in
force pending a future learning iteration that clears the 13b DoD.
