# Simultaneous Chess — Learning-System Design, v1

*Companion to `simultaneous_chess_spec_v1.md` (hereafter **spec**) and
`INVARIANTS.md` (hereafter **inv**). This document is the design for the
self-play learning system of Phase 13 (`docs/DEVELOPMENT_addendum_v1.1.md`
§Phase 13a). It specifies the formal setting, the search-and-learning
algorithm, the network, two hardware profiles, the Rust-port annex, the
evaluation protocol, and the imperfect-information variant hook. It fixes
enough to implement Phase 13b (`learn/`) without further design decisions.*

*Licensed under CC BY 4.0.*

**Scope.** Design only — **no learning-system code** ships with this
document (addendum §13a DoD). The one exception is measurement: the §4
profiling numbers were produced by a throwaway harness run against the
committed engine (methodology in Appendix A); that harness is not part of
the package and is not committed. The LIGHT profile is *built* in 13b; the
HEAVY profile and the Rust port are *documented only* (ruling A9).

**Source of truth for the tensor encoding.** §3 folds in and extends the
observation encoding defined in `src/simult_chess/interop/openspiel_adapter.py`
(`SimultChessObserver`), which Phase 12 established as the canonical encoding.
Where this document and that docstring differ, the difference is an
*intentional extension* flagged here (the reservation-pairing channel, §3.2)
and must be propagated back into the adapter docstring when 13b implements it.

---

## 0. Notation and prerequisites

Symbols follow the spec's §1 table: $s$ a state, $\Phi(s,\pi_\mathrm W,\pi_\mathrm B)$
the transition operator, $\pi_\omega\in\Pi_\omega(s)$ a *program* (a tuple of
$1$–$N$ actions) for color $\omega\in\{\mathrm W,\mathrm B\}$, $L(s,\pi)$ the
legality predicate (spec §4.4), $\chi$ the color-swap involution (spec §5,
inv M3), $H$ the no-progress horizon (spec §10), and $u\in\{-1,0,+1\}$ the
terminal utility (White's perspective; spec §8.1). $N=2$ in v1 (`RuleSet.n_actions`).
The `RuleSet` (`src/simult_chess/rules/ruleset.py`) parameterizes $\Phi$; the
learning system treats it as a fixed input, never a learned quantity.

---

## 1. Formal setting

### 1.1 The game object

Under perfect information (the base version — the hidden-information variant
is §7), $\Gamma=(s_0,\{A_\omega(s)\},\Phi,T,u)$ is a **finite, deterministic,
zero-sum, two-player simultaneous-move stochastic game** in the sense of
Shapley (1953) (spec §8, line 264). "Stochastic game" is used in Shapley's
sense — the stage structure, not chance in the transition: $\Phi$ is
deterministic (the game registers with pyspiel as
`chance_mode=DETERMINISTIC`, Phase 12), so there are no chance nodes. The
"stochasticity" that matters is entirely in the players' **mixed strategies**.

Each decision phase at state $s$ is a **matrix game**
$U(s)\in\mathbb R^{|A_\mathrm W(s)|\times|A_\mathrm B(s)|}$ with
$U_{ij}(s)=u^\ast\!\big(\Phi(s,\pi_i,\pi_j)\big)$, where $u^\ast$ is the
continuation value of the successor (spec §8, line 264). This is exactly the
object Phase 10's `solver/stage_matrix.py` builds for a *restricted* support;
the learning system's job is to approximate $u^\ast$ and to play the induced
matrix games well without enumerating them.

### 1.2 Value and why it exists

Define the value operator on states by

$$V(s) \;=\; \operatorname*{val}_{x\in\Delta(A_\mathrm W(s)),\;y\in\Delta(A_\mathrm B(s))}\;
\mathbb E_{\substack{\pi_\mathrm W\sim x\\ \pi_\mathrm B\sim y}}\big[\,r(s') + V(s')\,\big],
\qquad s' = \Phi(s,\pi_\mathrm W,\pi_\mathrm B),$$

where $r(s')=u(s')$ if $s'$ is terminal and $0$ otherwise (reward is
terminal-only; spec §8.1, Phase 12 `reward_model=TERMINAL`), and
$\operatorname{val}$ is the minimax value of the enclosed matrix game.

**Existence.** By the minimax theorem each stage game has a value in mixed
strategies; by Shapley (1953) the zero-sum stochastic game has a value. The
no-progress rule (spec §10, T4) makes the horizon *effectively finite* — the
public component of the state cannot cycle indefinitely without triggering a
draw at $H$ no-progress phases — so $V$ is obtained by **backward induction
over matrix games and no discounting is required** (spec §8, line 266). We
adopt the undiscounted return throughout (§2.4, §3.4): $\gamma=1$.

**Why the max operator is inadmissible.** AlphaZero backs a *scalar* value up
a tree with $V(s)=\max_a Q(s,a)$ (single-agent, or alternating-move via
negamax). That operator presumes a pure optimal action exists. Here it does
not: the spec embeds **Matching Pennies** in a minimal king-dodge subgame
(spec §8, line 280) — attacker aims at $x_1$ or $x_2$, king stays or flees;
$A$ wins iff it matches — whose unique equilibrium is uniform mixing with
value $\tfrac12$ and **no pure equilibrium**. Zermelo's theorem does not apply
(spec §8, line 266). A max-based backup would collapse this to a pure choice
and reintroduce exactly the first-mover / commitment artifact that
simultaneity and inv **M3** (χ-equivariance — the spec's "single most
important test") were designed to eliminate. The backup operator must
therefore be $\operatorname{val}$ (matrix-game minimax), approximated by
regret-based search (§2), and the policy target must be a **mixed** strategy
(§3.3). The spec states the consequence directly: "an engine must approximate
stage equilibria (SM-MCTS, regret matching / CFR) rather than perform pure
minimax" (spec §8, line 280).

### 1.3 What "playing well" means here

A strategy is evaluated against the *value*, not against a fixed opponent:
the natural error metric is **exploitability / NashConv** (§6.3), not raw
win-rate against a baseline. Win-rate ladders (§6.1) are a coarse,
interpretable proxy; exploitability is the metric with game-theoretic
standing. This framing is inherited straight from §1.2 and drives §6.

---

## 2. Algorithm: policy-value network + SM-MCTS with regret matching

### 2.1 Overview

The agent is an AlphaZero-shaped self-play loop with one structural change
forced by §1.2: the tree search is a **simultaneous-move MCTS (SM-MCTS)**
whose per-node selection rule is a **regret-matching** (equivalently
Exp3-family) no-regret learner over each player's own action set, and whose
per-node output is a **mixed strategy** for each player. A policy-value
network $f_\theta(s)=(\mathbf p_\mathrm W,\mathbf p_\mathrm B,v)$ supplies
action priors and a leaf value. Self-play generates $(s, \text{search mixed
strategies}, z)$ tuples; the network is trained to match them; the improved
network seeds a stronger search; repeat.

### 2.2 Why regret matching, not decoupled UCT

Decoupled UCT (DUCT) runs an independent UCB1 bandit per player at each node.
It is simple and often strong empirically, but it is **not guaranteed to
converge to a stage-game equilibrium** — there are simultaneous-move
counterexamples where DUCT's marginals cycle or settle away from the minimax
value (Shafiei, Sturtevant & Schaeffer 2009; Lisý, Lanctot & Bowling 2013).
Because §1.2 makes equilibrium (not a pure best response) the target,
convergence to the stage equilibrium is a correctness property, not a nicety.

**Regret matching** (Hart & Mas-Colell 2000) and the **Exp3**-based variant
run a no-regret learner per player per node; the *average* strategy provably
converges to the stage-game minimax value in self-play, and the SM-MCTS built
on it converges to the subgame equilibrium in the tabular limit (Lisý et al.
2013; Lanctot et al. 2014). We adopt **regret matching** as the v1 in-tree
rule (its updates are cheaper than Exp3's exponential weighting and it has no
temperature to tune); **Exp3** is the documented fallback if a state class
shows poor convergence. The choice is a search parameter, not a rule, and is
exposed in the `learn/` config.

### 2.3 The search, concretely

Each tree node caches its native `State` and, per color, the network prior
$\mathbf p_\omega$ over that node's legal actions (§3.3), plus a
regret-matching accumulator (cumulative regret $R_\omega$ and cumulative
strategy $\bar\sigma_\omega$) over each color's own action set. One simulation:

1. **Descend.** At each internal node, form each player's current strategy
   $\sigma_\omega = \mathrm{RM}(R_\omega)$ (regret-matching: positive-regret
   normalization, uniform if none positive), blended with the network prior
   as an initializer/anchor. Sample $a_\mathrm W\sim\sigma_\mathrm W$,
   $a_\mathrm B\sim\sigma_\mathrm B$ **independently** (decoupled selection),
   forming the joint program pair.
2. **Transition.** $s' = \Phi(s, \pi_\mathrm W, \pi_\mathrm B)$ (one cheap
   call, §4). Follow the cached edge if it exists, else expand.
3. **Expand & evaluate.** At a new leaf, run $f_\theta$ once for value $v$ and
   the two priors. Terminal leaves use $u(s')$ directly (no bootstrap).
   **Leaves are evaluated by the network value, never by rollouts** — random
   rollouts would pay the pure-Python move-generation cost (§4, the measured
   bottleneck) per ply for a high-variance estimate; the value head is both
   cheaper and lower-variance.
4. **Back up (decoupled, zero-sum).** Propagate $v$ to every node on the path.
   At each node, update each player's regret accumulator with the
   counterfactual value of each of its actions against the *opponent's
   sampled action* (decoupled regret update; White maximizes $v$, Black
   minimizes it — one signed scalar, $u^\ast$ is White's utility).

After $M$ simulations, the node's **average strategy** $\bar\sigma_\omega /
\sum\bar\sigma_\omega$ is the search-derived mixed strategy for color
$\omega$; this — not a visit-count argmax — is the policy target (§3.3) and
the move actually played in self-play (sampled, with the usual early-game
temperature for exploration).

### 2.4 Training targets and loss

Self-play stores, per phase, $(s,\ \bar\sigma_\mathrm W,\ \bar\sigma_\mathrm B,\ z)$
where $z\in\{-1,0,+1\}$ is the game's terminal outcome (White's perspective;
$\gamma=1$, undiscounted per §1.2). The loss is the AlphaZero form with a
**two-headed policy** (one head per slot, §3.3), cross-entropy against the
search strategies and MSE against the outcome:

$$\mathcal L(\theta) = \underbrace{(v_\theta(s)-z)^2}_{\text{value}}
\;-\;\underbrace{\textstyle\sum_{\omega}\bar\sigma_\omega\cdot\log \mathbf p_{\theta,\omega}(s)}_{\text{policy (both colors, both slots)}}
\;+\;\underbrace{c\lVert\theta\rVert^2}_{\text{L2}}.$$

Because the network sees a perfect-information, player-symmetric state (§3.1),
it predicts *both* colors' strategies from one forward pass — the natural
analogue of `solver/agent.py`'s "build a support for both colors from the same
public state" (Phase 10). Self-play draws the White move from
$\mathbf p_{\theta,\mathrm W}$'s search refinement and the Black move from
$\mathbf p_{\theta,\mathrm B}$'s, so a single search tree yields both moves of
a phase.

### 2.5 Self-play loop

Standard AlphaZero outer loop through the **Phase 12 adapter** (self-play uses
the registered `simult_chess` game and its `clone()` for tree expansion — the
clone override validated in Phase 12 is what makes tree search over the native
state cheap and correct): generate games → append to a replay buffer → sample
minibatches → SGD → checkpoint → promote the new network into the actors.
Every game is seeded and bit-reproducible (the `Agent` contract and
`play_one_game`'s determinism, Phase 6); the learned agent conforms to the
`Agent` Protocol (`agents/base.py`) so it drops into `referee/match.py`,
`harness/selfplay.py`, and the evaluation ladder unchanged.

---

## 3. Network

### 3.1 Input encoding (extends the Phase 12 `SimultChessObserver`)

The input is the canonical tensor encoding of
`interop/openspiel_adapter.py`, **plus a reservation-pairing channel** (§3.2).
The base encoding (Phase 12, verbatim intent):

- **`planes`, shape `(17, 8, 8)`** — 12 board-occupancy planes (one per
  `(color, type)`, spec §1.1); 1 **cooldown** plane (spec §7); 4
  **reservation-actor** planes `(white_defenders, white_proteges,
  black_defenders, black_proteges)`, one square marked per live token in
  that role in ≥1 active reservation.
- **`scalars`, shape `(7,)`** — the four `CastlingRights` booleans (spec
  §1.2), the no-progress counter $\nu$, the horizon $H$, and phase-index
  parity ($\texttt{phase\_index}\bmod 2$).

Two encoding facts verified against the spec during this design, both
load-bearing:

1. **The binary cooldown plane is loss-free.** Cooldown is *definitionally
   one phase*: $C'=\{\text{tokens that displaced this phase}\}\setminus(\text{pawns}\cup\text{kings})$,
   recomputed fresh each phase (spec §6.7 line 243, §7 line 252). There is no
   remaining-duration counter to lose — a token is cooling or it is not.
2. **Perfect information ⇒ the encoding is player-independent.** `set_from`
   ignores the `player` argument (Phase 12); the same tensor serves both
   colors. This is what lets one forward pass predict both colors' policies
   (§2.4). It is also precisely the fact that *changes* under the hidden
   variant (§7).

### 3.2 The reservation-pairing channel (extension, decision of 2026-07-17)

The four base reservation planes mark *where* defenders and protégés stand,
not *which defender protects which protégé* — the observer docstring flags
this as a Phase-13 revisit, because a full pairwise `(square, square)`
relation does not fit a per-square plane. This omission is material: the
entire strategic core of the game is *who defends whom* — the defended-pair
theorem (spec §12.1) makes the exact exchange value of a capture depend on
whether the defender is a **contact** defender (classical exchange
arithmetic) or a **ranged** defender vulnerable to interposition (a
refutation with no classical analogue). A value head blind to pairing cannot
distinguish these and is handicapped exactly where the game is most subtle.

**Design:** add a lightweight **pairing channel** to `planes` so the trunk
sees the $D\!\to\!Q$ relation of each standing reservation. The v1 realization
is **2 additional planes per color** (4 total), keyed on the age-oldest
reservation per square (mirroring the *oldest-valid-fires* rule, spec §6.4,
R-multi-in line 123):

- a **defender→protégé displacement plane**: at each defender's square, encode
  the *offset* $(\Delta\mathrm{file},\Delta\mathrm{rank})$ to its (oldest
  active) protégé, as two channels (normalized to $[-1,1]$);
- symmetric protégé→defender offset planes are **not** added — the relation is
  recoverable from either endpoint, and a single directed encoding at the
  defender square suffices (the defender is the actor that fires).

This makes the input **`planes` shape `(21, 8, 8)`** (17 base + 4 pairing) and
leaves `scalars` unchanged. When a square hosts a defender of several
protégés (spec R-multi-out line 124), the oldest active reservation is
encoded, matching firing precedence; the multiplicity is a rare, second-order
signal deliberately not encoded in v1 (flagged for revisit alongside any
future depth-of-defense features). **This changes the Phase 12 source-of-truth
encoding**; when 13b implements it, the `SimultChessObserver` docstring and
`_NUM_PLANES` must be updated in lockstep, and the Phase 12 conformance test's
tensor-shape assertions with them.

### 3.3 Policy head: fixed factored grid, autoregressive over the two slots

A pyspiel "action" is a **state-dependent index** into
`enumerate_legal_programs` (index 5 is a rook move in one state, a reservation
in another; Phase 12 docstring). A softmax over that index is meaningless
across states, so the policy is a **fixed, state-independent factored grid**,
masked to the legal set per state — the AlphaZero approach, adapted to this
game's four action kinds. Decision of 2026-07-17.

**Per-slot action grid** (fixed output geometry; illegal entries masked to
$-\infty$ before softmax):

| Action kind | Encoding | Size |
|---|---|---|
| `Move(p, τ, promo)` | `from-square (64) × move-type (76)`: 56 sliding (8 dir × 7 dist), 8 knight, 12 promotion (3 last-rank directions × 4 promo types {n,b,r,q}). A last-rank pawn move is encoded **only** via the promotion planes (promotion is explicit and forced, spec §6.5 — unlike AlphaZero's implicit-queen convention). | 4 864 |
| `Castle(side)` | 2 dedicated entries {king, queen}. Kept separate from king moves because ruling A3 made the actor set `{king, flank_rook}` — it is its own action. | 2 |
| `Reserve(D, Q)` | `defender-square (64) × protégé-square (64)`. | 4 096 |
| `Cancel(ρ)` | **protégé-square-keyed** (64), with an **oldest-valid tie-break** when a protégé square hosts several standing reservations — mirroring R-multi-in's "oldest valid fires" so the head's meaning matches how the engine disambiguates. Decision of 2026-07-17. | 64 |
| **Total per slot** | | **≈ 9 026** |

**Autoregressive factorization over the $N=2$ slots.** The policy is
$p(\pi) = p(a_1)\,p(a_2\mid a_1)$: a **slot-1 head** over the grid, then the
chosen $a_1$ is embedded and concatenated to the trunk features for a
**slot-2 head** (conditioned on $a_1$), which also carries a **"no second
action"** entry for single-action programs (legal when $N=2$ but only one
action is declared; spec §4.1). Conditioning is required, not cosmetic:

- the **aggressive dual** (spec §6.2 line 126) — move a piece with slot 1,
  then reserve *that arriving piece* as a protégé with slot 2 — is only
  coherent if slot 2 sees slot 1's destination;
- the **interposition refutation** (spec §12.1(b)) — capture a ranged
  defender's protégé with slot 1, interpose on the recapture ray with slot 2
  — only makes sense given slot 1.

A flat joint softmax is dismissed on cardinality (the count the addendum asks
for): the joint program space is $\sim(9\text{k})^2\approx 8\times10^7$
ordered pairs — squarely the spec's own $10^6$–$10^8$ stage-matrix estimate
(spec §8 line 293) — while the autoregressive factorization is two heads of
$\sim9\text{k}$, i.e. $\sim1.8\times10^4$ outputs. The factorization also has a
*measured* efficiency payoff at search time (§4.3): it lets a node compute two
$O(\text{pool})$ legality masks instead of the $O(\text{pool}^2)$ full program
enumeration — ~2 ms vs ~35 ms per node.

**Action space includes `Cancel` — the full $L(s,\pi)$.** The learned agent's
action space is the *complete* legal-program set, `Move | Reserve | Castle |
Cancel`. This is a deliberate correction of a gap: neither
`agents/candidates.py` nor the Phase 12 `enumerate_legal_programs` currently
emits `Cancel` (both build only Move/Castle/Reserve), which is exactly why the
Phase 11b campaign reports a *structural* cancellation rate of $0.000$ and
labels the `cancellation_enabled` A/B arm a "confirmed null by construction"
(`reports/campaign_v1.md`, and PROJECT_STATUS). Decision of 2026-07-17: 13b
**extends the exhaustive enumeration to emit `Cancel` actions** (and adds the
`Cancel` grid above + its legality masking), so the learned agent can actually
use the mechanic. This is what makes the post-13b campaign re-run (ruling A5)
able to measure cancellation usage under strong play rather than re-confirm a
construction artifact.

### 3.4 Trunk and value head

A small **residual trunk**: a 3×3 conv stem to $F$ filters, then $B$
residual blocks (each two 3×3 convs + BatchNorm + ReLU, identity skip), over
the $(21,8,8)$ input. **Policy heads**: a 1×1 conv to a small channel count,
flattened to a `Linear` producing the per-slot grid logits (slot-2's Linear
takes the extra $a_1$ embedding). **Value head**: a 1×1 conv to 1 plane →
`Linear` → ReLU → `Linear` → `tanh`, output $v\in[-1,1]$, matching the
terminal utility range (spec §8.1) and the pyspiel `min/max_utility=±1`
(Phase 12). $B$ and $F$ are fixed by §4's profiling: **LIGHT = $B{=}6$,
$F{=}64$.**

---

## 4. Hardware profiles (ruling A9)

**One codebase, two configuration profiles.** LIGHT is built in 13b; HEAVY is
documented only; no distributed code is in scope. The defaults below are fixed
by measurement, not guessed — the addendum requires it. All numbers were
produced on the actual development machine by the Appendix A harness.

### 4.1 The measured machine

> **Apple M3** (4 performance + 4 efficiency cores), 16 GB unified memory,
> macOS 26.5; Python 3.10.11; PyTorch **2.13.0** with MPS
> (`torch.backends.mps.is_available() == True`). Φ and legality run in
> pure-Python, single-threaded; the network runs on MPS.

*Note on the "M4" in ruling A9:* the addendum nominally names an M4 laptop;
the machine on hand is an **M3**. The LIGHT profile therefore targets this
class of 8-core Apple-silicon laptop as measured; an M4 offers a modest uplift
and does not change any sizing conclusion below (the bottleneck is
single-thread pure-Python, §4.4).

### 4.2 Profiling table

Representative states: 300 pre-phase states sampled across mixed
greedy/random self-play games (opening through midgame). Full method and raw
output: Appendix A.

| Quantity | Median | p90 | Throughput |
|---|---:|---:|---|
| **Φ transition** (one phase, programs given) | 0.139 ms | — | **7 191 phases/s** |
| **Full $O(\text{pool}^2)$ program enumeration**, both colors (the `enumerate_legal_programs` path) | 35.0 ms | 107 ms | ~29 states/s |
| — pool build only, both colors | 0.42 ms | 1.8 ms | |
| — slot-1 legality mask $O(\text{pool})$, both colors | 0.66 ms | 1.5 ms | |
| — slot-2 mask given slot-1, one color | 0.48 ms | 1.1 ms | |
| **Per-slot masking total** (autoregressive head, per node) | **~2.0 ms** | ~4 ms | ~500 nodes/s |
| `\|`legal programs`\|` both colors | 1 174 | 2 978 | (max 5 340) |
| pool size both colors | 52 | — | (max 114) |

Network (`(21,8,8)` input, $B{=}6$, $F{=}64$, ~9 k policy logits, MPS):

| Op | bs=1 | bs=8 | bs=64 | bs=256 |
|---|---:|---:|---:|---:|
| forward (ms/batch) | 1.40 | 1.51 | 2.81 | 11.0 |
| forward (pos/s) | 716 | 5 292 | 22 787 | 23 266 |
| forward+backward (ms/batch) | — | — | 15.6 | 45.5 |
| forward+backward (pos/s) | — | — | 4 110 | **5 623** |

### 4.3 Deriving the LIGHT defaults

**Per-simulation cost** (autoregressive head, one expanded leaf): one $O(\text{pool})$
masking (~2 ms) + one Φ (~0.14 ms) + one network value. With **leaf-batched
network evaluation across parallel actors** (the practical way to batch in
SM-MCTS, since a single tree's leaf evals are sequential), the amortized
network cost is ~0.04–0.3 ms/position (bs 64–256 forward). So

$$t_\text{sim} \;\approx\; \underbrace{2.0}_{\text{mask}} + \underbrace{0.14}_{\Phi} + \underbrace{0.05\text{–}0.3}_{\text{net, batched}} \;\approx\; 2.2\text{–}2.5\ \text{ms}.$$

**Simulations per move.** The addendum's ~$10^2$ target is well-founded here:
at $M=128$ simulations, a move costs $\approx 0.3$ s; a median ~50-phase game
(campaign phase-count medians, `reports/campaign_v1.md`) is $\approx 15$ s of
search. With the 4 P-cores running parallel self-play actors, aggregate
throughput is on the order of $10^3$ games/hour — enough for a first LIGHT
run over an overnight-to-few-days budget. **LIGHT default: $M=128$
simulations/move** (config-exposed; sweepable 64–256).

**Network size.** $B{=}6$, $F{=}64$ trains at ~5.6 k positions/s (bs=256
fwd+bwd) and evaluates at ~23 k/s (batched forward) — both comfortably faster
than move-generation, so the net is *not* the bottleneck and there is no case
for shrinking it below this for speed. (Its ~18.6 M parameters are dominated
by the final policy `Linear` of ~9 k outputs, not the trunk; a conv-based
policy head would cut parameters substantially without changing the
throughput story, and is a reasonable 13b micro-optimization — noted, not
mandated.) **LIGHT defaults: $B{=}6$ residual blocks, $F{=}64$ filters.**

### 4.4 The bottleneck (a measured correction to the addendum's prediction)

The addendum predicted *pure-Python Φ throughput* would be the bottleneck. The
measurement says otherwise and sharpens the picture: **Φ is cheap (0.14 ms);
the bottleneck is pure-Python legal-move generation and legality checking**
(`core.geometry.pseudo_legal_trajectories` + `core.legality.is_legal_program`).
Two consequences, both already reflected above:

1. **The autoregressive fixed-factored head is not just a cardinality choice
   — it is the primary throughput lever.** Materializing the full legal
   *program* set per node (the exhaustive `enumerate_legal_programs`) costs
   ~35 ms (p90 107 ms); the two $O(\text{pool})$ masks the autoregressive head
   actually needs cost ~2 ms — a **~17× reduction**, and the difference
   between a ~2 ms and a ~35 ms per-node search step is the difference between
   a usable and an unusable LIGHT profile.
2. **It re-aims the Rust annex (§5).** The port's payoff target is the
   legality/geometry move-generation path in `core/`, not Φ per se.

**HEAVY (documented only).** Larger trunk ($B\sim20$, $F\sim256$), more
simulations ($M\sim800$), and many parallel actors feeding a batched
inference server; a GPU makes the (already non-bottleneck) network free
relative to move-gen, so HEAVY's marginal returns come almost entirely from a
faster core (the Rust port, §5) and more actors, not a bigger net. No
distributed/training-server code is in scope for v1 (ruling A9).

---

## 5. Rust `core` port — annex, not a work item (ruling A9)

**Not built.** This section fixes the contract a future port must meet so that
it can be dropped in without re-opening the design.

**Scope: `core/` only.** The port covers the transition operator and its hot
dependencies — the modules §4.4 identifies as the bottleneck:
`core/geometry.py` (trajectory generation), `core/legality.py` (the
$L(s,\pi)$ clauses), and `core/phi.py` with its stages
(`core/stages/*`). Everything above `core/` (referee, agents, solver, interop,
learn) stays Python and calls the port through a thin FFI boundary. **Semantics
live in the spec; a port is an implementation, never a fork** — it may not
introduce, drop, or reinterpret any rule.

**FFI boundary.** The port exposes exactly the pure functions the Python side
needs: `phi(state, program_W, program_B, ruleset) → (state', PhiTrace)`, the
legality predicate, and exhaustive legal-action generation (for the search
masks). State crosses the boundary by a canonical serialization (the same
token-id/reservation-index scheme `net/protocol.py` already uses to move a
`Program` between peers is the model). The Python `PhiTrace` /
`PhiResult` shapes are the contract; the port must reproduce them field for
field.

**Conformance bar (hard gate, before any training run may use the port).**

1. **Bit-identical `PhiTrace`s** against the Python reference over the entire
   golden-fixture suite (every worked example in the spec and the existing
   `tests/` fixtures), compared field-by-field, not just on the successor
   state — the trace is the contract because the invariants read it.
2. **A $\geq 10^3$-game seeded co-play sweep** (port vs. Python reference from
   identical seeds) producing bit-identical state trajectories *after every
   phase*, in the shape of the Phase 12 conformance test — reusing that test's
   harness, pointed at the port instead of the adapter.
3. **Invariant-harness-clean** over that sweep at every severity (the sweep
   runs the harness in lenient mode, as every campaign sweep does): **zero
   S0/S1**, S2/S3 reported.

Only after all three pass may a training run consume the port. Until then,
LIGHT runs on pure-Python `core`; the port is a *drop-in accelerator* gated
behind proof of identical behavior, exactly as the `oracle`/`solver`/`openspiel`
extras are quarantined today.

---

## 6. Evaluation protocol

Every figure below is seeded and regenerable; all sweeps run the invariant
harness in lenient mode (learning does not get to skip correctness). This is
the "analyze the learning results" contract for the 13b report
(`reports/learning_v1.md`).

### 6.1 Relative-Elo ladder

The learned checkpoint plugs into the `Agent` Protocol and plays seeded
matches against the **fixed ladder** {`random_legal`, `greedy`, `matrix_1ply`}
(`agents/`, `solver/agent.py`). Report relative Elo with bootstrap CIs over a
seeded match schedule, per checkpoint, to give a learning *curve*. **Budget
caveat:** `matrix_1ply` is 15–55× slower than the stdlib agents
(PROJECT_STATUS, Phase 11b) because it LP-solves both colors' supports every
phase — its match counts are sized down accordingly, and it is the ladder's
throughput constraint, not the learned net.

### 6.2 Primary strength gate (13b DoD)

The 13b definition-of-done strength bar: the final checkpoint defeats
`matrix_1ply` in a seeded match at **one-sided exact binomial $p<0.01$** — the
same test Phase 10 used for `matrix_1ply` vs `random_legal`, reused for
continuity. This is a strength check, not a balance claim.

### 6.3 Exploitability / NashConv (the metric with standing)

Per §1.3, strength against a fixed opponent is a proxy; exploitability is the
metric that measures distance from equilibrium.

- **Exact restricted-support NashConv**, on the Phase 10 M5 fixtures. On the
  χ-symmetric fixtures (standard start + the two hand-built midgames), with
  χ-closed restricted supports, `solver/lp.py` gives the exact matrix-game
  value and both players' equilibrium strategies. Against that exact value we
  compute the learned policy's NashConv on the restricted stage game — an
  **exact** number (not a bound) on a small, controlled set. M5's proven
  $\operatorname{val}=0$ on these fixtures gives a free sanity anchor: a
  well-trained value head should read ~0 there.
- **Approximate exploitability**, game-wide, via **best-response training**
  against frozen checkpoints: train a from-scratch best-responder against a
  frozen learned agent and report its win-rate as a **lower bound** on
  exploitability (labelled as such — a learned best response under-approximates
  the true best response). Report the trend across ≥3 checkpoints; the 13b DoD
  asks for a *decreasing* estimate.

### 6.4 Learning-diagnostic figures

- **Stage-policy entropy over training.** Does self-play discover genuinely
  *mixed* play, as §1.2 says it must? Track the mean entropy of the search
  strategies $\bar\sigma_\omega$ over training. A collapse toward pure
  strategies in Matching-Pennies-like positions would be a red flag; healthy
  training keeps positive entropy where the game demands mixing.
- **Value-head calibration.** Bin predicted $v$ against Monte-Carlo returns
  from that state under current self-play; plot calibration curves per
  checkpoint. Miscalibration localizes value-head problems independently of
  the policy.
- **Color-symmetry spot-check.** Cheap M3 hygiene reused from Phase 11b: among
  decisive self-play games, White-win fraction with a two-sided exact binomial
  against $\tfrac12$. Since the operator is provably χ-symmetric (M3), any
  rejection localizes to a learned-agent asymmetry or a bug, never the rules.

---

## 7. Variant hook: imperfect information

The same **trunk and plane layout** and the same **Φ** are the intended
substrate for the hidden-information variant (spec Ch. 11 / §11; the "more
interesting" game the designer flagged, and the R-NaD/CFR milestone of the
v1.0 Phase-9 sketch). Stating precisely what changes and what does not —
because the addendum's shorthand "encoding does not change" is *not quite
right* and the difference matters:

**What does not change.**
- **$\Phi$** — the transition operator is identical (spec §11: the hidden
  variant "changes the solution concept … but not the transition operator").
- **The plane/scalar *layout* and the residual trunk** — same shapes, same
  architecture; the net is re-usable.
- The autoregressive factored action head (§3.3): the action *space* is
  unchanged; only what conditions the policy differs.

**What does change.**
- **The observation *function* (not merely the layout).** Under perfect
  information the encoding is player-independent (§3.1). Under hidden
  information it is **not**: standing (unfired) reservations, cancellations,
  and fizzled moves are **private** (spec §11 line 326); a reservation becomes
  public only *at the instant it fires*. So the two colors get **different**
  tensors — each sees its own reservations but not the opponent's standing
  ones — and the §3.2 pairing channel is populated only from the observer's
  own reservations plus fired-this-phase public ones. The `set_from(state,
  player)` signature that Phase 12 currently ignores its `player` argument
  becomes genuinely player-dependent. This is the single most important code
  consequence and the reason "the encoding does not change" is imprecise: the
  *tensor shape* is reused, the *function that fills it* is replaced.
- **The solution concept and the search.** The game becomes a finite
  zero-sum **extensive-form game of imperfect information** with perfect
  recall (spec §11 line 322); states are replaced by **information sets**,
  and the target shifts from stage-matrix minimax to the **CFR family** —
  MCCFR at scale, or **R-NaD** (regularized Nash dynamics, the DeepNash line)
  as the neural realization, with **belief-state / public-belief** search
  replacing the perfect-information tree. The regret-matching machinery of §2
  is the right primitive to carry forward — CFR *is* counterfactual regret
  minimization — but it runs over information sets, not states.

The upshot: Phase 13b's LIGHT system is a strict stepping-stone. The trunk,
the encoding *shapes*, the action head, and Φ transfer to the hidden variant
unchanged; the observation function and the search/solution concept are the
two seams where the variant is cut. Neither seam touches the rules engine.

---

## Appendix A — Profiling methodology (regeneration)

Numbers in §4 were produced by two throwaway scripts (not committed; kept in
the session scratchpad) run against the committed engine on the machine of
§4.1. Method:

- **Representative states.** 8 self-play games (mixed
  `greedy`/`random_legal` pairings), `max_phases=80`, capturing every
  pre-phase `state_before` via `play_one_game`'s `on_phase_result` hook;
  shuffled (seed 0) and truncated to 300 (60 for the costlier enumeration
  decomposition).
- **Φ throughput.** Programs pre-selected with `random_legal_program` (to
  isolate Φ from candidate generation), 30-state warmup, then timed
  `phi(...)` calls; reported as phases/s and ms/phase.
- **Enumeration decomposition.** Per state, timed: (a) full
  `enumerate_legal_programs` both colors; (b) single-action pool build; (c)
  $O(\text{pool})$ slot-1 legality mask; (d) $O(\text{pool})$ slot-2 mask
  given a fixed slot-1. (a) is the adapter path; (b)+(c)+(d) is what the
  autoregressive search node actually pays.
- **Network.** A representative residual net ($B{=}6$, $F{=}64$,
  `(20,8,8)`–`(21,8,8)` input, ~8 834 policy logits) on MPS: forward latency
  at batch {1, 8, 64, 256} with `torch.mps.synchronize()` around a warmed
  timing loop; forward+backward with an SGD step and a dummy MSE loss at batch
  {64, 256}.

Raw output is reproduced inline in §4.2. The scripts depend only on the
committed engine plus `torch` (added to the dev venv for profiling); nothing
in `learn/` exists yet, consistent with the 13a "no code" DoD.

---

## Appendix B — Decisions folded into this design

| # | Decision | Where |
|---|---|---|
| D1 | Policy head is a **fixed factored grid**, autoregressive over the 2 slots (not a pointer-over-candidates head). | §3.3 |
| D2 | Profiling is **measured now** (torch added), not deferred; Φ, enumeration decomposition, and net all benchmarked on-device. | §4, App. A |
| D3 | Learned action space is the **full $L(s,\pi)$ including `Cancel`**; 13b extends the exhaustive enumeration to emit it. | §3.3 |
| D4 | `Cancel` is **protégé-square-keyed with an oldest-valid tie-break** on the grid. | §3.3 |
| D5 | Input encoding gains a **reservation-pairing channel** (17→21 planes); edits the Phase 12 source-of-truth encoding in lockstep at 13b. | §3.2 |
| D6 | Measured **bottleneck is move-gen, not Φ** (correcting the addendum); it re-aims the Rust annex and justifies the autoregressive head on throughput grounds. | §4.4, §5 |
