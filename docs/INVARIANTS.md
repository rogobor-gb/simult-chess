# Simultaneous Chess — Invariant Checklist, v1.1

*Companion to `simultaneous_chess_spec_v1.md` (hereafter **spec**). This document is
the ground-truth specification of the **validation harness**: the set of predicates
that must never be falsified by a correct implementation of the transition operator
$\Phi$. It is written to be implemented directly — every invariant states a formal
predicate, its spec anchor, the point in the pipeline at which it is checked, a
severity, and (where relevant) the `RuleSet` parameter on which its truth depends.*

*Licensed under CC BY 4.0.*

**v1.1 (2026-07-14).** Rulings record (`docs/DEVELOPMENT_addendum_v1.1.md` §A, binding):
L3's actor extraction for `Castle` now yields `{king, flank rook}` (A3, was king-only);
`cancellation_enabled` and `pawn_same_square_fizzle_scope` move from `[OPEN]`/`[C,
confirm]` to resolved `[C]` in §8 (A1, A2 — values unchanged, epistemic status only);
M5's permanent scope is clarified to symmetric fixtures with restricted action supports
(A8); M4's `[K]` branching is confirmed ahead of its Phase 11 exercise (A4).

---

## 0. How to read and run this checklist

### 0.1 Anatomy of an invariant

Each entry has the form

> **ID — Name.** *(severity, check-point, [K] if convention-coupled)*
> Formal statement.
> *Ref.* spec section. *Check.* how the harness verifies it.

### 0.2 Check-points

| Point | Meaning |
|---|---|
| `PRE` | Asserted on $(s,\pi_\mathrm{W},\pi_\mathrm{B})$ **before** $\Phi$ runs (a legality precondition; failure ⇒ the move generator emitted an illegal program). |
| `STATE` | Asserted on every state $s$ and every successor $s'=\Phi(s,\cdot,\cdot)$ (well-formedness). |
| `TRACE` | Asserted on the internal resolution trace of a single $\Phi$ call (fizzle set, matching, precedence DAG, fired reservations, closure). |
| `META` | A **metamorphic / property-based** test: run $\Phi$ on randomized-but-equivalent inputs (reordered internals, mirrored board) and compare outputs. Run in the self-play harness and in CI, not on every production call. |

### 0.3 Severity

| Level | Class | Meaning |
|---|---|---|
| **S0** | Soundness | $\Phi$ is not a well-defined, color-symmetric, conservative operator. A single S0 violation invalidates the engine. |
| **S1** | Structural | A state is malformed (occupancy, cooldown, reservation, or bookkeeping integrity). |
| **S2** | Semantic fidelity | A specific spec rule (fizzle, intermezzo, cooldown, promotion, castling) is misimplemented. |
| **S3** | Convention-tied | Correctness is contingent on a `[K]` `RuleSet` choice; the invariant and the parameter must move together. |

### 0.4 Harness modes

- **strict** — on violation, raise and emit the repro dump (§9), halting the game. Default in unit tests and CI.
- **lenient** — on violation, log the repro dump, tag the phase, and continue. Default in the headless self-play fuzzer, so a single run harvests *all* violations rather than dying on the first.

`[K]` marks any invariant whose truth is coupled to a `RuleSet` parameter (§8). The
checker reads the parameter; it never hardcodes the constant. Changing the parameter
for a variant **must** be accompanied by an edit to every `[K]` invariant listed
against it in §8 — this is the mechanism that keeps variants honest.

---

## 1. Notation (defer to spec §1; fixed additions below)

State $s=(\beta,\,C,\,R_\mathrm{W},\,R_\mathrm{B},\,\eta)$ with occupancy
$\beta:\mathcal P^{\text{live}}\hookrightarrow\mathcal S$, cooldown set $C$,
age-ordered reservation lists $R_\omega$, bookkeeping
$\eta=(\text{castling rights},\text{repetition ledger},\text{no-progress counter }\nu,\text{phase index }t)$.
Programs $\pi_\omega=(\alpha_1,\dots,\alpha_k)$; declared moves $M=M_\mathrm{W}\sqcup M_\mathrm{B}$;
executing moves $M^\ast$ (spec §3). Swept set $\sigma(\tau)=\{q_1,\dots,q_\ell\}$
(origin **excluded**), directed edges $\varepsilon(\tau)$ (spec §1.3).

**Actor(s) of an action** (used by `L-*`). For $\mathrm{Move}(p,\tau)$ the actor is
the moved token $p$; for $\mathrm{Reserve}(D,Q)$ the actor is the **defender** $D$;
$\mathrm{Cancel}$ has no board actor. **[C, v1.1, A3]** For $\mathrm{Castle}$ the
actor **set** is **{king, flank rook}** — both synchronized sub-movers of spec §6.6
are actors, not the king alone (v1.0 treated the rook as a mere co-mover; this was a
gap, since it let a program castle and separately move the same rook without
tripping L3). A **protégé $Q$ is not an actor** — hence a protégé may be cooled or
moving (spec §4.3, §7: a cooled piece may be defended).

**The color-swap involution $\chi$** (central to M3). Define the vertical rank
reflection $\mu:(c,r)\mapsto(c,7-r)$ (files fixed) composed with color inversion.
$\chi$ acts on a state by: relabelling each live token $p$ to a token $\bar p$ with
$\mathrm{col}(\bar p)=-\mathrm{col}(p)$, $\mathrm{typ}(\bar p)=\mathrm{typ}(p)$,
$\beta(\bar p)=\mu(\beta(p))$; mapping $C\mapsto\{\bar p:p\in C\}$; mapping each
reservation $(D,Q,a)\mapsto(\bar D,\bar Q,a)$ into the opposite player's list; and
reflecting castling rights (kingside↦kingside, queenside↦queenside — files are
preserved, so the two flanks are **not** swapped). $\chi$ is an involution,
$\chi^2=\mathrm{id}$, and maps a White pawn's forward push $\Delta r=+1$ to a Black
pawn's forward push $\Delta r=-1$, i.e. it respects pawn directionality. $\chi$ acts
on a program by relabelling every actor/trajectory under $\mu$ and color inversion.

---

## 2. Index

| ID | Name | Sev | Check | K |
|---|---|---|---|---|
| WF1 | Occupancy injectivity | S1 | STATE | |
| WF2 | Type/color domain, type-constancy | S1 | STATE | |
| WF3 | Cooldown membership | S1 | STATE | |
| WF4 | King count | S1 | STATE | |
| WF5 | Reservation list well-order | S1 | STATE | |
| WF6 | Reservation referential integrity | S1 | STATE | |
| WF7 | Bookkeeping ranges & monotonicity | S1 | STATE | K |
| L1 | Budget | S2 | PRE | K |
| L2 | Mandatory displacement | S2 | PRE | |
| L3 | Distinct actors | S2 | PRE | |
| L4 | Cooldown respected | S2 | PRE | |
| L5 | Own-consistency ⇒ bipartite conflict | S0 | PRE | |
| L6 | Geometric legality | S2 | PRE | |
| R1 | Fizzle F1 (vacated pawn capture) | S2 | TRACE | |
| R2 | Fizzle F2 (pawn convergence) | S2 | TRACE | K |
| R3 | Edge-conflict mutual annihilation | S2 | TRACE | |
| R4 | Annihilation matching by rank | S2 | TRACE | K |
| R5 | One surviving mover per destination | S0 | TRACE | |
| R6 | Vacated-square theorem | S0 | TRACE | |
| R7 | Intermezzo unconditional precedence | S2 | TRACE | K |
| R8 | Oldest-valid reservation fires | S2 | TRACE | |
| R9 | Defender fires ≤ once per phase | S2 | TRACE | |
| R10 | Mover-as-defender forbidden | S2 | TRACE | |
| R11 | Mutual-defense SCC → base semantics | S2 | TRACE | |
| R12 | Cascade termination & token descent | S0 | TRACE | |
| R13 | Cooldown update | S2 | TRACE | K |
| R14 | Promotion | S2 | TRACE | |
| R15 | Castling resolution | S2 | TRACE | |
| R16 | Fizzled mover inertness | S2 | TRACE | |
| R17 | Cancellation at closure | S2 | TRACE | K |
| R18 | Token conservation | S0 | TRACE | |
| T1 | King-capture terminal / synchronous draw | S2 | STATE | |
| T2 | No check/checkmate | S2 | PRE/STATE | |
| T3 | Threefold repetition of $(\beta,C)$ | S2 | STATE | |
| T4 | No-progress horizon | S2 | STATE | K |
| M1 | Purity / determinism | S0 | META | |
| M2 | Internal-order independence | S0 | META | |
| M3 | Color-swap equivariance | S0 | META | |
| M4 | Reservation-order independence of defense | S0 | META | K |
| M5 | Symmetric-position value antisymmetry | S1 | META | |

---

## 3. State well-formedness `WF-*` (check-point `STATE`)

**WF1 — Occupancy injectivity.** *(S1)*
$\beta$ is injective: $\forall p\ne p'\in\mathcal P^{\text{live}},\ \beta(p)\ne\beta(p')$.
No square holds two live tokens.
*Ref.* §1.2. *Check.* assert $|\operatorname{image}(\beta)|=|\mathcal P^{\text{live}}|$.

**WF2 — Type/color domain and type-constancy.** *(S1)*
$\forall p\in\mathcal P^{\text{live}}:\ \mathrm{col}(p)\in\{\mathrm W,\mathrm B\},\ \mathrm{typ}(p)\in\mathcal T$.
Between $s$ and $s'$, $\mathrm{col}$ is immutable and $\mathrm{typ}(p)$ changes **only**
if $p$ promoted this phase (§6.5); no other type mutation occurs.
*Ref.* §1.1, §6.5. *Check.* diff $s\to s'$ on `typ`; every change must be a recorded promotion.

**WF3 — Cooldown membership.** *(S1)*
$C\subseteq\mathcal P^{\text{live}}$ and $C\cap(\{p:\mathrm{typ}(p)=\mathsf p\}\cup\{p:\mathrm{typ}(p)=\mathsf k\})=\varnothing$.
*Ref.* §1.2, §7. *Check.* set membership + type filter.

**WF4 — King count.** *(S1)*
In every **non-terminal** reachable state, $|\{p:\mathrm{typ}(p)=\mathsf k,\ \mathrm{col}(p)=\omega\}|=1$
for each $\omega\in\{\mathrm W,\mathrm B\}$; universally $\le 1$.
*Ref.* §10. *Check.* count kings per color; if either is $0$ the state must be flagged terminal (see T1).

**WF5 — Reservation list well-order.** *(S1)*
Each $R_\omega$ is strictly ordered by age stamp $a=(\text{phase},\text{intra-index})$;
stamps are unique across $R_\mathrm{W}\cup R_\mathrm{B}$; the list order equals the age order.
*Ref.* §1.2, §4.3. *Check.* stamps strictly increasing along the list; global uniqueness.

**WF6 — Reservation referential integrity.** *(S1)*
$\forall(D,Q,a)\in R_\omega$: $D,Q\in\mathcal P^{\text{live}}$; $\mathrm{col}(D)=\mathrm{col}(Q)=\omega$;
$D\ne Q$; $\beta(D)$ and $\beta(Q)$ are the squares at which the reservation was
registered (neither has displaced since). *(In v1 the protégé is always an owned live
token; "trap reservations" against arbitrary arrivals are a deferred variant, §13.5.)*
This asserts standing integrity only; **path clarity is re-checked at fire time** (R7/R8),
not here, because interposition may block a ranged recapture (Prop. 12.1b).
*Ref.* §4.3, §6.7. *Check.* per reservation: liveness, color, distinctness, square-stability.

**WF7 — Bookkeeping ranges & monotonicity.** *(S1, [K])*
Castling rights are monotone non-increasing over a game (never regained); the
no-progress counter $\nu\in[0,H]$; the phase index increments by exactly $1$ per phase;
the repetition ledger is keyed on the **public** position $(\beta,C)$ only.
*Ref.* §6.7, §10. *Check.* compare $\eta\to\eta'$. **[K]** upper bound $H$ = `RuleSet.H`.

---

## 4. Declaration-legality preconditions `L-*` (check-point `PRE`)

These encode $L(s,\pi_\omega)$ (§4.4). A failure means the agent/move-generator
produced an illegal program; $\Phi$ must `assert L` before resolving.

**L1 — Budget.** *(S2, [K])* $1\le|\pi_\omega|\le N$. *Ref.* §4.4.1. **[K]** $N$ = `RuleSet.N`.

**L2 — Mandatory displacement.** *(S2)*
At least one action is a $\mathrm{Move}$ or $\mathrm{Castle}$, **unless** $\omega$ has
no geometrically legal displacement at all on $\beta$, in which case a
reservation-only or empty program is admitted for that phase (degenerate exception,
§4.4.2). *Ref.* §4.4.2, §9, §11.3. *Check.* count movers; if $0$, verify the
no-legal-displacement predicate holds.

**L3 — Distinct actors.** *(S2)*
Each token is the actor of at most one action in $\pi_\omega$ (a token may *move* while
*another* token defends it; a token may not move twice, nor both move and serve as the
**fired** defender). **[C, v1.1, A3]** A `Castle` action contributes **two** actors,
`{king, flank_rook}` (§1's actor definition) — so a program that castles and also
declares a separate action for that same rook (a second `Move`, or naming it as a
`Reserve` defender) fails L3. *Ref.* §4.4.3, §4.3, spec §4.1/§6.6 (v1.1). *Check.*
actor multiset — built from *every* action's full actor set, including both of
`Castle`'s — has no repeats.

**L4 — Cooldown respected.** *(S2)*
No **actor** lies in $C$: a cooled token can neither move, castle, declare a
reservation (as defender), nor fire one. A cooled token **may** be a protégé.
*Ref.* §4.4.4, §7. *Check.* actor set $\cap\,C=\varnothing$.

**L5 — Own-consistency ⇒ bipartite conflict.** *(S0)*
The player's own executing moves are pairwise non-conflicting under (V)/(E) (§6.1),
and none lands on a square occupied by that player's own **stationary** piece.
*Consequence to assert downstream:* every (V)/(E) conflict in $M^\ast$ is
cross-color (bipartite) — the precondition for §6.3 being a bipartite matching.
*Ref.* §4.4.5, §6.1. *Check.* intra-color conflict count $=0$; then verify the conflict
graph on $M^\ast$ has no monochromatic edge (this is the S0 hook).

**L6 — Geometric legality.** *(S2)*
Every $\mathrm{Move}$ satisfies §4.2 (pattern, interior empty on $\beta$, destination
empty for non-capture / enemy-occupied for capture; pawn push vs. diagonal-capture
distinction); every $\mathrm{Reserve}$ satisfies §4.3 (admissible capturing pattern
$\beta(D)\to\beta(Q)$ at declaration); every $\mathrm{Castle}$ satisfies §6.6; every
$\mathrm{Cancel}(\rho)$ names an existing $\rho\in R_\omega$.
*Ref.* §4.2, §4.3, §6.6. *Check.* per action, delegate geometry to the oracle on
$\beta$ only (no look-ahead).

---

## 5. Resolution semantics `R-*` (check-point `TRACE`)

**R1 — Fizzle F1 (vacated pawn capture).** *(S2)*
A pawn diagonal capture $m$ onto $x=\mathrm{dst}(m)$ with declaration-time occupant
$V=\beta^{-1}(x)$ fizzles **iff** $V$ executes a move (vacates $x$), whether $V$
completes elsewhere or dies en route. No executing pawn-capture removes a token that
itself moved. *Ref.* §6.2 (F1), Lemma 6.2. *Check.* backward induction over the
functional fizzle digraph; assert out-degree $\le 1$ and that all cycles are the
2-cycles exported to (E) (R3).

**R2 — Fizzle F2 (pawn convergence).** *(S2, [K])*
Two **opposing pawn** moves declaring the same destination both fizzle; by the
realizability lemma this occurs only as opposing **pushes** onto a shared empty
square. After resolution both pawns remain on their origins, with no capture and no
cooldown. **Scope [K, resolved A2]:** the same-square fizzle triggers only when *both*
movers are pawns; a mixed pawn/non-pawn convergence onto one square is ordinary
(V)-annihilation (R4), **not** a fizzle. *Ref.* §6.2 (F2), §6.5. **[K]** =
`RuleSet.pawn_same_square_fizzle_scope` (v1: `both_pawns`, canonical since the 2026-07-14
ruling — see changelog; the field's value was already the default, only its
epistemic status changed from `[C, confirm]` to resolved `[C]`).

**R3 — Edge-conflict mutual annihilation.** *(S2)*
Any opposing pair with a shared edge in opposite orientation
($(u,v)\in\varepsilon(m_1),(v,u)\in\varepsilon(m_2)$) both annihilate. Assert that
(E) is **not** subsumed by (V): the swept sets of a head-on swap are disjoint, so
(V) alone would miss it. Head-on adjacent pawns (pushes or mutual diagonal captures)
resolve here, not as F2. *Ref.* §6.1, §6.5. *Check.* both die; verify
$\sigma(m_1)\cap\sigma(m_2)=\varnothing$ for the swap case.

**R4 — Annihilation matching by rank.** *(S2, [K])*
On the bipartite conflict graph over $M^\ast$, process candidate edges in increasing
rank $r(W_i,B_j)=(\max(i,j),\min(i,j))$ lexicographically; on reaching a live,
unmatched pair, annihilate both. A mover none of whose conflict-partners survive to
meet it completes unharmed. *Ref.* §6.3. *Check.* greedy match reproduces the canonical
surviving set. **[K]** = `RuleSet.annihilation_reading` (v1: `B` = declaration-priority
pairing; alt: timed model, §13).

**R5 — One surviving mover per destination.** *(S0)*
Entering Stage B, at most one surviving mover targets any given square.
*Ref.* §6.4 (opening). *Check.* group survivors by destination; assert each group size
$\le 1$ (a size-$2$ group is an unresolved (V)-conflict ⇒ upstream bug).

**R6 — Vacated-square theorem.** *(S0)*
No token is captured on a square it vacated. Formally: if $\mathrm{tok}(m)$ executes,
no capture event removes $\mathrm{tok}(m)$ at $\mathrm{org}(m)$; and a mover sweeping a
square vacated by an enemy passes unharmed (origin exclusion from $\sigma$).
*Ref.* §6.3 Remark. *Check.* construct a slider-through-vacated-square fixture; assert
survival. This is a corollary of the origin exclusion — a violation means $\sigma$
wrongly includes origins.

**R7 — Intermezzo unconditional precedence.** *(S2, [K])*
When a stationary $V$ holding a valid reservation $(D,V,a)$ is captured by $\alpha$,
$D$ recaptures on $x=\beta(V)$, and any **same-phase** capture of $D$ at $\beta(D)$
**misses** (the fired recapture vacates $\beta(D)$ first), *independent of declaration
order*. A **fired defender is never captured at its origin in the same phase**. A
reservation is valid at fire time iff $D$ is alive, on $\beta(D)$, $\notin C$, and the
recapture trajectory is unobstructed on the current board.
*Ref.* §6.4, Lemma 6.4b, §12.1. *Check.* on the precedence DAG, assert: (i) every fired
defender's origin is vacated before any capture targeting it resolves; (ii) the
terminal survivor set is invariant to the attacker's action ordering (delegate to M4).
**[K]** = `RuleSet.intermezzo_reading` (v1: `(ii)` = unconditional defensive precedence;
alt: `(i)` attacker-sequenced, §13.4).

**R8 — Oldest-valid reservation fires.** *(S2)*
On a trigger, exactly the **age-minimal valid** reservation on the protégé fires; all
other reservations on that protégé expire for that trigger (R-multi-in).
*Ref.* §4.3, §12. *Check.* selected reservation $=\arg\min_a\{$valid reservations on $V\}$;
at most one fires per trigger.

**R9 — Defender fires ≤ once per phase.** *(S2)*
A single defender fires at most once per phase; firing displaces it and self-invalidates
its remaining reservations (R-multi-out).
*Ref.* §4.3. *Check.* per token, count fires $\le 1$; post-fire, its other reservations
are invalidated at closure (R13/§6.7).

**R10 — Mover-as-defender forbidden.** *(S2)*
No reservation **fired** this phase has a defender $D$ that also executed a $\mathrm{Move}$/$\mathrm{Castle}$
this phase. (Automatic: a displaced $D$ has left $\beta(D)$, so its reservation is
invalid at fire time.) *Ref.* §4.3. *Check.* fired-defender set $\cap$ mover set $=\varnothing$.

**R11 — Mutual-defense SCC → base semantics.** *(S2)*
If the precedence relation $\prec$ contains a cycle (e.g. $P$ defends $Q$, $Q$ defends
$P$, both attacked), resolve each strongly-connected component under **base semantics**:
no recapture in the SCC fires; all its attacked defenders are actually captured.
*Ref.* §6.4 [C]. *Check.* detect SCCs of $\prec$; within any non-trivial SCC assert
zero fired recaptures and full capture of its attacked members.

**R12 — Cascade termination & token descent.** *(S0)*
The capture/recapture cascade reaches quiescence in $<|\mathcal P^{\text{live}}|$ steps;
each fired recapture strictly decreases the live-token count and never resurrects a
token. *Ref.* Lemma 6.4c. *Check.* bound the cascade loop by $|\mathcal P^{\text{live}}|$;
assert strict monotone descent; any overrun is an S0 non-termination bug.

**R13 — Cooldown update.** *(S2, [K])*
$C' = \{\text{tokens that displaced this phase}\}\setminus(\text{pawns}\cup\text{kings})$,
**including** recapturers and promoted pieces. *Composition invariant (§7 near-theorem):*
a token holding a valid reservation has not displaced ⇒ is never in $C'$ ⇒ is always
fire-eligible; conversely a token in $C$ can never fire (it displaced ⇒ left origin ⇒
invalid). *Ref.* §6.7, §7. *Check.* recompute $C'$ from the displacement set; assert the
composition (no token is simultaneously in $C'$ and a fired defender).
**[K]** whether a **recapture** contributes to $C'$ = `RuleSet.recapture_cooldown`
(v1: `on`).

**R14 — Promotion.** *(S2)*
A pawn reaching the last rank becomes the declaration-time chosen type and enters $C'$
for one phase; this is the sole permitted `typ` mutation (cf. WF2). A pawn promoted in
one stage may be the object of a recapture in a later stage.
*Ref.* §6.5, §7. *Check.* every promoted token $\in C'$; type change recorded.

**R15 — Castling resolution.** *(S2)*
$\mathrm{Castle}$ consumes one slot and contributes two synchronized sub-trajectories,
each assessed under (V)/(E). A hit on the king component ends the game; a hit on the
rook component removes the rook while the king completes. Post-resolution: the **rook**
$\in C'$, the **king** $\notin C'$; the relevant castling right is consumed;
"through check" does not apply (no check exists).
*Ref.* §6.6. *Check.* king cooldown-exempt; rook cooled; rights decremented.

**R16 — Fizzled mover inertness.** *(S2)*
A fizzled move (F1 or F2) leaves its token on the origin, contributes **no** board
delta, adds the token to **neither** the displaced set nor $C'$, yet **consumes** its
action slot (slots are spent at declaration). *Ref.* §6.2. *Check.* fizzled tokens
unmoved and cooldown-free; slot budget still debited.

**R17 — Cancellation at closure.** *(S2, [K])*
$\mathrm{Cancel}(\rho)$ removes $\rho$ from $R_\omega$ at phase closure (§6.7), is
**free and slot-less**, and is a blind pre-commitment (decided before observing whether
the opponent would trigger it). *Ref.* §9, §6.7. *Check.* cancelled reservations absent
from $R'_\omega$; no slot debited. **[K, resolved A1]** = `RuleSet.cancellation_enabled`
(v1: `on`, canonical since the 2026-07-14 ruling retaining cancellation — see
changelog; if disabled, this invariant is void and $\mathrm{Cancel}$ must be rejected
at L6).

**R18 — Token conservation.** *(S0)*
Across a phase, $\mathcal P^{\text{live}}$ is non-increasing; no dead token is
resurrected; the only identity-preserving mutation is promotion (type, not identity).
*Ref.* §3, §6. *Check.* $\mathcal P^{\text{live}}(s')\subseteq\mathcal P^{\text{live}}(s)$;
removed tokens stay removed for the remainder of the game.

---

## 6. Termination & bookkeeping `T-*`

**T1 — King-capture terminal / synchronous draw.** *(S2, `STATE`)*
Removal of a king in **any** stage of a phase is terminal: the owner loses ($u=\pm1$).
Both kings removed in the **same** phase (any stages) is a draw ($u=0$).
*Ref.* §10. *Check.* on king removal, assign $u$; if both removed same phase, $u=0$.

**T2 — No check/checkmate.** *(S2, `PRE`/`STATE`)*
No legality rule references check, checkmate, or "leaving the king en prise." A king
may legally move into an attacked square; a king–king trajectory conflict is a draw
under (V)/(E). *Ref.* §10. *Check.* assert the legality predicate never consults an
attack map for king safety; king moves into attacked squares are admitted at L6.

**T3 — Threefold repetition of $(\beta,C)$.** *(S2, `STATE`)*
Threefold repetition of the **public** position $(\beta,C)$ (not reservation contents)
is a draw. *Ref.* §10. *Check.* ledger keyed on $(\beta,C)$; draw at third occurrence.

**T4 — No-progress horizon.** *(S2, `STATE`, [K])*
$H$ phases without a capture or a pawn displacement is a draw; $\nu$ resets to $0$ on
any capture or pawn displacement and otherwise increments by $1$; draw at $\nu=H$.
*Ref.* §6.7, §10. *Check.* reset/increment logic; **[K]** $H$ = `RuleSet.H` (v1: $50$).

---

## 7. Metamorphic / global properties `M-*` (check-point `META`)

These are the highest-value fuzzing targets: they encode the operator's soundness
lemmas as executable relations over *pairs* of $\Phi$-evaluations.

**M1 — Purity / determinism.** *(S0)*
$\Phi$ is a pure function: repeated evaluation of the same $(s,\pi_\mathrm{W},\pi_\mathrm{B})$
yields bit-identical $s'$ and identical trace, with no dependence on wall-clock, global
state, or unseeded randomness. *Ref.* §5, §14. *Check.* evaluate twice; assert equality
of $s'$ and canonicalized trace.

**M2 — Internal-order independence.** *(S0)*
The output is invariant to the internal processing order at each ordered stage:
(a) the backward-induction order of fizzle resolution (Lemma 6.2);
(b) the processing order of **equal-rank** annihilation edges (Lemma 6.3a);
(c) any topological order of the precedence DAG $\prec$ (Lemma 6.4a, Newman).
*Ref.* Lemmas 6.2 / 6.3a / 6.4a. *Check.* generate a random legal
$(s,\pi_\mathrm{W},\pi_\mathrm{B})$; run $\Phi$ under $k$ random admissible internal
orderings; assert all $s'$ agree. **A discrepancy localizes the non-determinacy to the
disagreeing stage.**

**M3 — Color-swap equivariance.** *(S0)*
$\chi\big(\Phi(s,\pi_\mathrm{W},\pi_\mathrm{B})\big)=\Phi\big(\chi(s),\,\chi(\pi_\mathrm{B}),\,\chi(\pi_\mathrm{W})\big)$
— note the program swap: after $\chi$, White's program becomes Black's. This is the
formal statement that priority (§6.3) and defensive precedence (§6.4) confer **no
first-mover advantage**. *Ref.* Lemmas 6.3b / 6.4b, spec §0. *Check.* mirror state and
programs under $\chi$, resolve, and assert the result equals the $\chi$-image of the
unmirrored result. **This is the single most important non-trivial test in the suite.**

**M4 — Reservation-order independence of defense.** *(S0, [K])*
The defensive outcome (who survives an attack on a defended structure) is invariant to
the **attacker's** declaration order of its captures — the operational essence of
Reading (ii). *Ref.* §6.4 worked example, §12.1. *Check.* fix a defended fixture; permute
the attacker's intra-program order over all $k!$ orderings; assert identical survivors.
**[K]** contingent on `RuleSet.intermezzo_reading = (ii)`; under Reading `(i)` this
invariant is *deliberately false* and must be replaced by the order-dependent
specification of §13.4. **Confirmed ahead of exercise (A4):** this branch is the
harness's contract for Phase 11a, which is the first place `intermezzo_reading = "i"`
is actually driven through the registry and this invariant's `(i)`-branch fires for
real; nothing here changes as a result of that exercise, since the branching
language was already the intended reading.

**M5 — Symmetric-position value antisymmetry.** *(S1, requires solver layer)*
For a color-symmetric position $s=\chi(s)$, the one-phase stage game is symmetric, hence
its value is $0$ and the payoff matrix satisfies $u^\ast(s,\pi,\pi')=-u^\ast(s,\chi\pi',\chi\pi)$.
*Ref.* §8.1–8.2. *Check.* **only once the stage-matrix/value layer exists**; on symmetric
fixtures assert matrix antisymmetry under $\chi$ and value $0$. Until then, keep as a
documented target, not an active assertion. **Scope, clarified (A8):** this is a
*permanent* scope restriction, not a Phase 10 bootstrapping stopgap — M5 runs only on
$\chi$-symmetric fixtures ($s=\chi(s)$: the standard start plus hand-built midgame
positions) with $\chi$-closed action supports, never on arbitrary states drawn from a
self-play sweep. Symmetry of the *fixture* is what makes the value-$0$ claim provable
at all; a generic sweep state has no such guarantee.

---

## 8. Convention-dependence map (the variant safety net)

Every `[K]` invariant is coupled to a `RuleSet` parameter. Changing a parameter for a
variant **requires** editing the coupled invariants in lockstep; the harness reads the
parameter, never a literal.

| `RuleSet` parameter | v1 value | Coupled invariants | Effect if changed |
|---|---|---|---|
| `N` (actions/phase) | `2` | L1 | Budget bound; branching (§8.4). |
| `H` (no-progress horizon) | `50` | WF7, T4 | Draw horizon; convertibility (§8.3). |
| `recapture_cooldown` | `on` | R13 | Whether recapturers enter $C'$. |
| `cancellation_enabled` | `on` | R17, (L6) | If `off`: reject `Cancel` at L6; void R17; defense irrevocable. |
| `pawn_same_square_fizzle_scope` | `both_pawns` | R2 | Mixed convergence becomes (V)-annihilation vs. fizzle. |
| `annihilation_reading` | `B` (priority pairing) | R4 | Alt: timed one-tick model (§13.2). |
| `intermezzo_reading` | `(ii)` (unconditional) | R7, M4 | Alt: `(i)` attacker-sequenced (§13.4) — flips M4 from *true* to *deliberately order-dependent*. |

**Both spec items resolved 2026-07-14 (see changelog; not `[OPEN]` any more):**
`cancellation_enabled` (spec §9, retained — A1) and `pawn_same_square_fizzle_scope`
(spec §13, `both_pawns` confirmed — A2). Both fields' *values* were already the v1
defaults before the ruling; only their epistemic status moved from `[OPEN]`/`[C,
confirm]` to resolved `[C]`. They remain explicit `RuleSet` fields regardless, so a
future re-ruling is still a one-line change, not a code hunt.

---

## 9. Repro-dump schema (emitted on any violation)

On a fired invariant the harness serializes a self-contained, replayable record:

```
{
  "invariant_id":   "<e.g. M3>",
  "severity":       "S0|S1|S2|S3",
  "check_point":    "PRE|STATE|TRACE|META",
  "phase_index":    <t>,
  "rng_seed":       <int>,                 # reproduces agent sampling
  "ruleset":        { ... full RuleSet ... },
  "state_pre":      <s serialized>,        # β, C, R_W, R_B, η
  "program_W":      <π_W>,
  "program_B":      <π_B>,
  "trace": {                               # populated for TRACE/META
      "fizzled":        [...],             # F1/F2 outcomes
      "M_exec":         [...],
      "annihilated":    [...],             # matched pairs with ranks
      "precedence_dag": <edges of ≺>,
      "fired":          [...],             # reservations that fired (D,Q,a)
      "cooldown_next":  [...]              # C'
  },
  "state_post":     <s' serialized>,
  "detail":         "<predicate that failed, with the offending witnesses>"
}
```

The record must suffice to replay the exact $\Phi$ call deterministically (via
`rng_seed` + `ruleset` + `state_pre` + programs) with no reference to external state —
consistent with the pure-function contract (M1) and the referee/observation separation
of spec §14.

---

### Appendix — coverage cross-reference to spec lemmas

| Spec lemma | Enforced by |
|---|---|
| 6.2 well-foundedness of fizzle | R1, M2(a) |
| 6.3a determinacy of matching | R4, M2(b) |
| 6.3b color-symmetry of matching | M3 |
| 6.4a confluence of capture DAG | R7, M2(c) |
| 6.4b color-symmetry of precedence | M3 |
| 6.4c termination of cascade | R12 |
| 12.1 contact vs. ranged defense | R7 (path re-check), WF6 (fire-time validity) |
| §0 no first-mover advantage | M3, M4 |
| §8.1 existence of value | M5 (solver-layer target) |
