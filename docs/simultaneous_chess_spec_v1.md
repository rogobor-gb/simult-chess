# Simultaneous Chess — Formal Specification, v1.0

*(working title; the game is unnamed. This document is the ground-truth rule specification and is intended to become the reference against which the rules engine is validated.)*

---

## 0. Purpose, scope, and the type of the object

This document specifies a deterministic, zero-sum, two-player **simultaneous-move** board game played on the standard $8\times 8$ chess state space. The game alternates a **decision phase**, in which both players privately commit a *program* of up to $N$ actions, and a **resolution phase**, in which a deterministic operator $\Phi$ maps the two programs and the current state to a successor state. Play continues until a terminal predicate fires (a king removed, or a draw condition).

**Base version (this document): perfect information.** After each resolution phase the full outcome — including all reservations, whether they fired or not — is common knowledge. Under this assumption the game is a finite **perfect-information simultaneous-move stochastic game** in the sense of Shapley (1953): each decision phase is a finite matrix game, the value of the whole game exists, and — because simultaneity embeds Matching Pennies (§8) — optimal play is generically in **mixed (behavioral) strategies**. A dedicated **imperfect-information variant**, in which reservations are hidden until they fire, is specified in Chapter 11 and changes the solution concept (from stage-matrix minimax to CFR-type methods) but not the transition operator.

Design invariant threaded through every rule below: **simultaneity is never silently broken.** Any construction that lets one player's action see the board left by another player's action within the same phase reintroduces tempo — the very thing the framework exists to abolish — and is rejected. The two places where ordering *does* enter (the annihilation matching of §6.3 and the defensive-precedence of §6.4) are proved color-symmetric, so neither confers a first-mover advantage on White.

Notation is fixed once in §1 and used throughout. All symbols are defined; all conventions that resolve an underdetermination are marked **[C]** (convention) and, where a different choice is defensible, the alternative is named. Items still requiring the designer's ruling are marked **[OPEN]**.

---

## 1. Preliminaries and notation

### 1.1 Board, colors, pieces

- **Squares.** $\mathcal S = \{0,\dots,7\}^2$, a square written $q=(c,r)$ with file $c$ and rank $r$; algebraic names (`a1`$=(0,0)$, …, `h8`$=(7,7)$) are used informally.
- **Colors.** $\Omega=\{\mathrm W,\mathrm B\}$; for $\omega\in\Omega$ write $-\omega$ for the opponent.
- **Piece tokens.** A finite set $\mathcal P$ of *tokens*, each with a fixed color $\mathrm{col}:\mathcal P\to\Omega$ and a *mutable* type $\mathrm{typ}:\mathcal P\to\mathcal T$, $\mathcal T=\{\mathsf p,\mathsf n,\mathsf b,\mathsf r,\mathsf q,\mathsf k\}$ (pawn, knight, bishop, rook, queen, king). Type is constant except under promotion (§6.5). Tokens, rather than (square, type) pairs, are primitive because reservations and captures track *identity*.

### 1.2 State

A **state** is
$$
s \;=\; \big(\,\beta,\; C,\; R_{\mathrm W},\; R_{\mathrm B},\; \eta\,\big),
$$
where:

- **Occupancy** $\beta:\mathcal P^{\text{live}}\hookrightarrow\mathcal S$ is an injective partial map from the live tokens $\mathcal P^{\text{live}}\subseteq\mathcal P$ to squares; $\beta^{-1}:\mathcal S\rightharpoonup\mathcal P^{\text{live}}$ is its partial inverse (the occupant of a square, if any). A square is *empty* if $\beta^{-1}$ is undefined there.
- **Cooldown set** $C\subseteq\mathcal P^{\text{live}}$: tokens that *displaced* in the immediately preceding resolution phase and are consequently inert this phase (§7). By construction $C$ contains no pawn or king token.
- **Reservations** $R_\omega$: a finite, **age-ordered** list of *active reservations* held by player $\omega$ (§4.3).
- **Bookkeeping** $\eta=(\text{castling rights},\;\text{repetition ledger},\;\text{no-progress counter},\;\text{phase index})$. (En passant is dropped in v1; see §6.5.)

### 1.3 Trajectories, swept sets, edges

A **move** displaces a token along a **trajectory** $\tau=(q_0,q_1,\dots,q_\ell)$, $\ell\ge 1$, a lattice path legal for the token's type on the *declaration-time* board (§4.2). Two derived objects drive collision resolution:

$$
\sigma(\tau) \;=\; \{q_1,\dots,q_\ell\}\qquad\text{(\textbf{swept set}: path squares \emph{excluding the origin} $q_0$),}
$$
$$
\varepsilon(\tau) \;=\; \big\{(q_j,q_{j+1}) : 0\le j<\ell\big\}\qquad\text{(\textbf{directed edge set}).}
$$

**Jumpers.** For a knight, $\tau=(q_0,q_1)$ with $\sigma(\tau)=\{q_1\}$ and $\varepsilon(\tau)=\varnothing$: a knight *does not traverse* the intervening squares or edges, so it is immune to interior collisions — the correct chess intuition. The king and pawn are stepwise (nonempty $\varepsilon$); sliders (bishop, rook, queen) are stepwise along a ray.

The exclusion of the origin from $\sigma(\tau)$ is not cosmetic: it is what makes "no capture of a piece that vacates" a *theorem* (§6.3, Remark) rather than a special case.

---

## 2. The game in one paragraph (informal)

Each phase, both players simultaneously submit a program of $1$ to $N$ actions ($N=2$ in v1). An action is a **standard move/capture** (displace a piece by chess geometry) or a **reservation** ("defender $D$ shall recapture on protégé $Q$'s square if $Q$ is captured"), optionally a **castling** or **cancellation**. Programs are revealed and resolved by $\Phi$: pawn captures onto vacated squares *fizzle*; opposing moves whose paths meet *annihilate* (both removed); surviving movers arrive and capture stationary enemies; captures of defended pieces trigger **immediate recaptures that pre-empt** any same-phase capture of the recapturing defender (the *intermezzo*), possibly cascading. A displaced non-pawn, non-king piece is **inert next phase** (cooldown). A player whose king is removed loses; both kings removed in one phase is a draw.

---

## 3. Terminology for the resolution

Fix a state $s$ and two programs $\pi_{\mathrm W},\pi_{\mathrm B}$. Extract from them the multiset of **declared moves** $M=M_{\mathrm W}\sqcup M_{\mathrm B}$ (castling contributes two synchronized sub-trajectories, §6.6). For $m\in M$ write $\mathrm{org}(m),\mathrm{dst}(m)$ for origin and destination, $\sigma(m),\varepsilon(m)$ for its swept set and edges, and $\mathrm{tok}(m)$ for its token.

- A move **executes** if it is not *fizzled* (§6.2); otherwise the token remains on $\mathrm{org}(m)$.
- A token **vacates** its origin iff it executes a move (whether or not it later dies en route). *Annihilation does not un-vacate:* a piece killed mid-path has still left its origin square.
- $M^\ast\subseteq M$ denotes the set of **executing** moves.

---

## 4. Actions, programs, and declaration legality

### 4.1 Action types

An action $\alpha$ is one of:

1. $\mathrm{Move}(p,\tau)$ — displace token $p$ along trajectory $\tau$. Subsumes standard moves and standard captures (a capture is a move whose destination holds an enemy token at declaration; for pawns, captures are diagonal, pushes are straight).
2. $\mathrm{Reserve}(D,Q)$ — register a reservation with defender token $D$ and protégé token $Q$ (§4.3).
3. $\mathrm{Castle}(\text{side})$, side $\in\{\text{king},\text{queen}\}$ (§6.6).
4. $\mathrm{Cancel}(\rho)$ — remove an active reservation $\rho\in R_\omega$ **[OPEN, default: retained]** (§9).

### 4.2 Move geometry (declaration-time legality)

A trajectory $\tau$ for token $p$ of type $t$ at $q_0=\beta(p)$ is **geometrically legal on $\beta$** iff it matches the classical pattern for $t$, with **all interior squares empty on $\beta$** and destination empty (non-capturing) or holding an enemy token (capturing). Pawns: straight pushes (one, or two from the start rank, interior and destination empty) are non-capturing; diagonal one-step moves are capturing and require an enemy on the destination *at declaration*. Sliders require an unobstructed ray on $\beta$. Knight/king are single steps to an admissible target. Legality is assessed **on the current board $\beta$ only**; simultaneity is handled entirely by $\Phi$, never by look-ahead in declaration.

### 4.3 Reservations

A reservation is a triple $\rho=(D,Q,a)$: defender token $D$, protégé token $Q$, **age stamp** $a=(\text{phase},\text{intra-program index})$ ordering all reservations linearly. It is **admissible at declaration** iff the pattern $\beta(D)\to\beta(Q)$ is a *legal capturing trajectory* for $D$ on the current board (the defender could, geometrically, capture on the protégé's square). Semantics (§6.4): if $Q$ is captured while stationary on $\beta(Q)$, the reservation *fires* — $D$ captures the capturing piece on $Q$'s square — and this recapture **pre-empts** any same-phase capture of $D$.

Two structural allowances, both confirmed as core (not variant):

- **(R-multi-in)** A protégé may carry several reservations (multiple defenders). On firing, the **oldest valid** one fires; the rest expire for that trigger.
- **(R-multi-out)** A single defender may hold several reservations (defending several protégés, "for solidity"). It fires **at most once per phase**: firing displaces it, self-invalidating its remaining reservations.

The self-defense pattern of the running example — defended piece stationary, defender stationary (`e3 defends d4`) — and its aggressive dual — *"move a piece and let the arriving piece defend a square"* (`Kf4 ; e3 defends Kf4`) — are both admissible: the protégé may itself be a moving piece, provided it and the defender are **distinct tokens** and the defender is stationary. What is **forbidden [C]** is *mover-as-defender*: a token that displaces this phase cannot also be the defender of a reservation *fired* this phase (a token acts at most once; a reservation is an act). Formally this is automatic — a displaced defender has left $\beta(D)$, so the reservation is invalid at fire time (§6.4) — but it is stated explicitly.

### 4.4 The declaration-legality predicate $L(s,\pi_\omega)$

$\pi_\omega=(\alpha_1,\dots,\alpha_k)$, an **ordered** tuple, is legal iff:

1. **Budget.** $1\le k\le N$.
2. **Mandatory displacement.** At least one $\alpha_i$ is a $\mathrm{Move}$ or $\mathrm{Castle}$. *(Passing is illegal; §9 explains why reservations alone would otherwise constitute a covert pass. Degenerate exception **[C]**: if $\omega$ has no geometrically legal displacement at all, a reservation-only or empty program is permitted for that phase — there is no stalemate to protect, and the no-progress counter handles abuse.)*
3. **Distinct actors.** Each token appears in at most one action (no token both moves twice, nor moves and serves as the *fired* defender; but a token may move while *another* defends it).
4. **Cooldown respected.** No action's actor lies in $C$: a cooled token can neither move, castle, declare a reservation, nor fire one (§7).
5. **Own-consistency.** The player's own executing moves are pairwise non-conflicting under (V)/(E) (§6.3) and none lands on a square occupied by the player's own stationary piece. *An internally-conflicting program is illegal, not self-annihilating.*
6. **Geometric legality.** Each $\mathrm{Move}$ trajectory satisfies §4.2; each $\mathrm{Reserve}$ satisfies §4.3; each $\mathrm{Castle}$ satisfies §6.6; each $\mathrm{Cancel}$ names an $\rho\in R_\omega$.

The action set at $s$ is $A_\omega(s)=\{\pi:L(s,\pi)\}$.

---

## 5. The transition operator: overview

$$
s' \;=\; \Phi(s,\pi_{\mathrm W},\pi_{\mathrm B})
$$
is computed by the pipeline
$$
\underbrace{\text{6.2 Fizzle}}_{\text{who executes}}\;\to\;\underbrace{\text{6.3 Annihilation matching}}_{\text{who collides mid-path}}\;\to\;\underbrace{\text{6.4 Defense-precedence capture/recapture cascade}}_{\text{who is taken}}\;\to\;\underbrace{\text{6.5–6.7 closure}}_{\text{promotion, cooldown, reservations, bookkeeping}}.
$$
Each stage below is deterministic; the two stages involving order (6.3, 6.4) carry proofs of **determinacy** (order-independence) and **color-symmetry** (no White bias). §6.3 governs *mid-path collisions* by declaration-priority pairing; §6.4 governs *captures on destination squares* by unconditional defensive precedence. **These are two different principles in two different stages, and keeping them separate is what makes the operator coherent.**

---

## 6. The transition operator: full specification

### 6.1 Two collision primitives

For opposing executing moves $m_1\in M^\ast_\omega,\ m_2\in M^\ast_{-\omega}$:

- **(V) Vertex conflict:** $\sigma(m_1)\cap\sigma(m_2)\ne\varnothing$. Subsumes coincident destinations, landing into a slider's interior, and perpendicular path crossings.
- **(E) Edge conflict:** $\varepsilon(m_1)$ and $\varepsilon(m_2)$ share an edge in *opposite* orientation, i.e. $(u,v)\in\varepsilon(m_1)$ and $(v,u)\in\varepsilon(m_2)$. This is the head-on swap (e.g. `e4`–`e5` vs `e5`–`e4`, or two rooks passing on a file). **(E) is not implied by (V)** — swept sets of a swap are disjoint — hence its separate clause.

A pair **conflicts** if (V) or (E) holds (possibly at several squares/edges; still one conflict). By own-consistency (L5), same-color executing moves never conflict, so **every conflict is bipartite** between colors — the fact that makes §6.3 a bipartite matching.

### 6.2 Stage F — fizzle resolution (who executes)

Two fizzle sources; both are functions of *declarations alone*, hence computable before any collision, because whether a token vacates depends only on whether it *starts* moving (§3), not on whether it survives.

- **(F2) Pawn convergence.** If two opposing pawn moves declare the same destination, **both fizzle**. *Realizability lemma:* this can occur **only** as two opposing pushes onto a shared empty square (every push/capture or capture/capture combination onto one square is geometrically impossible: a diagonal capture requires an enemy occupant, a push requires vacancy, and the destination cannot be simultaneously occupied-by-one-color and empty). So (F2) is exactly "opposing pushes collide," matching the design intent that two pawns landing together simply lose their moves.
- **(F1) Vacated pawn capture.** A pawn's *diagonal capture* onto square $x=\mathrm{dst}(m)$, targeting occupant $V=\beta^{-1}(x)$, **fizzles** iff $V$ executes a move (i.e. $V$ vacates $x$ — whether it completes elsewhere or dies en route). A pawn cannot capture an empty square.

**Lemma 6.2 (well-foundedness of fizzle).** *Fizzle status is a well-defined function of declarations.*
*Proof.* (F2) is syntactic and independent. For (F1), form the dependency digraph on pawn-capture moves: an edge $m\to m'$ iff $m$'s target is $\mathrm{tok}(m')$. Each node has out-degree $\le 1$ (a move waits on at most its single target's move), so the digraph is *functional*; its only cycles are 2-cycles. A 2-cycle is two pawns each diagonally capturing onto the other's square — i.e. both traversing the single diagonal edge between two adjacent squares in opposite directions — which is an **(E) conflict**: both execute and are annihilated in §6.3, so neither fizzles and the cycle leaves the F1 system. Cycles of length $\ge 3$ are geometrically impossible (alternating diagonal steps close only in even length; a would-be 3-cycle forces two same-color pawns onto one square, barred by L5). A functional digraph whose only cycles are exogenously resolved 2-cycles is well-founded; fizzle is computed by backward induction along its finite chains. $\qquad\blacksquare$

Output: $M^\ast$, the executing moves. Fizzled moves incur **no cooldown** (no displacement occurred) but **consume their action slot** (slots are spent at declaration). A fizzle produces no board change and is invisible to the opponent in the hidden variant (Ch. 11); some fizzles are *inferable* by their owner (a pawn push fizzled by (F2) reveals the opponent declared the same push) — a feature, not a leak.

### 6.3 Stage A — annihilation matching (who collides mid-path)

On $M^\ast$, build the **bipartite conflict graph** $G=(M^\ast_{\mathrm W},M^\ast_{\mathrm B},E)$ with $\{m,m'\}\in E$ iff $m,m'$ conflict (6.1). Index each player's moves by their **declaration order** $1,\dots,k$ (the internal order of the program, chosen by its owner — a genuine strategic decision). Define the rank of a candidate annihilation edge $\{W_i,B_j\}$:
$$
r(W_i,B_j)\;=\;\big(\max(i,j),\,\min(i,j)\big),\qquad\text{ordered lexicographically.}
$$

**Canonical greedy matching.** Process the edges of $E$ in increasing rank; on reaching $\{W_i,B_j\}$, if both endpoints are still alive and unmatched, **annihilate** the pair (both tokens removed; each pair member's intended capture never occurs). A move none of whose conflict-partners survive to meet it **completes unharmed**.

**Lemma 6.3a (determinacy).** *The surviving set is independent of the order in which equal-rank edges are processed.*
*Proof.* Equal rank occurs only for $\{W_i,B_j\}$ and $\{W_j,B_i\}$ with $i\ne j$; these edges are vertex-disjoint (four distinct tokens), so they commute in the greedy scan. All other ranks are distinct. $\qquad\blacksquare$

**Lemma 6.3b (color-symmetry).** *The resolution is equivariant under the color-swap involution $W_i\leftrightarrow B_i$; White gains no advantage from priority.*
*Proof.* The swap sends $\{W_i,B_j\}\mapsto\{W_j,B_i\}$, of identical rank $(\max,\min)$. Hence the rank order on $E$ is swap-invariant, and greedy matching commutes with the swap. $\qquad\blacksquare$

*Remark (vacated-square theorem).* If an enemy knight vacates `a5` while your rook plays `a1`–`a8`, the knight's swept set is $\{$its destination$\}$; the intersection with the rook's path is empty (origins excluded), so the rook passes through the vacated `a5` unharmed. "No capture of a piece that moves away" is thus a corollary of the origin-exclusion in $\sigma$, not a separate rule.

*Worked cases.* (i) Rook `a5`–`h5` crossing a black rook at `d5` (=$B_1$) and a black bishop at `f5` (=$B_2$), stationary knight on `h5`: edge $\{W_1,B_1\}$ (rank $(1,1)$) fires — rook and $B_1$ annihilate; $B_2$'s only partner is dead, so the bishop completes; the `h5` knight survives because the rook's capture never arrived. (ii) Four-rook cycle $W_1\!-\!B_1\!-\!W_2\!-\!B_2\!-\!W_1$: $\{W_1,B_1\}$ (rank $(1,1)$) then $\{W_2,B_2\}$ (rank $(2,2)$) fire — two clean $1$-for-$1$ trades; the rank-$(2,1)$ cross edges are skipped (dead endpoints). Cyclic conflict is a non-issue for *matching* — it was only pathological for fixed-point semantics, which we never use.

*Timed model (declined; see Ch. 12).* The untimed set model annihilates perpendicular crossings and interior-landings identically ("the pieces met in the fog of simultaneity"). The only principled way to distinguish them geometrically is a one-tick-per-square timed simulation, which reintroduces a raft of edge cases; it is a variant lever, not the base game.

### 6.4 Stage B — defense-precedence capture and recapture (who is taken)

After §6.3, at most one surviving mover targets any given square (two movers sharing a destination would have conflicted under (V) and been matched, unless one fizzled). Let the **pending captures** be the pairs $c=(\alpha,V,x)$ where surviving mover $\alpha$ arrives at $x=\mathrm{dst}$ holding stationary enemy $V=\beta^{-1}(x)$.

The **intermezzo** (unconditional defensive precedence; the designer's confirmed Reading (ii)): when $V$ is captured, its **oldest valid** reservation fires — defender $D$ recaptures on $x$, capturing $\alpha$ — and this recapture **categorically pre-empts** any same-phase capture of $D$, *regardless of declaration order*. Mechanically, $D$'s recapture vacates $\beta(D)$, so any pending capture targeting $D$ there *misses*. A reservation is **valid at fire time** iff $D$ is alive, still on $\beta(D)$, not in $C$, and the recapture trajectory $\beta(D)\to x$ is legal on the *current* (post-preceding-events) board (path unobstructed for sliders).

**The precedence DAG.** Over all capture events (initial and recapture-generated), place a directed edge $c\prec c'$ iff executing $c$ triggers a reservation whose recapture vacates the square targeted by $c'$ (thereby voiding $c'$). Resolve capture events in any topological order of $\prec$; each event, when resolved, (a) removes $V$, (b) fires $V$'s oldest valid reservation as a new capture event $c_{\text{re}}=(D,\alpha,x)$ if any, else (c) leaves $\alpha$ standing on $x$. Because a firing $D$ may capture an $\alpha$ that is *itself* defended, recaptures chain on a single contested square (a "battery") until one side exhausts valid reservations aimed at that square.

**Lemma 6.4a (confluence / determinacy).** *On the acyclic part of $\prec$, the surviving set is invariant across topological orders.*
*Proof.* The sole coupling between distinct capture events is "a firing defender vacates its origin, voiding a capture targeting that origin" — which is *exactly* a $\prec$-edge. Events incomparable under $\prec$ act on disjoint squares and commute; along each $\prec$-edge the behavior is fixed. The local diamond property holds, so by Newman's lemma the terminal state is order-independent. $\qquad\blacksquare$

**Lemma 6.4b (color-symmetry).** *Defensive precedence favors defense, never White.*
*Proof.* $\prec$ is defined through the color-agnostic *defends*-relation; the color-swap involution maps $\prec$ to itself. Precedence is granted to *whichever* side is defending, symmetrically. (Contrast the rejected sequential-execution reading, which handed White's *offense* a free tempo; the intermezzo hands *both* defenses the same categorical priority.) $\qquad\blacksquare$

**Mutual-defense cycle (the only tie-break).** If $P$ defends $Q$ and $Q$ defends $P$ and both are attacked in one phase, $\prec$ has a 2-cycle: no topological order exists. **[C]** Resolve each strongly-connected component under *base* semantics (no precedence): both defenders are actually captured, neither recapture fires. This is the only symmetric resolution — two mutual defenders cannot each dodge by recapturing while each is being taken — and it essentially never arises in practice.

**Cascade across stages.** Recaptures are themselves captures and may trigger deeper reservations; the process is the transitive closure of $\prec$ plus per-square battery chains.

**Lemma 6.4c (termination).** *The cascade reaches quiescence in $<\lvert\mathcal P^{\text{live}}\rvert$ steps.*
*Proof.* Every fired recapture removes at least one token (the captured $\alpha$) and never resurrects one; token count strictly decreases and is bounded below by $0$. $\qquad\blacksquare$

**Worked example (the designer's ambiguity, resolved).** White pawns on `d4`,`e3`; reservation $(\text{e3-pawn}\ \text{defends}\ \text{d4-pawn})$. Black plays $R\times$`d4` and $R\times$`e3`. Precedence: capturing `d4` triggers `e3`'s recapture $e3\!\to\!d4$, which vacates `e3` and voids the capture of `e3`; hence $(\text{cap }d4)\prec(\text{cap }e3)$. Topological order resolves `d4` first: `d4`-pawn dies, `e3`-pawn recaptures on `d4` (killing the black rook), `e3` is now empty, so the black rook aimed at `e3` merely occupies an empty square (no capture). Net: Black loses a rook for a pawn — the defense holds, **independent of Black's declaration order**, which is the essence of Reading (ii).

*A pleasing corollary — the ghost of check.* A king that greedily captures a defended piece is executed by the stage-$1$ recapture and the game ends. Reservations are the simultaneous-game surrogate for "check," and they bite **even from beyond the grave of the defended piece.**

### 6.5 Stage C — arrivals, promotion, pawns

Surviving, non-recaptured movers occupy their destinations. **Promotion** resolves here: a pawn reaching the last rank becomes the type chosen at declaration; being no longer a pawn, it enters cooldown for one phase (§7). A pawn promoted in one stage can be the object of a recapture in a later stage — well-defined. **En passant** is dropped in v1 **[C]**: the swept-set semantics already punish a double-step past an enemy pawn *if the enemy declared the interception* (the double step has a nonempty swept set and participates in (V)), which is the honest simultaneous analogue; retaining en passant would graft an alternation-native rule onto a simultaneous game.

**Pawn special cases (consolidated).**
- Two opposing pawns onto one square: **both fizzle** (F2), no cooldown (pawns are cooldown-exempt anyway).
- Head-on adjacent pawns (`e4`–`e5` vs `e5`–`e4`, or mutual diagonal captures): **(E) conflict**, mutual annihilation. The pawn exception is kept **as narrow as possible** — it covers convergent pushes only, not swaps.

### 6.6 Castling

$\mathrm{Castle}$ is a single action (one slot) contributing two synchronized sub-trajectories (king two squares, rook to the king's far side), each assessed independently under (V)/(E). A hit on the king component ends the game; a hit on the rook component kills the rook while the king completes. The king is never cooled; the **rook is cooled**. Classical "cannot castle through check" is *vacuous* here (no check exists), so castling legality reduces to the geometric/history conditions in $\eta$ (king and rook unmoved, squares between empty on $\beta$).

### 6.7 Stage D — phase closure

- **Cooldown update.** $C' = \{\text{tokens that displaced this phase}\}\setminus(\text{pawns}\cup\text{kings})$, including recapturers (they displaced) and promoted pieces. All others thaw.
- **Reservation update.** A reservation $(D,Q,\cdot)$ is invalidated (removed from $R_\omega$) iff $D$ is dead or off $\beta(D)$, or $Q$ is dead or off $\beta(Q)$ (the protégé displaced), or it fired this phase. Cancellations (§9) apply here. Surviving reservations persist with unchanged age.
- **Bookkeeping.** Update castling rights, repetition ledger (on the public position, §10), no-progress counter (reset on any capture or pawn displacement; else $+1$), phase index.
- **Event log** (perfect-info: the full delta). Emitted per Ch. 11's observation function in the hidden variant.

---

## 7. Cooldown — total inertia

A token is in $C$ iff it *displaced* in the previous phase and is not a pawn or king. While cooled it is **fully inert**: it cannot move, castle, declare a reservation, or fire a recapture. This is the deliberate classical-chess vulnerability window ("a sitting duck") that gives the game a sense of the ordinary take and kills the flashing/oscillation exploits (knight off-and-back, uncapturable rooks, infinite finales). Pawns (slow, forward-only) and kings (must always be able to move) are exempt.

*Internal consistency (near-theorem).* "A cooled piece cannot fire a recapture" is almost forced: cooling requires displacement, displacement vacates the origin, and a reservation is invalid once its defender leaves its square. Conversely, a defender holding a *valid* reservation has, by definition, not moved, hence is never cooled, hence is always eligible to fire — cooldown and defense compose cleanly. The design consequence flagged earlier holds: a defended piece *on cooldown cannot flee*, so its reservation is its only shield — friction and defense reinforce each other.

*Tunable.* Whether a **recapture** incurs cooldown is a balance parameter **[TUNABLE]**. v1 default: **yes** (recapturing exposes the defender next phase, pricing the defense). Suppressing it yields faster, more fluid games; to be decided empirically (Ch. 13).

---

## 8. Game-theoretic characterization

### 8.1 The object

Under perfect information (base version), $\Gamma=(s_0,\{A_\omega(s)\},\Phi,T,u)$ is a finite two-player zero-sum stochastic game with **deterministic** transition $\Phi$ but **simultaneous** action choice. Each decision phase at state $s$ is a matrix game $\big(u^\ast(s,\pi_{\mathrm W},\pi_{\mathrm B})\big)_{\pi_{\mathrm W}\in A_{\mathrm W}(s),\ \pi_{\mathrm B}\in A_{\mathrm B}(s)}$, where $u^\ast$ is the continuation value of $\Phi(s,\cdot,\cdot)$.

**Existence of value.** By the minimax theorem each stage game has a value in mixed strategies; by Shapley (1953) the zero-sum stochastic game has a value. The no-progress rule (§10) makes the horizon *effectively finite* (the state's public component cannot cycle indefinitely without a draw), so the value is obtained by **backward induction over matrix games** and *no discounting is required*. Zermelo's theorem does **not** apply: pure optimal strategies generically fail to exist.

### 8.2 Matching Pennies is unavoidable (mixed strategies are essential)

Consider a minimal *dodge* subgame. Attacker $A$ may aim its capture at square $x_1$ or $x_2$; the defending king sits on $x_1$ and may *stay* ($x_1$) or *flee* ($x_2$). Payoff to $A$ (win $=1$, else $0$), reading (V) semantics on coincident arrivals:

$$
\begin{array}{c|cc}
 & K\text{ stays }x_1 & K\text{ flees }x_2\\\hline
A\text{ aims }x_1 & 1 & 0\\
A\text{ aims }x_2 & 0 & 1
\end{array}
$$

$A$ wins iff it *matches* the king's square (aim-$x_1$/stay, or aim-$x_2$/flee where both land on $x_2$ and the king is annihilated). This is **Matching Pennies**: no pure equilibrium, unique value $\tfrac12$, optimal play uniform mixing. The embedding shows that simultaneity injects irreducible mixing into the endgame; an engine must approximate stage equilibria (SM-MCTS, regret matching / CFR) rather than perform pure minimax.

### 8.3 Elementary "wins" need not be forced — the KQ–K remark

The §8.2 subgame is the atom of a **pursuit–evasion** game: $\mathrm K\mathrm Q$ vs $\mathrm K$ under simultaneity is a cousin of *cops-and-robbers with simultaneous moves* (Isaacs-style pursuit). The per-epoch capture probability is bounded strictly below $1$ (the king escapes with positive probability whenever it retains $\ge 2$ safe squares the queen cannot simultaneously cover). Two questions are therefore genuinely open and **empirical, not to be legislated**:

1. Whether the *cumulative* capture probability over the no-progress horizon reaches $1$, i.e. whether the pursuer can force the king's safe region to shrink faster than the horizon expires, or whether the king can perpetually regenerate escape options (yielding a value strictly $<1$).
2. How the horizon length $H$ (§10) trades convertibility of material against draw-inflation.

This is arguably the most novel feature of the design: **material advantage is probabilistic, and $H$ is a first-class balance parameter.** It should be measured once the engine exists, not fixed dogmatically now.

### 8.4 Branching and computational regime

With $N=2$, per-player declarations scale as $\binom{b+d}{2}$ where $b\approx 35$ (legal displacements) and $d$ the count of admissible reservations; stage matrices reach $10^6$–$10^8$ entries. Exact LP per node is infeasible; the engine lives on **sampled stage equilibria** from the outset — an argument for keeping v1 rules lean and holding $N=2$ (Ch. 13).

---

## 9. Cancellation of reservations **[OPEN — default: retained]**

Cancellation is *useless unless recapture is automatic* — and it is (the reservation is a genuine pre-commitment "if $Q$ is taken, $D$ *will* recapture"). Given automaticity, cancellation earns its keep in exactly one situation: the committed recapture has become **bad** (it would drag $D$ into a fork, or the exchange has turned unfavorable) and the player wants $D$ to stay put *without executing it*. Moving $D$ also cancels the reservation but costs a slot and relocates the piece; cancellation drops the commitment while $D$ stays. Because of simultaneity it is a **blind withdrawal** — committed in a decision phase before seeing whether the opponent triggers it — a Schelling-style deterrence/commitment lever (keep the reservation as a visible deterrent vs. withdraw to dodge a forced bad recapture). Under the mandatory-move rule (L2), cancellation cannot be abused as a covert pass, so v1 makes it **free and slot-less** (apply at closure, §6.7). *Alternative:* drop cancellation entirely, making the defense irrevocable and shrinking the state — coherent and simpler; the designer's call.

---

## 10. Termination and draws

- **King capture is terminal.** A player whose king is removed in *any* stage of a phase **loses** ($u=\pm1$). Both kings removed in the *same phase* (any stages) is a **draw** ($u=0$) — the "synchronous checkmate," embraced as a thematic feature.
- **No check/checkmate.** These are alternation-native and do not survive simultaneity (there is no well-defined "illegal because it leaves the king en prise" when the threat itself may dodge). The king is ordinary matter under (V)/(E): a running king whose path conflicts with an enemy trajectory dies; a king–king conflict is a draw.
- **Repetition.** Threefold repetition of the **public position** $(\beta,C)$ is a draw. *(Repetition is defined on the public component because, in the hidden variant, players cannot verify each other's reservations; in the base version $(\beta,C)$ is still the right coordinate because two boards differing only in dormant reservations are strategically distinct and should not be forcibly drawn on board-repetition alone — but see the caveat that a losing player must not evade indefinitely.)*
- **No-progress.** $H$ phases without a capture or a pawn displacement is a draw. $H$ is a **balance parameter** (§8.3); v1 placeholder $H=50$, to be tuned. This is the mechanism that caps the winning probability of pursuit-type endgames and prevents a losing player's pieces from "jumping around" to evade capture indefinitely.

---

## 11. Chapter — the hidden-information variant

The base game is perfect-information. This chapter specifies the **imperfect-information** variant, in which reservations are *private until they fire*. It reuses $\Phi$ unchanged; only the observation structure differs.

### 11.1 The object

$\Gamma^{\mathrm{hid}}$ is a finite two-player zero-sum **extensive-form game with imperfect information** (simultaneity being a special case realized by information sets). Perfect recall holds by construction (each player observes its own past actions and all public events). Von Neumann's theorem gives a value and minimax-optimal **behavioral** strategies. The computational target shifts from stage-matrix solving to the **CFR family** (MCCFR at scale) with **belief-state** search; states are replaced by information sets.

### 11.2 Observation function (the executed-event log)

After each phase, **public** information is: every executed displacement (origin→destination, which determines the trajectory for all types), every annihilation (revealing *both* trajectories of the matched pair — the vanished pieces "tell their story"), every stationary capture, and every reservation **at the instant it fires** (defender square, target square). **Private** information: standing (unfired) reservations, cancellations, and fizzled moves (a fizzle has no observable effect, so hiding it is physically natural; owner-inferable fizzles remain a feature, §6.2). *Rejected alternative:* board-delta-only observation is strictly more hidden but makes even move-attribution ambiguous (identical rooks) and inflates information sets brutally.

### 11.3 The covert-pass problem and its fix

With hidden reservations and free cancellation, "declare junk reservations, cancel next phase" would be an undetectable pass. The fix is already in the base rules: **L2 (mandatory displacement)** — every program contains $\ge 1$ move/castle; reservation slots are the optional ones. With L2 in force, cancellation may remain free (the anti-stall work is done by L2, not by taxing cancellation).

### 11.4 Bluffing and strategic content

Hidden reservations make the *appearance* of a defense a first-class mechanic: a piece may look defended (deterring capture) while undefended, or vice versa. Under hidden information the **declaration order** of §6.3 is itself unobservable, adding a small layer of bluff to multi-conflict pairing. This is the variant the designer considers "more interesting"; it is the natural second milestone once the visible game is validated.

### 11.5 Protocol (arbiter-free remote play)

Hidden commitments require either a trusted arbiter or a **commit–reveal** scheme: at declaration each action is published as a salted hash commitment; at resolution, only *fired/executed* actions are selectively revealed (with salt) and verified; unfired reservations stay committed but concealed. Building the referee as **referee + two observation channels + commit–reveal** from day one *is* the information structure of the game, and it makes the perfect-information base a trivial special case (reveal everything).

---

## 12. Chapter — double reservations and defensive structures (core)

Multiple reservations are **core** to v1 (a loss of tempo traded for solidity). They induce a **defensive digraph** $\mathcal D$ on live tokens: an arc $D\to Q$ whenever $\omega$ holds reservation $(D,Q,\cdot)$. The cascade of §6.4 is a guided traversal of $\mathcal D$: a captured protégé fires its oldest in-arc; a defender with several out-arcs fires at most once per phase.

**Proposition 12.1 (contact defenses restore classical exchange arithmetic; ranged defenses admit a strictly-simultaneous refutation).**
*Let $Q$ be a piece defended by a single reservation with defender $D$, and suppose $N\ge 2$.*
*(a) If $D$ is a **contact** defender ($\beta(D)$ adjacent to $\beta(Q)$, single-step recapture), then in a single phase the attacker cannot avoid the recapture: capturing $Q$ costs the capturer, and attacking $D$ simultaneously is pre-empted by the intermezzo. The exchange therefore evaluates exactly as in classical sequential chess — attacker nets $\mathrm{val}(Q)-\mathrm{val}(\text{capturer})$ — and the free-capture-of-defender exploit of naïve simultaneity is eliminated.*
*(b) If $D$ is a **ranged** (slider) defender, the recapture traverses $\ge 1$ interior square; with a second action the attacker may **interpose** a piece on the recapture ray. The recapture path is then blocked, the reservation **fizzles at fire time** (§6.4 validity), and $Q$ is won while $D$ never recaptures — a refutation with **no classical analogue**.*

*Proof.* (a) The recapture $\beta(D)\to\beta(Q)$ has empty interior, so no interposition exists; and a same-phase capture of $D$ is voided because the fired recapture vacates $\beta(D)$ before that capture resolves (defensive precedence, Lemma 6.4b). Hence the only outcome is the trade. (b) The interior of the ranged recapture is nonempty; the attacker's second action places a token on an interior square. At fire time the recapture trajectory is obstructed, so the reservation is invalid; $Q$'s capture stands and no recapture occurs. $\qquad\blacksquare$

**Corollary 12.2.** *Contact-defended structures (e.g. a base pawn defending two diagonal pawns) are categorically more solid than ranged-defended ones, for a reason absent from ordinary chess.* This asymmetry should be playtested deliberately; it is a genuine design discovery, and it is the precise sense in which the intermezzo "gives defensive structures teeth."

**Depth.** Breaking a defense of depth $k$ (defender-of-defender chains) requires the attacker to either **out-cascade** it (remove recapturers down the chain) or **strike a defender in a phase when its protégé is not attacked** (no trigger $\Rightarrow$ the defender dies normally) — which the opponent answers by defending the defender. This recursion is exactly the "defensive structures" the framework was designed to support.

---

## 13. Parameters, conventions, and open items

**Parameters (tunable by playtest).**
- $N$: actions per phase. **v1: $N=2$**, held as a parameter (branching and per-phase variance arguments favor the minimum $N$ exhibiting all rule interactions, §8.4). $N\ge 3$ and larger boards deferred to variants.
- $H$: no-progress horizon (§10), a balance parameter controlling convertibility of material (§8.3). Placeholder $H=50$.
- Recapture cooldown (§7): default **on**; candidate to switch off for faster play.

**Conventions requiring a designer ruling / confirmation.**
- **[OPEN]** Cancellation retained (default) vs. dropped for an irrevocable defense (§9).
- **[C, confirm]** Pawn same-square fizzle triggers only when *both* movers are pawns (§6.2/6.5); the mixed pawn/non-pawn case is ordinary (V)-annihilation.

**Deferred variant levers (Ch.-length treatments to follow if adopted).**
1. **Hidden information** (Ch. 11) — the intended second milestone.
2. **Timed (one-tick-per-square) resolution** — distinguishes perpendicular crossings, enables in-phase interception/tempo; large edge-case surface.
3. **Geometric conflict selection** — pair a multi-crossing mover with its *first* victim along the path; requires stable-matching with tie-breaks (rejected for v1 in favor of declaration-priority, §6.3).
4. **Attacker-sequenced intermezzo (Reading (i))** — a leaner, order-dependent defense; to be A/B-tested against Reading (ii).
5. **Trap reservations** — reserve a square against *any* arrival (not just defense of an owned piece); flagged as too random for v1.
6. **$N\ge 3$, larger boards, reinstated en passant** — future variants.

---

## 14. What the operator does, in pseudocode (reference for the engine)

```
Φ(s, π_W, π_B):
    assert L(s, π_W) and L(s, π_B)                      # §4.4
    M = declared_moves(π_W, π_B)                         # §3
    # Stage F — fizzle (§6.2)
    fizz = resolve_fizzles(M)                            # F2 syntactic; F1 by backward induction
    M_exec = M \ fizz
    # Stage A — annihilation matching (§6.3)
    G = bipartite_conflict_graph(M_exec)                # (V) and (E)
    killed_A = greedy_match_by_rank(G)                  # rank (max(i,j), min(i,j)), lexicographic
    survivors = movers(M_exec) \ killed_A
    apply_arrivals_provisional(survivors)               # place non-capturing arrivals
    # Stage B — defense-precedence capture/recapture (§6.4)
    events = pending_captures(survivors)                # (α, V, x)
    resolve_by_precedence_DAG(events, R_W, R_B)          # topo order; oldest-valid reservation;
                                                        # mutual-defense SCC → base semantics
    # Stage C — promotion, pawn cases (§6.5)
    apply_promotions(); finalize_pawn_cases()
    # Stage D — closure (§6.7)
    C'      = displaced_this_phase \ (pawns ∪ kings)
    R_W, R_B = update_reservations(fired, invalidated, cancellations)
    η'      = update_bookkeeping(captures, pawn_moves, repetition, castling)
    emit_event_log()                                     # §11.2 in the hidden variant
    s' = (β', C', R_W', R_B', η')
    if king_removed: assign u ∈ {−1,0,+1}                # §10
    return s'
```

The reference implementation will follow PEP 8 with NumPy-style docstrings, will expose $\Phi$ as a pure function of $(s,\pi_{\mathrm W},\pi_{\mathrm B})$, and will separate the **referee** (state + $\Phi$) from **observation channels** (§11.2) and the **commit–reveal protocol** (§11.5), so that the perfect-information base is the special case "reveal everything." Stage equilibria for the engine (§8.4) will be approximated (SM-MCTS / regret-matching for the base game, MCCFR for the hidden variant); exact LP is used only for validation on small hand-built positions.

---

### Appendix A — lemma index

- 6.2 Well-foundedness of fizzle (functional digraph; 2-cycles exit via (E)).
- 6.3a Determinacy of annihilation matching (equal-rank edges are vertex-disjoint).
- 6.3b Color-symmetry of annihilation matching (rank invariant under color swap).
- 6.4a Confluence of the capture/recapture DAG (Newman's lemma; single coupling = $\prec$-edge).
- 6.4b Color-symmetry of defensive precedence (defends-relation is color-agnostic).
- 6.4c Termination of the cascade (strict token-count descent).
- 12.1 Contact vs. ranged defense (classical exchange arithmetic vs. interposition refutation).
