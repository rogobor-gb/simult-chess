# Collaboration scope — the engine and the application layer

*Licensed under CC BY 4.0. This document draws the line between the **engine**
(this repository) and an **application** built on top of it, so that "who owns
the rules" is answered in writing **before** an app has users — much easier to
settle in advance than after. It is organisational hygiene, **not legal
advice**: anything with money or contracts attached should be seen by a lawyer.*

## The two layers

**The engine — this repository.** Owned by the maintainer. It is:

- the formal **specification** (`docs/simultaneous_chess_spec_v1.md`) and the
  executable **invariants** (`docs/INVARIANTS.md`);
- the proven-correct transition operator **Φ** and the rules
  (`core/`, `rules/`), with a **frozen, fingerprinted `RuleSet`**;
- the **referee**, agents, the stage-matrix solver, and the self-play learning
  system (`referee/`, `agents/`, `solver/`, `learn/`);
- the **network layer** — handshake, commit-reveal, match services, the
  concurrent-bank clock, and the disposable relay (`net/`), documented as a
  versioned wire protocol (`docs/PROTOCOL.md`);
- the **game-record format** (`referee/record.py`, `.scn`) and the local Tk
  window (`ui/`).

Licensed: **code Apache-2.0**, **`docs/` CC BY 4.0** (`LICENSE`, `README.md`).

**The application layer — a separate project.** Whatever product a collaborator
builds *on top of* the engine: a web client, matchmaking, accounts, hosting,
persistence, presentation. It is **not** in this repository, is owned by
whoever builds it, and may carry its own licence and terms. It consumes the
engine as a dependency and/or speaks its wire protocol.

## The boundary — who owns the rules

**The rules are the engine's.** The spec, the invariants, `RuleSet`, and Φ are
changed only through the engine's spec-first process (`CONTRIBUTING.md`). An
application:

- **binds to the `RuleSet.fingerprint()`** and must not fork or silently
  reinterpret the rules; a different rule set is a different fingerprint, and
  game records and the handshake make a mismatch detectable, not silent;
- **must not reimplement Φ from memory.** If an application needs the operator
  in another language, that is a *port* of the spec with the engine's
  conformance bar (bit-identical traces over the golden fixtures and a seeded
  sweep, gated by the invariant harness) — never an independent re-derivation.

## The stable interface across the boundary

The engine exposes, as its public contract to an application:

- **the wire protocol** (`docs/PROTOCOL.md`, `protocol_version`) — the message
  schema, ordering, commitment construction, clock rules, and divergence checks;
- **the `.scn` game-record format** — self-describing, fingerprint-checked,
  replayable;
- **the engine-side UI primitives** a client needs — `check_partial_program`
  (live legality) and `ui/affordances.py` (legal destinations, reservation
  pairings, the threat overlay). These were built to serve a web client
  identically to the bundled Tk window.

Everything above that line — rendering, input, matchmaking, accounts, hosting,
storage, product decisions — belongs to the application.

## Practical division

- **Engine work** (rules, Φ, invariants, referee, net protocol, records, the
  learning system) goes to **this** repository, follows `CONTRIBUTING.md`, and
  is licensed as above.
- **Application work** lives in the **application's** repository, under its own
  terms, and treats the engine as a versioned upstream. Track the engine by its
  release tag and `RuleSet.fingerprint()`.
- A change that turns out to need a rule change stops and goes through the
  engine's spec-first process first; it does not get "worked around" in the app.
