# Simultaneous Chess — Wire Protocol v1

*Licensed under CC BY 4.0. Ground truth for the network layer; the rules
themselves are `simultaneous_chess_spec_v1.md`. This document is what a second
implementation (e.g. a web client) needs to interoperate with the reference
peer in `simult_chess/net/`.*

`protocol_version = 1`.

## 0. Transport and framing

A match is a single duplex byte stream between two peers — a direct TCP
connection (`host`/`connect`) or one relayed by a room code
(`net/relay.py`, which pipes bytes and parses none of this). Messages are
**newline-delimited JSON objects**, UTF-8, one per line, each with a string
`"type"` field. Numbers are JSON numbers; binary values (salts) are lowercase
hex strings. There is no length prefix and no message may contain a raw newline.

The two peers are symmetric: there is no client/server role once connected. The
relay is transparent — a peer joins a room by sending the room code as its first
line **before** any protocol message; everything after is piped verbatim.

## 1. Compatibility policy

`protocol_version` is exchanged in the handshake (§3) and must match exactly;
there is no negotiation of older versions in v1. A breaking change to any
message schema, the ordering contract (§2), the commitment construction (§4),
or the divergence checks (§5) bumps `protocol_version`. Adding an **optional**
field that an older peer ignores does not. `spec_version` (the rules) and the
`ruleset_fingerprint` (the rule *values*, `RuleSet.fingerprint()`) are versioned
independently and also cross-checked in the handshake.

## 2. Message sequence (the ordering contract)

```
handshake        (both, once)          §3
[dmax ping/pong] (both, if timed)      §6
── per phase k ────────────────────────────────────
  action slot    (both)                §3   commit | resign | abort | accept_draw
  reveal         (both, iff both committed)   §3/§4
  ack            (both, iff both committed)   §5
────────────────────────────────────────────────────
```

- `ping`/`pong` may be interleaved **anywhere** after the handshake as
  keepalive; a receiver answers a `ping` with a `pong` and otherwise ignores
  both. They never change game state.
- Each phase, each peer sends exactly **one action-slot** message. A `commit`
  means "I played a program"; `resign`/`abort`/`accept_draw` end the match.
- `reveal` and `ack` occur **only** when both peers committed. They should be
  sent immediately (they are bounded by a short *transport* timeout); the time a
  peer spends before its commit is *decision* time and is unbounded (or governed
  by the clock, §6).
- A draw offer is not a message; it rides on a `commit` as `"offer_draw": true`
  and stands until the offerer's peer's next action slot (accept with
  `accept_draw`, or decline by simply committing).

## 3. Message schema

| type | direction | fields |
|---|---|---|
| `handshake` | both, once | `protocol_version:int`, `spec_version:str`, `ruleset_fingerprint:hex64`, `initial_state_hash:hex64`, `max_phases:int`, `proposed_color:"W"|"B"`, `nonce:int` |
| `commit` | both, per phase | `phase_index:int`, `hash:hex64` (§4), `offer_draw:bool`, `elapsed_us:int?` **not sent here** — the time is *bound* into `hash` and disclosed only at `reveal` |
| `reveal` | both, iff committed | `phase_index:int`, `salt:hex`, `program:Action[]` (§7), `elapsed_us:int?` (timed only, §6) |
| `ack` | both, iff committed | `phase_index:int`, `state_hash:hex64` (§5), `clock_hash:hex64?` (timed only) |
| `ping` / `pong` | both, any time | `phase_index:int` (advisory); timed games also use `dmax_seq:int` (§6) |
| `resign` | either | `phase_index:int` |
| `abort` | either | `phase_index:int` |
| `accept_draw` | either | `phase_index:int` (valid only against a standing `offer_draw`) |

**Colour negotiation.** Each peer proposes a colour and a random `nonce`. If the
proposals differ, both are honoured. If they collide, the higher nonce keeps the
colour and the other flips; equal nonces (astronomically unlikely) abort. Any
mismatch of `protocol_version`, `spec_version`, `ruleset_fingerprint`,
`initial_state_hash`, or `max_phases` aborts before phase 0 with a named error.

**Outcome.** A match ends with an outcome in `{white_wins, black_wins, draw}`
(the payoff domain) and a separate `termination_reason ∈ {regicide, repetition,
no_progress, resignation, timeout, illegal_program, abort, draw_agreement,
phase_limit}`. The reason is metadata; it never enters the game state.

## 4. Commitment construction

A `commit`'s `hash` binds the program (and, in a timed game, the thinking time)
so neither peer can change it after seeing the other's:

```
hash = SHA256( salt ‖ canonical_json(program) [ ‖ "|t=" ‖ str(elapsed_us) ] )
```

- `salt` is 16 random bytes, disclosed (hex) only at `reveal`.
- `canonical_json` is `json.dumps(program, sort_keys=True, separators=(",",":"))`.
- The `‖ "|t=" ‖ elapsed_us` term is present **iff** the game is timed; an
  untimed commit omits it and hashes exactly as protocol v0 did.

On `reveal`, the receiver recomputes the hash from the revealed `salt`,
`program`, and `elapsed_us` and rejects a mismatch (`ProtocolError`). It then
checks the revealed program is legal under the agreed `RuleSet`; an illegal
program **forfeits its sender** (`termination_reason = illegal_program`), a
verdict both peers reach identically from identical data.

## 5. Divergence checks

Both peers resolve Φ locally (it is pure), then exchange an `ack`:

- `state_hash` — SHA-256 of the successor state's public position key
  (`referee.serialize.state_hash`). A mismatch aborts (`state diverged`).
- `clock_hash` (timed only, §6) — SHA-256 of the phase's clock-ledger entry.
  A mismatch aborts (`clock ledger diverged`).

## 6. Time control (timed matches only)

A concurrent-bank clock (`net/clock.py`, spec §13.5) is session metadata, never
part of the rules or state. When a match is timed:

1. **`d_max` estimate.** Right after the handshake, before the phase loop, the
   peers exchange a few `ping`/`pong` rounds tagged `dmax_seq`. Each measures
   half the largest round trip (floored) as its one-way-delay bound `d_max`.
2. **Self-measured time.** Each peer measures its own thinking time `t` from its
   local phase start to its commit, quantised to microseconds, and **binds it
   into the commitment** (§4). It is revealed as `elapsed_us` only at `reveal`.
3. **Cross-check.** On reveal, a peer checks the opponent's claimed `t` against
   its own observed arrival: the claim is admissible iff
   `|t_claimed − (arrival − phase_start)| ≤ d_max + ε`. An out-of-tolerance
   claim aborts by name.
4. **Ledger.** Both peers feed the two microsecond-quantised times into the same
   bank update — `B' = B − t + ι + β`, with `β` the race bonus (default
   capped-difference `min(b, |t_W − t_B|)` for the faster player) — so their
   ledgers are byte-identical and a flag-fall (`B − t ≤ 0`, reason `timeout`) is
   agreed by construction. The per-phase `clock_hash` (§5) confirms it.

## 7. Action encoding

A `program` is a JSON array of 1–2 actions (`net/protocol.py`):

- `{"kind":"move","token_id":int,"path":[[file,rank],…],"is_jump":bool,"promotion":str|null}`
- `{"kind":"reserve","defender_id":int,"protege_id":int}`
- `{"kind":"castle","side":"king"|"queen"}`
- `{"kind":"cancel","index":int}` (index into the mover's reservation list)

Tokens are referenced by integer id; both peers reconstruct the same token from
the same pre-phase state (Φ is deterministic), so colour/type are not resent.

## 8. Worked transcript — one full phase, captured live

A real capture (`net/session.py` over an in-memory transport, one untimed phase;
White plays `g2g3`, Black `a7a5`). Lines are labelled by sender; message order
interleaves the two peers exactly as it occurred.

```
White→ {"type":"handshake","protocol_version":1,"spec_version":"1.1","ruleset_fingerprint":"bf2bb9dab0f020b107e5cfb3d964f825f08fbcdb1a1c8c729776670f30d1491c","initial_state_hash":"44d5927d354d9c572e4f7c9747ac0da89ed342e6d8cc5fd258a28256071be846","max_phases":1,"proposed_color":"W","nonce":10499958131665514997}
Black→ {"type":"handshake","protocol_version":1,"spec_version":"1.1","ruleset_fingerprint":"bf2bb9dab0f020b107e5cfb3d964f825f08fbcdb1a1c8c729776670f30d1491c","initial_state_hash":"44d5927d354d9c572e4f7c9747ac0da89ed342e6d8cc5fd258a28256071be846","max_phases":1,"proposed_color":"B","nonce":15921556852572072307}
White→ {"type":"ping","phase_index":0}
Black→ {"type":"ping","phase_index":0}
White→ {"type":"commit","phase_index":0,"hash":"0be4d0b8420ed948c7809b00cfbea8e486b6cfb0cbfe8bdd74ba6dc45a98c614","offer_draw":false}
White→ {"type":"pong","phase_index":0}
Black→ {"type":"pong","phase_index":0}
Black→ {"type":"commit","phase_index":0,"hash":"b2add1388b1f6278191c3ce15c1c21b57749814418b750e25508e2f7e0905c5a","offer_draw":false}
Black→ {"type":"reveal","phase_index":0,"salt":"5213cb51f00cd0942e907b4c14d476be","program":[{"kind":"move","token_id":25,"path":[[0,6],[0,5],[0,4]],"is_jump":false,"promotion":null}]}
White→ {"type":"reveal","phase_index":0,"salt":"908dd1ec1c5c21f86129df9dcae0a12b","program":[{"kind":"move","token_id":15,"path":[[6,1],[6,2]],"is_jump":false,"promotion":null}]}
White→ {"type":"ack","phase_index":0,"state_hash":"bcc0cbe1dcf28642efbaba8012e0c3f71908c85e7e02088c849940d7489cdf29"}
Black→ {"type":"ack","phase_index":0,"state_hash":"bcc0cbe1dcf28642efbaba8012e0c3f71908c85e7e02088c849940d7489cdf29"}
```

Reading it: both handshake and agree; a keepalive `ping`/`pong` pair flows while
each side thinks; both `commit`; both `reveal` their programs (Black moves the
a7 pawn a7→a5, White the g2 pawn g2→g3); each verifies the other's commitment
and legality, resolves Φ, and both `ack` the **identical** successor
`state_hash` — the online analogue of "post-phase event logs are byte-identical".
This block is regenerated from a live session, not written by hand.
