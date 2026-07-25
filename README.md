# simult-chess

A deterministic, zero-sum, two-player **simultaneous-move** chess variant.
Each phase, both players privately commit a program of up to $N=2$ actions
(moves, castling, reservations, cancellations); a pure transition operator
$\Phi$ resolves both programs at once — no alternation, no turns. Ground
truth for the rules is `docs/simultaneous_chess_spec_v1.md` (**spec**); the
executable correctness contract is `docs/INVARIANTS.md` (**inv**).

`core`/`rules` are standard-library only (no runtime dependencies); optional
extras add a UI, network play, and a test-only cross-validator.

## Install

```bash
# core engine only
pip install -e .

# + UI/CLI play, network play, dev tooling
pip install -e ".[dev]"

# + geometry cross-validation against python-chess (test-only; GPL-3.0-or-later,
# quarantined behind this extra so the license never attaches to core/rules)
pip install -e ".[oracle]"

# + the stage-matrix/LP solver layer and its matrix_1ply agent (numpy, scipy;
# quarantined behind this extra so core/rules stay standard-library only)
pip install -e ".[solver]"

# + the OpenSpiel (pyspiel) simultaneous-move adapter (open_spiel, numpy;
# quarantined behind this extra so core/rules/referee stay free of it)
pip install -e ".[openspiel]"
```

Requires Python ≥3.10.

## Play

Local, one terminal, two humans (commit-reveal so neither sees the other's
program early):

```bash
python -m simult_chess.ui.cli hotseat
```

Local, human vs. an agent:

```bash
python -m simult_chess.ui.cli agent --human white --agent greedy --seed 0
```

Online, direct TCP connection (one side hosts, one connects; no relay/NAT
traversal in v1):

```bash
# host, e.g. playing White, waiting for a human on the other end
python -m simult_chess.net.cli host --port 5000 --color white --agent human

# connect from the other machine/terminal
python -m simult_chess.net.cli connect --remote-host <host-ip> --port 5000 --color black --agent human
```

Before phase 0 the two peers run a **handshake** that agrees the protocol and
spec versions, the ruleset fingerprint, the initial position, and the phase
limit, and *negotiates* colour — so two `--color white` peers get a playable
assignment instead of a crash, and a fingerprint or spec mismatch aborts with a
named error rather than a wrong game. During play a keepalive lets a side think
as long as it likes without tripping a false timeout, while still detecting a
genuinely dead peer; `--transport-timeout`, `--keepalive-interval`, and
`--liveness-deadline` tune those bounds. At the prompt a human may type a
program, `resign`, `abort`, `accept` (a standing draw offer), or prefix a
program with `draw ` to offer one. A match ends with an outcome in
`{white_wins, black_wins, draw}` and a separate reason (`regicide`,
`resignation`, `draw_agreement`, `abort`, …).

Both CLIs accept `--agent {human,random,greedy}` (net) or `--agent
{random,greedy}` with `--human {white,black}` (ui). Programs are entered in a
short text DSL — see `src/simult_chess/ui/notation.py`'s module docstring for
the grammar (e.g. `Nf3`, `e4=Q`, `O-O`, `e3 def d4`, `cancel 0`).

## Rules: the frozen defaults, and variants

Play defaults to the **frozen provisional v1.1** rule set (`RuleSet()`), whose
identity is its fingerprint — `bf2bb9dab0f0…`, a SHA-256 over the rule-bearing
fields. Every rule value the Phase 11b campaign tested and the freeze declined
stays playable *by name*, never by forking:

```bash
python -m simult_chess.ui.cli variants                       # list them
python -m simult_chess.ui.cli hotseat --variant horizon_30   # play one
```

Both CLIs take `--variant`; each prints its rule set and fingerprint at start.
For a networked game the handshake cross-checks that fingerprint automatically
and aborts a mismatch by name, so both peers must pass the same `--variant`.
"Frozen" means *versioned*, not *proven optimal* — see the freeze block at the
top of [`reports/campaign_v1.md`](reports/campaign_v1.md) and §13 of the spec.

## Game records (`.scn`)

Every game — self-play, hot-seat, or online — can be written to a `.scn`
record: a header pinning the spec/protocol versions, the ruleset fingerprint
and full field dump, the initial position, player labels, and seed, then one
line per phase carrying both programs in the notation DSL and the resolved
outcome. Replay re-derives every state through Φ and **verifies** the recorded
resolutions rather than trusting them:

```bash
python -m simult_chess.referee.record_cli replay game.scn
python -m simult_chess.referee.record_cli replay game.scn --expect-variant horizon_30
python -m simult_chess.referee.record_cli to-fixture game.scn --phase 12 --out phase12.json
```

A record whose stored fingerprint disagrees with its dumped rules, or with an
`--expect-variant` you name, refuses to replay with a named error rather than
silently producing different results — so a record is a durable, citable object
that always reproduces the exact game it was made from (build a record from a
finished game with `referee.record.build_record`).

## Tests and sweeps

```bash
pytest                      # full suite (unit + property), fast
pytest -m "not slow"        # excludes exhaustive/high-volume checks
scripts/check.sh            # ruff + mypy --strict + the fast pytest subset
```

`tests/property/` holds the metamorphic suite (inv M1–M4: purity, internal
order-independence, χ-color-swap equivariance) and the geometry
cross-validation against python-chess (needs the `oracle` extra; skips
cleanly without it). `tests/unit/test_openspiel_*.py` cover the OpenSpiel
adapter (needs the `openspiel` extra): registration, a 100-game conformance
check against the native referee, and smoke tests running a full game
through OpenSpiel's own `uniform_random` bot and a small clone()-based
simultaneous-move rollout search.

Headless self-play sweeps (seeded, invariant-checked, aggregated by
violation severity) run via `harness/selfplay.py:run_sweep` — see its
docstring for a `K`-game example against `agents/random_legal.py` and
`agents/greedy.py`.

## Repository map

```
src/simult_chess/
├── core/        # state algebra, geometry oracle, legality L(s,π), Φ
├── rules/       # RuleSet (frozen v1.1 + fingerprint), named variants, stage registry
├── invariants/  # WF/L/R/T/M checks + severity classification
├── referee/     # setup, observation channel, match loop, .scn game record + replay
├── agents/      # Agent protocol + random_legal, greedy
├── harness/     # seeded self-play sweeps, violation reports
├── ui/          # notation DSL, ASCII board render, hot-seat/human-vs-agent sessions
├── net/         # handshake, commit-reveal + match services, keepalive, TCP transport
├── solver/      # stage-matrix/LP layer (needs the solver extra): matrix_1ply
└── interop/     # OpenSpiel/pyspiel adapter (needs the openspiel extra)

docs/
├── simultaneous_chess_spec_v1.md   # ground-truth rule specification (spec)
└── INVARIANTS.md                    # validation-harness contract (inv)
```

## License

Code is licensed under [Apache-2.0](LICENSE). `docs/` (spec, invariants) is
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
