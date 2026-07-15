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

Both CLIs accept `--agent {human,random,greedy}` (net) or `--agent
{random,greedy}` with `--human {white,black}` (ui). Programs are entered in a
short text DSL — see `src/simult_chess/ui/notation.py`'s module docstring for
the grammar (e.g. `Nf3`, `e4=Q`, `O-O`, `e3 def d4`, `cancel 0`).

## Tests and sweeps

```bash
pytest                      # full suite (unit + property), fast
pytest -m "not slow"        # excludes exhaustive/high-volume checks
scripts/check.sh            # ruff + mypy --strict + the fast pytest subset
```

`tests/property/` holds the metamorphic suite (inv M1–M4: purity, internal
order-independence, χ-color-swap equivariance) and the geometry
cross-validation against python-chess (needs the `oracle` extra; skips
cleanly without it).

Headless self-play sweeps (seeded, invariant-checked, aggregated by
violation severity) run via `harness/selfplay.py:run_sweep` — see its
docstring for a `K`-game example against `agents/random_legal.py` and
`agents/greedy.py`.

## Repository map

```
src/simult_chess/
├── core/        # state algebra, geometry oracle, legality L(s,π), Φ
├── rules/       # RuleSet — every [K] (convention-tied) invariant's parameter
├── invariants/  # WF/L/R/T/M checks + severity classification
├── referee/     # standard setup, commit-reveal observation channel, match loop
├── agents/      # Agent protocol + random_legal, greedy
├── harness/     # seeded self-play sweeps, violation reports
├── ui/          # notation DSL, ASCII board render, hot-seat/human-vs-agent sessions
├── net/         # commit-reveal protocol, asyncio TCP transport, online session
└── solver/      # stage-matrix/LP layer (needs the solver extra): matrix_1ply

docs/
├── simultaneous_chess_spec_v1.md   # ground-truth rule specification (spec)
└── INVARIANTS.md                    # validation-harness contract (inv)
```

## License

Code is licensed under [Apache-2.0](LICENSE). `docs/` (spec, invariants) is
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
