# Contributing

Thanks for your interest. This repository is the **rules engine** for a
simultaneous-move chess variant: a formal specification, a proven-correct
transition operator, an executable-invariant test harness, and the tooling
around them. It is deliberately small, strict, and spec-driven. Please read
this before opening a pull request.

## The one rule that matters most: the spec is ground truth

`docs/simultaneous_chess_spec_v1.md` (the **spec**) and `docs/INVARIANTS.md`
(the **invariants**) define the game. The code implements them; it never
invents, "improves", or silently reinterprets a rule.

- **Spec-first.** No implementation of a rule change lands before the formal
  ground truth is settled. A change to how the game *works* edits the spec and
  the invariants **first**, in the same PR, and the code follows.
- **Every rule cites its spec section** in its docstring; **every check cites
  its invariant ID** (`WF*`/`L*`/`R*`/`T*`/`M*`).
- **A variant is a `RuleSet` field or a swapped stage implementation
  (`rules/registry.py`), never a fork** of the operator. Rule-bearing defaults
  are frozen and identified by `RuleSet.fingerprint()`; changing one changes
  the fingerprint, which is intentional and must be deliberate (see the freeze
  block in `reports/campaign_v1.md`).

If you are unsure whether something is a rule change, it probably is — open an
issue first.

## Local setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # ruff, mypy, pytest, hypothesis
# add extras only if you touch those layers:
#   .[solver]   numpy + scipy   (stage-matrix / LP solver)
#   .[oracle]   python-chess    (geometry cross-validation, tests only)
#   .[openspiel] open_spiel      (pyspiel adapter)
#   .[learn]    torch + numpy    (self-play learning)
```

## The gate

Every change must pass the local gate, which is exactly what CI runs on your PR:

```bash
scripts/check.sh          # ruff, mypy --strict, pytest -m "not slow"
```

- **`ruff`** for style (PEP 8, import order) and **`mypy --strict`** for full
  type annotations — no exceptions merge with either failing.
- **`pytest`**. Add tests for new behaviour. Mark exhaustive/high-volume checks
  `@pytest.mark.slow` so they stay out of the fast gate but still run in the
  full suite.
- The **invariant harness must stay clean**: zero `S0`/`S1` violations over any
  sweep. A rule change that trips an invariant means the invariant needs
  updating in lockstep (spec-first), or the change is wrong.

## The extras quarantine

`simult_chess.core`, `simult_chess.rules`, and `simult_chess.referee` import
**only the standard library**. Every third-party dependency lives behind a
named optional extra (`solver`, `oracle`, `openspiel`, `learn`) and is imported
only inside the layer named for it. CI enforces this: a base job installs no
extras and must still pass, and dedicated tests assert the import graph
(e.g. `tests/unit/test_learn_quarantine.py`, `tests/unit/test_relay.py`). Do
not add an import that breaks the quarantine.

## Commits and pull requests

- **Small, focused commits split by concern** (a fix, a feature, a docs update
  are separate commits) rather than one large bundle.
- **Conventional-commit messages**: `feat(net): …`, `fix(core): …`,
  `docs(spec): …`, `test(...): …`, `ci: …`.
- A PR that changes a rule updates the spec and `docs/INVARIANTS.md` in the
  **same** PR.
- CI (ruff, mypy, the fast tests on each supported Python, plus the per-extra
  jobs) must be green. It is required for merge.

## Scope

What is the engine and what is an application built on top of it is written
down in [`docs/COLLABORATION.md`](docs/COLLABORATION.md). Engine contributions
go here, under this repository's licenses (code Apache-2.0, `docs/` CC BY 4.0).
