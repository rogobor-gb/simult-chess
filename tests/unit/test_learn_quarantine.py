"""Torch-quarantine test (Phase 13b, docs/LEARNING_DESIGN.md): the engine's
import graph -- ``core``/``rules``/``referee`` and the stdlib agents/harness --
must never pull in torch, exactly as the ``openspiel``/``solver`` extras keep
``pyspiel``/``scipy`` out of the engine.

Run in a fresh subprocess so a torch import elsewhere in the pytest session
cannot mask a real leak: we assert torch is absent from ``sys.modules`` after
importing the engine, which only holds if none of these modules imports it
transitively.
"""

from __future__ import annotations

import subprocess
import sys

_ENGINE_MODULES = (
    "simult_chess.core.phi",
    "simult_chess.core.legality",
    "simult_chess.core.geometry",
    "simult_chess.core.collision",
    "simult_chess.rules.ruleset",
    "simult_chess.rules.registry",
    "simult_chess.referee.match",
    "simult_chess.referee.setup",
    "simult_chess.harness.selfplay",
    "simult_chess.agents.candidates",
    "simult_chess.agents.random_legal",
    "simult_chess.agents.greedy",
)


def test_engine_import_graph_is_torch_free() -> None:
    imports = "; ".join(f"import {module}" for module in _ENGINE_MODULES)
    code = (
        f"{imports}; import sys; "
        "leaked = sorted(m for m in sys.modules "
        "if m == 'torch' or m.startswith('torch.')); "
        "assert not leaked, 'torch leaked into the engine import graph: ' "
        "+ repr(leaked)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"engine import graph is not torch-free (or failed to import):\n"
        f"{result.stdout}\n{result.stderr}"
    )
