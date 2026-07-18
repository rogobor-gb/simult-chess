"""LIGHT-profile self-play learning system (Phase 13b, docs/LEARNING_DESIGN.md).

Quarantined behind the optional ``[learn]`` extra (torch): only this package
imports torch. ``core``, ``rules``, and ``referee`` stay torch-free -- the same
quarantine pattern the ``oracle``/``solver``/``openspiel`` extras use for
``chess``/``scipy``/``pyspiel``, verified by
``tests/unit/test_learn_quarantine.py``.

The SM-MCTS hot path operates on the native ``simult_chess.core.types.State``
and ``core.phi.phi`` directly (design §2.3/§4): routing every node through the
pyspiel adapter's ``apply_actions`` would pay the ~35 ms O(pool^2) enumeration
the design explicitly rejects (§4.4), whereas a direct ``phi`` call is ~0.14 ms.
pyspiel is therefore not a runtime dependency of this package; adapter
compatibility (the design §2.5 intent) is validated separately by an
importorskip-guarded test.
"""
