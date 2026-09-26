"""The brain (ARCHITECTURE.md §4): one serial decision loop per character, on the core's loop.

- ``loop``: :class:`~aivtube.brain.loop.Brain` (the §4.1 loop, speech tracking, operator
  control, watchdog), ``BrainDeps``, ``BrainSettings`` and ``brain_cfg`` (config → ``cfg``).
- ``decision``: ``DecisionSlot``, the single path to an LLM decision.
- ``intake``: transcripts, chat, support and operator input become stimuli (§4.2).
- ``arbiter``: rank/TTL selection, cadence, ``user_speaking`` gating, merging (§4.3).
- ``preempt``: the pure priority/interrupt table (§4.4).
- ``state``: the state machine, ``StateChanged`` and avatar poses (§4.5).
- ``reply``: LLM deltas → chunks → output gate → segments, and the Filtered path (§4.6, §4.10).
- ``prompt``: cache-stable prompt assembly and budgets (§4.8).
- ``history``: append-only history, heard text frozen at first render (§4.1, §4.8).
- ``tool_flow``: tool calls in call order and the one follow-up round (§4.9).
- ``background``: compaction, epoch flips on the inactive slot, slot files, episodes (§4.8, §6).
- ``idle``: the idle timer (§4.13).

Wiring (the app): build the parts, then ``Brain(BrainDeps(...))``; spawn ``brain.run()`` as a
critical supervised task and ``background.run()`` as a normal one; feed chat with
``brain.on_chat``; route operator commands to ``brain.control``; pass ``brain.on_auto_strict``
to ``LayeredSafetyGate(on_auto_strict=...)``.
"""
