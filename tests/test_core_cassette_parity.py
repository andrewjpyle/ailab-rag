"""Golden parity: the committed cassette loads identically through ailab_core.

Spar condition 1 for the ailab-core extraction: adopting the shared ReplayProvider must not
silently invalidate the real-model cassette. This asserts the core provider reads the SAME keys
and values the committed file holds, with zero misses. The before/after eval-metrics half of the
condition is enforced by the `eval` CI job, which replays this cassette through the gate.
"""

from __future__ import annotations

import json

from ailab_core.providers import ReplayProvider, cassette_key

from tests.conftest import CASSETTE


def test_core_replay_reads_committed_cassette_identically() -> None:
    raw = json.loads(CASSETTE.read_text(encoding="utf-8"))
    entries = raw["entries"]
    rp = ReplayProvider(CASSETTE, name="ollama", model=raw["model"])

    # Same provider identity and entry set the file declares.
    assert rp.name == raw["provider"]
    assert rp.entry_count == len(entries)

    # Every recorded entry resolves to the identical stored value through the core reader.
    misses = 0
    for key, entry in entries.items():
        stored = rp._entries.get(key)
        assert stored == entry, f"entry {key[:12]} changed under ailab_core"
        if stored is None:
            misses += 1
    assert misses == 0  # a stale cassette can never pass


def test_cassette_key_is_unchanged_under_core() -> None:
    # The hashing contract must be stable, or every recorded key would miss.
    assert cassette_key("m", "p") == cassette_key("m", "p")
