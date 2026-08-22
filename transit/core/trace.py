"""Run tracing: a full JSON record of what the agent did.

Reading stderr scrollback is fine for a 3-step run and useless for a 12-step
one, and it loses the thing you most want later: the exact tool results the
model was reasoning over. A trace file is greppable, diffable between runs,
and you can hand it to someone else.

traces/latest.json always points at the most recent run.
"""

import json
import os
import time
from pathlib import Path

from transit.core.threadstate import ThreadLocalList

TRACE_DIR = Path(os.getenv("TRACE_DIR", "traces"))
# Thread-local: stage 9 runs several agents at once, and a shared list
# would interleave their events into one unusable trace.
EVENTS = ThreadLocalList()


def reset() -> None:
    EVENTS.clear()


def event(kind: str, **fields) -> None:
    EVENTS.append({"kind": kind, "t": round(time.time(), 3), **fields})


def write(
    question: str,
    answer: str = "",
    *,
    provider: str,
    model: str,
    usage: dict,
    cache_stats: dict,
    flags: dict,
    extra: dict | None = None,
) -> Path:
    TRACE_DIR.mkdir(exist_ok=True)
    payload = {
        "question": question,
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provider": provider,
        "model": model,
        "usage": dict(usage),
        "cache": dict(cache_stats),
        # Sets aren't JSON-serialisable; flags carry sets of tool names.
        "flags": {
            k: (sorted(v) if isinstance(v, set) else v) for k, v in flags.items()
        },
        "events": EVENTS.snapshot(),
        "answer": answer,
        **(extra or {}),
    }
    blob = json.dumps(payload, indent=2, default=str)
    path = TRACE_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(blob, encoding="utf-8")
    (TRACE_DIR / "latest.json").write_text(blob, encoding="utf-8")
    return path
