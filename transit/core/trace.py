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


def _timing() -> dict:
    """Where the wall clock went, split by cause.

    The per-step gaps are the point. Aggregates say "the model was slow";
    the sequence says WHY — latency that climbs 4s → 9s → 12s → 23s is the
    conversation growing, because every turn resends everything before it.
    A flat sequence at the same total means something else entirely.

    Imported lazily: trace.py sits below llm.py in the layering, and a
    top-level import would make the cycle real.
    """
    from transit.core import llm

    events = EVENTS.snapshot()
    summary = llm.timing_summary()

    if events:
        wall = float(events[-1]["t"]) - float(events[0]["t"])
        tool_seconds = sum(float(e.get("seconds", 0)) for e in events
                           if e["kind"] == "tool_call")
        previous = float(events[0]["t"])
        gaps = []
        for event in events:
            gaps.append({
                "step": event.get("step"),
                "before": event.get("tool") or "final answer",
                "gap_seconds": round(float(event["t"]) - previous, 1),
            })
            previous = float(event["t"])
        summary.update(
            wall_seconds=round(wall, 1),
            tool_seconds=round(tool_seconds, 2),
            # Anything not generating, sleeping, pacing or in a tool: our own
            # Python. Expected to be near zero; if it isn't, that's news.
            unaccounted_seconds=round(
                wall - summary["model_seconds"] - summary["wait_seconds"]
                - summary["throttle_seconds"] - tool_seconds, 1),
            step_gaps=gaps,
        )
    return summary


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
        "timing": _timing(),
        "answer": answer,
        **(extra or {}),
    }
    blob = json.dumps(payload, indent=2, default=str)
    path = TRACE_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(blob, encoding="utf-8")
    (TRACE_DIR / "latest.json").write_text(blob, encoding="utf-8")
    return path
