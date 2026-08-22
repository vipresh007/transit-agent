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


# Live observers. A trace is written at the END of a run, which is the wrong
# time for anything watching one happen — a 60-second plan is 60 seconds of
# nothing followed by everything at once. Observers see each event as it
# occurs, without the pipeline knowing or caring who is listening.
_OBSERVERS: list = []


def subscribe(callback) -> None:
    """Call `callback(event_dict)` for every event until unsubscribed."""
    _OBSERVERS.append(callback)


def unsubscribe(callback) -> None:
    if callback in _OBSERVERS:
        _OBSERVERS.remove(callback)


def notify(kind: str, **fields) -> None:
    """Tell observers something happened, WITHOUT recording it.

    Two different jobs, deliberately separated. The trace is a durable record
    for later analysis; observers are a live view. "plan_journey is running
    right now" matters enormously to someone watching and not at all to
    someone reading the file afterwards — and recording it would corrupt the
    numbers, since timing derives per-step gaps from consecutive events and a
    start/finish pair would halve every gap.

    An observer that raises is dropped, not propagated. A progress bar must
    never be able to kill an agent run; watching a thing is not a licence to
    break it.
    """
    if not _OBSERVERS:
        return
    payload = {"kind": kind, "t": round(time.time(), 3), **fields}
    for callback in list(_OBSERVERS):
        try:
            callback(payload)
        except Exception:                                  # noqa: BLE001
            _OBSERVERS.remove(callback)


def reset() -> None:
    EVENTS.clear()


def event(kind: str, **fields) -> None:
    payload = {"kind": kind, "t": round(time.time(), 3), **fields}
    EVENTS.append(payload)
    notify(kind, **fields)


def _timing(events: list | None = None, wall_seconds: float | None = None,
            concurrent: bool = False) -> dict:
    """Where the wall clock went, split by cause.

    The per-step gaps are the point. Aggregates say "the model was slow";
    the sequence says WHY — latency that climbs 4s → 9s → 12s → 23s is the
    conversation growing, because every turn resends everything before it.
    A flat sequence at the same total means something else entirely.

    `events` and `wall_seconds` are overridable because EVENTS is THREAD-LOCAL.
    A crew run fills one list per subagent thread and then writes its trace
    from the main thread, whose list is empty — so the derived wall clock came
    out as 0.0s next to 321s of generating. Callers that fan out must hand in
    the merged events and the wall clock they measured themselves.

    Imported lazily: trace.py sits below llm.py in the layering, and a
    top-level import would make the cycle real.
    """
    from transit.core import llm

    events = EVENTS.snapshot() if events is None else events
    summary = llm.timing_summary()

    if events:
        wall = (wall_seconds if wall_seconds is not None
                else float(events[-1]["t"]) - float(events[0]["t"]))
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
            # Told, not inferred. Inferring it from "the buckets exceed the
            # wall clock" declared a single-threaded plan.py run to be 1.7x
            # parallel — the buckets covered the whole pipeline while the wall
            # clock covered only the last agent.run(). Two different scopes
            # compared as if they were one.
            concurrent=concurrent,
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
    events: list | None = None,
    wall_seconds: float | None = None,
    concurrent: bool = False,
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
        "events": EVENTS.snapshot() if events is None else events,
        "timing": _timing(events, wall_seconds, concurrent),
        "answer": answer,
        **(extra or {}),
    }
    blob = json.dumps(payload, indent=2, default=str)
    path = TRACE_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(blob, encoding="utf-8")
    (TRACE_DIR / "latest.json").write_text(blob, encoding="utf-8")
    return path
