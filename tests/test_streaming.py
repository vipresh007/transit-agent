"""Live observation, and the refactor that made a second front end possible.

    python tests/test_streaming.py

Streamlit itself isn't tested here — it isn't installed on CI and a browser
test would cost far more than it catches. What IS tested is everything the UI
depends on and could silently break:

  - observers see events as they happen, not at the end
  - an observer that raises cannot take the run down with it
  - notify() does NOT pollute the trace, because timing derives per-step gaps
    from consecutive events
  - plan() returns instead of printing, so the CLI and the UI share one
    pipeline rather than two that drift
"""

import sys
from pathlib import Path

from _harness import calls, check, clean_env, install_fake_openai, says, section

install_fake_openai()
clean_env()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from transit.core import trace                              # noqa: E402

try:
    from transit.pipeline import plan as plan_module        # noqa: E402
    HAVE_PYDANTIC = True
except ImportError as exc:                                  # pragma: no cover
    print(f"  (plan tests skipped: {exc})")
    plan_module, HAVE_PYDANTIC = None, False


def test_observers_see_events_live():
    section("observers")

    trace.reset()
    seen = []
    trace.subscribe(seen.append)
    try:
        trace.event("tool_call", tool="geocode", seconds=0.2)
        # The point of the whole mechanism: visible NOW, not after the run.
        check("the observer saw it immediately", len(seen), 1)
        check("with the payload", seen[0]["tool"], "geocode")
        check("and a timestamp", "t" in seen[0])

        trace.event("tool_call", tool="plan_journey", seconds=2.5)
        check("and keeps seeing them", len(seen), 2)
    finally:
        trace.unsubscribe(seen.append if False else trace._OBSERVERS[0])

    trace.event("tool_call", tool="after", seconds=0)
    check("unsubscribing stops delivery", len(seen), 2)


def test_a_broken_observer_cannot_kill_a_run():
    section("an observer is a guest, not a partner")

    trace.reset()
    good = []

    def explodes(_event):
        raise RuntimeError("the progress bar fell over")

    trace.subscribe(explodes)
    trace.subscribe(good.append)
    try:
        # A UI crash must not become an agent crash. Watching a thing is not
        # a licence to break it.
        trace.event("tool_call", tool="geocode", seconds=0.1)
        check("the run continued", True)
        check("the working observer still got it", len(good), 1)
        check("the broken one was dropped", explodes not in trace._OBSERVERS)

        trace.event("tool_call", tool="again", seconds=0.1)
        check("and stays dropped", len(good), 2)
    finally:
        trace.unsubscribe(good.append)
        trace.unsubscribe(explodes)


def test_notify_does_not_pollute_the_trace():
    section("live signals vs the durable record")

    trace.reset()
    seen = []
    trace.subscribe(seen.append)
    try:
        trace.notify("tool_start", tool="plan_journey", step=3)
        # Recording starts as well as finishes would double the event count
        # and halve every per-step gap scripts/timing.py computes. The live
        # view and the record answer different questions.
        check("observers were told", len(seen), 1)
        check("but nothing was recorded", len(trace.EVENTS.snapshot()), 0)

        trace.event("tool_call", tool="plan_journey", seconds=2.5)
        check("a real event is both", len(seen), 2)
        check("and recorded", len(trace.EVENTS.snapshot()), 1)
    finally:
        trace.unsubscribe(seen.append)


def test_plan_returns_instead_of_printing():
    section("one pipeline, two front ends")
    if not HAVE_PYDANTIC:
        print("  (skipped: pydantic not installed)")
        return

    # The UI first duplicated research/structure/verify by hand. Two copies of
    # a pipeline is two places to fix every bug, and the copy nobody runs from
    # the terminal is the one that rots.
    check("plan() exists for callers who don't want stdout",
          callable(getattr(plan_module, "plan", None)))
    check("_plan() is the printing wrapper",
          callable(getattr(plan_module, "_plan", None)))

    result = plan_module.PlanResult(
        question="q", research="notes",
        prefs=plan_module.constraints.Preferences(), remembered=[])
    check("a fresh result carries no flags", result.flags, [])

    result.no_schedule_data = True
    check("no retrieved times is surfaced", result.flags, ["UNVERIFIED TIMES"])

    result.truncated = True
    result.repeats = 2
    check("flags accumulate", result.flags,
          ["UNVERIFIED TIMES", "TRUNCATED", "2 blocked repeats"])

    # Flags belong to the RESULT, not to whoever renders it — otherwise the
    # CLI and the UI can disagree about whether an itinerary is trustworthy.
    result.error = "structuring failed"
    check("a failure outranks the rest", result.flags[0], "FAILED")


def test_ui_only_reads_what_exists():
    section("the UI's assumptions about PlanResult")
    if not HAVE_PYDANTIC:
        print("  (skipped: pydantic not installed)")
        return

    # Cheap guard against renaming a field in plan.py and discovering it via
    # a traceback in the browser.
    # Both files: the pure logic moved into view.py, so checking ui.py alone
    # would report a field as unused the moment it was properly factored out.
    front_end = ((ROOT / "ui.py").read_text(encoding="utf-8")
                 + (ROOT / "transit" / "pipeline" / "view.py").read_text(encoding="utf-8"))
    fields = set(plan_module.PlanResult.__dataclass_fields__) | {"flags"}
    for attribute in ("itinerary", "violations", "grounding", "research",
                      "no_schedule_data", "error", "flags"):
        check(f"PlanResult.{attribute} exists", attribute in fields)
        check(f"and the front end uses it", f".{attribute}" in front_end)


def test_view_helpers_match_the_apis_they_call():
    section("the UI's pure logic, where a test can reach it")

    from transit.pipeline import view
    from transit.tools import memory

    # THE BUG THIS EXISTS FOR. The sidebar did `memory.load().items()`, but
    # load() returns (preferences, notes). It crashed on first launch, and no
    # test could have caught it: ui.py imports streamlit at module scope, so
    # it cannot be imported at all. Untestable code is untested code.
    loaded = memory.load()
    check("memory.load() returns a pair", isinstance(loaded, tuple))
    check("preferences are a dict", isinstance(loaded[0], dict))
    check("notes are a list", isinstance(loaded[1], list))

    rows = view.remembered_rows()
    check("remembered_rows() returns rows", isinstance(rows, list))
    check("each row is (label, value, forgettable)",
          all(len(r) == 3 and isinstance(r[2], bool) for r in rows))
    # Preferences are enforced; notes are only shown to the model. Offering to
    # "forget" a note would imply it had been changing journeys.
    check("notes are never forgettable",
          [f for label, _, f in rows if label == "note"], [])


class _FakeResult:
    """badge_values only reads attributes, so this needs no pydantic."""

    def __init__(self, **kw):
        self.violations, self.no_schedule_data, self.grounding = [], False, {}
        self.__dict__.update(kw)


def test_badges_are_decided_once():
    section("badge values")

    from transit.pipeline import view

    result = _FakeResult(grounding={"coverage": 1.0})
    badges = view.badge_values(result)
    check("a clean result says verified", badges["Schedule"], "verified")
    check("and times came from the feed", badges["Times"], "from the feed")
    check("and grounding is a percentage", badges["Grounding"], "100%")

    result.no_schedule_data = True
    # The distinction the whole project keeps relearning: unfounded is not the
    # same as wrong, and it has to be said out loud or it reads as fine.
    check("no retrieved times says ESTIMATED",
          view.badge_values(result)["Times"], "ESTIMATED")

    result.grounding = {}
    check("missing grounding is a dash, not 0%",
          view.badge_values(result)["Grounding"], "—")


def test_counters_reset_between_runs():
    section("a long-lived process must not accumulate")

    from transit.core import cache, llm

    # The CLI dies after one run, so module-level counters were fine. Streamlit
    # keeps the process alive for hours, so without a reset the second question
    # reports the first one's requests, tokens and seconds added to its own —
    # totals that only ever climb, and a "slowest call" from an hour ago.
    llm.USAGE.update(n=7, prompt_tokens=1234, completion_tokens=99)
    llm.TIMING.update(model_seconds=42.0, wait_seconds=13.0, latencies=[1.0, 2.0])
    cache.STATS.update(hits=3, misses=4)

    llm.reset_run()

    check("requests zeroed", llm.USAGE["n"], 0)
    check("tokens zeroed", llm.USAGE["prompt_tokens"], 0)
    check("model seconds zeroed", llm.TIMING["model_seconds"], 0.0)
    check("waiting zeroed", llm.TIMING["wait_seconds"], 0.0)
    check("latencies cleared", llm.TIMING["latencies"], [])
    check("cache stats zeroed", cache.STATS["hits"], 0)


def test_concurrency_is_declared_not_guessed():
    section("parallelism is a fact, not an inference")

    from transit.core import llm, trace

    # Inferring "concurrent" from buckets-exceed-wall-clock called a
    # single-threaded plan.py run 1.7x parallel. The buckets covered the whole
    # pipeline; the wall clock covered only the last agent.run(), because
    # agent.run() resets the trace each time. Two scopes, compared as one.
    llm.reset_run()
    llm.TIMING.update(model_seconds=600.0, wait_seconds=0.0)
    events = [{"kind": "tool_call", "t": 1000.0, "seconds": 1.0, "tool": "a"},
              {"kind": "final", "t": 1100.0}]

    sequential = trace._timing(events, wall_seconds=100.0, concurrent=False)
    check("buckets over the clock alone prove nothing",
          sequential["concurrent"], False)

    parallel = trace._timing(events, wall_seconds=100.0, concurrent=True)
    check("a caller that fans out says so", parallel["concurrent"], True)
    llm.reset_run()


def test_the_cli_delegates_to_the_pipeline():
    """Static: does _plan() call plan(), or has it grown its own copy again?

    Runs without pydantic, which matters — the environment this was written
    in couldn't import plan.py at all, and "I couldn't test it" is not the
    same as "it works".
    """
    section("the CLI has not re-grown a second pipeline")

    import ast
    tree = ast.parse((ROOT / "transit" / "pipeline" / "plan.py")
                     .read_text(encoding="utf-8"))
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in ("plan", "_plan", "main"):
        check(f"{name}() is defined", name in defined)

    cli = next(n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "_plan")
    called = {n.func.id for n in ast.walk(cli)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    check("_plan() calls plan()", "plan" in called)
    # If the CLI starts calling these directly again, the duplication is back.
    for name in ("structure", "repair"):
        check(f"_plan() does NOT call {name}() itself", name not in called)


if __name__ == "__main__":
    for fn in (test_observers_see_events_live,
               test_the_cli_delegates_to_the_pipeline,
               test_a_broken_observer_cannot_kill_a_run,
               test_notify_does_not_pollute_the_trace,
               test_plan_returns_instead_of_printing,
               test_ui_only_reads_what_exists,
               test_view_helpers_match_the_apis_they_call,
               test_badges_are_decided_once,
               test_counters_reset_between_runs,
               test_concurrency_is_declared_not_guessed):
        fn()
    from _harness import PASSED
    print(f"\n{PASSED['n']} checks passed")
