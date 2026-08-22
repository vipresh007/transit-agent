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
    ui = (ROOT / "ui.py").read_text(encoding="utf-8")
    fields = set(plan_module.PlanResult.__dataclass_fields__) | {"flags"}
    for attribute in ("itinerary", "violations", "grounding", "research",
                      "no_schedule_data", "error", "flags"):
        check(f"PlanResult.{attribute} exists", attribute in fields)
        check(f"and the UI uses it", f".{attribute}" in ui)


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
               test_ui_only_reads_what_exists):
        fn()
    from _harness import PASSED
    print(f"\n{PASSED['n']} checks passed")
