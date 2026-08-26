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

import json
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


class _Leg:
    def __init__(self, mode, route, origin, destination, minutes=5,
                 depart="08:00:00", arrive="08:05:00"):
        self.mode, self.route = mode, route
        self.origin, self.destination = origin, destination
        self.duration_min = minutes
        self.depart, self.arrive = depart, arrive


class _Itin:
    def __init__(self, legs):
        self.legs = legs


def test_map_geometry():
    section("the journey as map geometry")

    from pathlib import Path as _P

    from transit import paths as _paths
    from transit.pipeline import view

    if not _paths.TRANSIT_DB.exists():
        print("  (skipped: transit.db not built)")
        return

    # Real stops resolve from the feed; neighbourhood names do not, because
    # GTFS stop names are intersections. Both cases must be handled, and the
    # unresolved ones must be REPORTED rather than silently dropped — a map
    # missing its start looks like a bug unless it says why.
    itinerary = _Itin([
        _Leg("walk", None, "Kensington Market",
             "Spadina Ave at Nassau St South Side (8128)", 3),
        _Leg("streetcar", "510", "Spadina Ave at Nassau St South Side",
             "Spadina Ave at King St West", 8),
        _Leg("streetcar", "504", "King St West at Spadina Ave East Side",
             "Distillery Loop", 21),
    ])
    layers = view.map_layers(itinerary)

    check("resolved stops become points", len(layers["points"]) >= 3)
    check("every point has coordinates",
          all("lat" in p and "lon" in p for p in layers["points"]))
    check("the id suffix doesn't create a duplicate point",
          len({p["name"] for p in layers["points"]}), len(layers["points"]))
    check("legs with both ends known become paths", len(layers["paths"]), 2)
    check("a walk is dashed",
          [p["dashed"] for p in layers["paths"]], [False, False])
    check("streetcar legs are red",
          layers["paths"][0]["colour"], list(view.MODE_COLOUR["streetcar"]))
    check("neighbourhood names are named, not dropped",
          layers["unresolved"], ["Kensington Market"])

    # A paraphrased platform name still gets a pin. The model wrote
    # "Yorkdale Station (northbound platform)" where the feed says
    # "Yorkdale Station - Northbound Platform"; the two platforms are metres
    # apart so for a MAP either is right. This fallback is deliberately
    # absent from the departure check — approximating which side of a
    # platform to draw a dot on is cartography, approximating which platform
    # a train leaves from is a claim someone acts on.
    check("a paraphrased platform still resolves for the map",
          view.locate("Yorkdale Station (northbound platform)") is not None)
    check("but an invented stop does not",
          view.locate("Line 1 subway platform (northbound)"), None)

    camera = view.viewport(layers["points"])
    check("the viewport centres on Toronto",
          43.4 < camera["latitude"] < 44.0 and -79.8 < camera["longitude"] < -79.0)
    check("and zooms in rather than out", 10.0 <= camera["zoom"] <= 15.0)
    check("no points means no camera", view.viewport([]), None)

    # A coordinate outside Toronto is a mis-geocode, and one bad point
    # stretches the viewport across an ocean. Refusing it beats plotting it.
    check("a point in another country is rejected", view._in_toronto(34.7, 135.5), False)
    check("a Toronto point is accepted", view._in_toronto(43.65, -79.38))


def test_route_geometry():
    section("real track geometry, and honest fallback")

    from transit import paths as _paths
    from transit.pipeline import view

    if not _paths.TRANSIT_DB.exists():
        print("  (skipped: transit.db not built)")
        return

    if not view.has_shapes():
        # Optional table. Saying so beats a silent skip — the map still works
        # without it, just with straight lines.
        print("  (skipped: shapes not loaded — run scripts/load_shapes.py)")
        return

    pts = view.leg_shape("510", "Spadina Ave at Nassau St South Side (8128)",
                         "Spadina Ave at King St West")
    check("a real leg gets real geometry", pts is not None and len(pts) > 2)
    check("points are [lon, lat] pairs",
          all(len(p) == 2 for p in pts))
    check("and they stay in Toronto",
          all(view._in_toronto(lat, lon) for lon, lat in pts))

    # The whole point of slicing by shape_dist_traveled rather than snapping
    # to the nearest point: the geometry must START at the boarding stop, not
    # wherever the route happens to pass closest.
    start = view.locate("Spadina Ave at Nassau St South Side (8128)")
    first_lon, first_lat = pts[0]
    check("the line starts at the boarding stop",
          abs(first_lat - start[0]) < 0.002 and abs(first_lon - start[1]) < 0.002)

    # Geometry must not depend on which service day the leg falls on. This
    # defaulted to service_id "1" and so drew nothing for the 304, whose trips
    # are all service 3 — the map showed one line of a two-line journey while
    # 1,917 shape points sat unused. Rails don't move on Sundays.
    night = view.leg_shape("304", "King St West at Spadina Ave East Side",
                           "Distillery Loop")
    check("a leg on a non-weekday service still gets geometry",
          night is not None and len(night) > 2)
    check("it is still real track, not a straight line",
          night is not None and len(night) > 10)
    # A leg BOARDING AT A TERMINUS has an empty shape_dist_traveled at its
    # first stop, because GTFS records distance travelled and at stop 1 none
    # has been. Excluding empty outright blanked the geometry for every such
    # leg: the 129 from Kennedy Station drew a straight line across
    # Scarborough with a 580-point shape sitting unused. Empty means zero
    # HERE and unknown everywhere else, which is why it isn't a COALESCE.
    terminus = view.leg_shape("129", "Kennedy Station - Platform B",
                              "Scarborough Centre Station")
    check("a leg boarding at a terminus still gets geometry",
          terminus is not None and len(terminus) > 10)

    # ...but asking for one service explicitly must still mean that service.
    check("an explicit service_id still filters",
          view.leg_shape("304", "King St West at Spadina Ave East Side",
                         "Distillery Loop", service_id="1") is None)

    check("a route that doesn't serve those stops has no shape",
          view.leg_shape("504", "Spadina Ave at Nassau St South Side",
                         "Spadina Ave at King St West"), None)
    check("neither does a nonexistent route",
          view.leg_shape("999", "A", "B"), None)

    # Falling back to a straight line is fine and must be LABELLED. An
    # approximate line is honest cartography; an unlabelled one invites the
    # reader to believe the route runs where it doesn't.
    itinerary = _Itin([
        _Leg("streetcar", "510", "Spadina Ave at Nassau St South Side",
             "Spadina Ave at King St West", 8),
        _Leg("walk", None, "Spadina Ave at King St West",
             "King St West at Spadina Ave East Side", 2),
    ])
    layers = view.map_layers(itinerary)
    exact = [p["exact"] for p in layers["paths"]]
    check("the transit leg is exact", exact[0], True)
    check("the walk is not, and says so", exact[1], False)
    check("an exact leg has more than two points",
          len(layers["paths"][0]["path"]) > 2)


def test_timeline_markup():
    section("the timeline, as markup a test can read")

    from transit.pipeline import view

    class _T(_Itin):
        feasible = True
        def risky_connections(self):
            return [(1, 3)]

    itinerary = _T([
        _Leg("walk", None, "Kensington Market", "Spadina at Nassau", 3),
        _Leg("streetcar", "510", "Spadina at Nassau", "Spadina at King", 8),
        _Leg("streetcar", "504", "King at Spadina", "Distillery Loop", 21),
    ])
    html = view.timeline_html(itinerary)

    check("one card per leg, plus arrival", html.count('class="leg'), 4)
    check("walks are marked", html.count("leg walking"), 1)
    check("the tight transfer is inline, not in a block above",
          html.count('class="gap"'), 1)
    check("and names the route you'd miss", "504" in html)
    check("the journey ends with an arrival card", "ARRIVE" in html)
    check("streetcars are red", view.HEX["streetcar"] in html)
    check("walks are grey", view.HEX["walk"] in html)

    # unsafe_allow_html means what it says, and stop names come from a public
    # feed while answers come from a model. Neither is trusted markup.
    nasty = _T([_Leg("walk", None, "<script>alert(1)</script>", "B", 2)])
    escaped = view.timeline_html(nasty)
    check("markup in a stop name is escaped", "<script>" not in escaped)
    check("and survives as text", "&lt;script&gt;" in escaped)


def test_stat_cards_show_severity():
    section("stat cards")

    from transit.pipeline import view

    clean = _FakeResult(grounding={"coverage": 1.0})
    html = view.stats_html(clean)
    check("a clean result is green", 'class="stat ok"' in html)

    # The threshold matches agent.MIN_GROUNDING: below it the loop itself
    # pushes back, so the badge must not look content when the run wasn't.
    check("0.9 grounding reads ok", view._grounding_tone(0.9), "ok")
    check("0.7 grounding is a warning", view._grounding_tone(0.7), "warn")
    check("0.3 grounding is bad", view._grounding_tone(0.3), "bad")
    check("missing grounding has no tone", view._grounding_tone(None), "")

    broken = _FakeResult(violations=[1, 2], no_schedule_data=True, grounding={})
    html = view.stats_html(broken)
    check("violations turn the card red", 'class="stat bad"' in html)
    check("and the count is shown", "2 problem(s)" in html)


def test_json_for_the_browser():
    section("what the web front end receives")

    from transit.pipeline import view

    class _T(_Itin):
        summary = "Take the 510 then the 504."
        feasible = True
        infeasible_reason = None
        caveats = ["Real-time delays not checked."]
        total_min = 43
        transfers = 1
        def risky_connections(self):
            return [(0, 3)]

    result = _FakeResult(
        question="how do I get there?", error=None, steps=5,
        itinerary=_T([
            _Leg("streetcar", "510", "Spadina at Nassau", "Spadina at King", 8,
                 "08:03:00", "08:11:00"),
            _Leg("streetcar", "504", "King at Spadina", "Distillery Loop", 21,
                 "08:15:00", "08:36:00"),
        ]),
        grounding={"coverage": 1.0, "claims": 12, "unsupported": []},
        flags=[])

    payload = view.result_to_dict(result)

    check("legs are serialised", len(payload["legs"]), 2)
    check("times are human-readable, not GTFS",
          payload["legs"][0]["depart"], "8:03 AM")
    check("each leg carries its own colour",
          payload["legs"][0]["colour"], view.HEX["streetcar"])

    # The warning rides ON the leg it concerns. Sending a separate list would
    # make the browser re-associate them by index, which is a chance to get
    # it wrong for no benefit.
    check("a tight transfer is attached to its leg",
          payload["legs"][0]["warning"] is not None)
    check("and the next leg is clean", payload["legs"][1]["warning"], None)

    # Every judgement is made in Python. If the browser decided what counted
    # as verified, the CLI and the web app could disagree about whether the
    # same itinerary was trustworthy — the worst possible disagreement.
    check("tone is decided server-side", payload["tone"]["Schedule"], "ok")
    check("stats are decided server-side",
          payload["stats"]["Times"], "from the feed")
    check("caveats come through", len(payload["caveats"]), 1)
    check("the JSON survives a round trip",
          json.loads(json.dumps(payload, default=str))["legs"][0]["route"], "510")

    # An infeasible answer must serialise too — that's a real outcome, not an
    # error, and the browser needs enough to say so.
    class _No(_Itin):
        summary = "No route without buses."
        feasible = False
        infeasible_reason = "Scarborough Town Centre is bus-only."
        caveats = []
        total_min = 0
        transfers = 0
        def risky_connections(self):
            return []

    blocked = view.result_to_dict(_FakeResult(
        question="q", error=None, steps=3, itinerary=_No([]),
        grounding={}, flags=["NO ROUTE"]))
    check("an infeasible result still serialises", blocked["feasible"], False)
    check("with the reason", "bus-only" in blocked["infeasible_reason"])
    check("and no map to draw", blocked["map"]["points"], [])


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


def test_a_finished_run_survives_an_unwritable_trace_dir():
    section("the log is not the product")

    import tempfile

    from transit.core import trace as trace_module

    # By the time trace.write() runs, the model requests are spent and the
    # answer exists. Raising here throws away a finished result to report
    # that we could not file the paperwork about it. Found by mounting
    # traces/ read-only into the container: a correct itinerary reached the
    # browser as "OSError: [Errno 30] Read-only file system".
    #
    # WHY THIS PATCHES write_text INSTEAD OF CHMODDING A DIRECTORY, which is
    # what it did first: os.chmod on Windows only toggles a read-only flag on
    # FILES. Directories stay writable, so the setup silently did nothing,
    # the write succeeded, and the test reported a passing behaviour as a
    # bug in the code. A test whose setup quietly fails is indistinguishable
    # from a test that found something — the same two-states-look-identical
    # problem this suite exists to catch, arriving from inside the suite.
    #
    # Patching the call makes the failure the real one (an OSError from the
    # write) on every platform, and cannot half-work.
    original_write = Path.write_text
    denied = OSError(30, "Read-only file system")

    def refuse(self, *args, **kwargs):
        raise denied

    with tempfile.TemporaryDirectory() as tmp:
        saved_dir = trace_module.TRACE_DIR
        trace_module.TRACE_DIR = Path(tmp)
        Path.write_text = refuse
        try:
            path = trace_module.write(
                "a question", "an answer", provider="test", model="test",
                usage={}, cache_stats={}, flags={})
            check("write() returns instead of raising", path is not None)
        except OSError as exc:
            check("write() did not raise OSError", False, f"raised {exc}")
        finally:
            Path.write_text = original_write
            trace_module.TRACE_DIR = saved_dir

        check("and nothing was actually written",
              not list(Path(tmp).iterdir()))

    # The guard must not swallow a working write. A handler that turns every
    # save into a no-op would pass the test above and lose every trace.
    with tempfile.TemporaryDirectory() as tmp:
        saved_dir = trace_module.TRACE_DIR
        trace_module.TRACE_DIR = Path(tmp)
        try:
            path = trace_module.write(
                "a question", "an answer", provider="test", model="test",
                usage={}, cache_stats={}, flags={})
            check("a writable directory still gets the trace", path.exists())
            check("and latest.json alongside it",
                  (Path(tmp) / "latest.json").exists())
        finally:
            trace_module.TRACE_DIR = saved_dir


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
               test_concurrency_is_declared_not_guessed,
               test_map_geometry,
               test_route_geometry,
               test_timeline_markup,
               test_stat_cards_show_severity,
               test_json_for_the_browser,
               test_a_finished_run_survives_an_unwritable_trace_dir):
        fn()
    from _harness import PASSED
    print(f"\n{PASSED['n']} checks passed")
