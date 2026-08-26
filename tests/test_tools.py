"""Tool logic that needs no network and no API key.

Split into two halves:
  - pure functions (time arithmetic, leg labelling, registry consistency)
  - SQL against transit.db, skipped automatically if the database isn't built

    python tests/test_tools.py
"""

import json
import os
import sys
from pathlib import Path

from _harness import check, section

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # tools resolve transit.db relative to the working directory

from transit import tools                                      # noqa: E402
from transit.tools.journey import _leg_dicts               # noqa: E402
from transit import paths

HAS_DB = paths.TRANSIT_DB.exists()


def test_registry():
    section("registry consistency")

    fns = set(tools.TOOL_FUNCTIONS)
    schemas = {t["function"]["name"] for t in tools.TOOL_SCHEMAS}
    check("every function has a schema and vice versa", fns, schemas)
    check("all registered entries are callable",
          all(callable(f) for f in tools.TOOL_FUNCTIONS.values()))
    # This set went stale twice, each time silently disabling a safety check.
    check("SCHEDULE_TOOLS only names real tools",
          tools.SCHEDULE_TOOLS <= fns)
    check("the journey planner counts as a schedule tool",
          "plan_journey" in tools.SCHEDULE_TOOLS)

    for schema in tools.TOOL_SCHEMAS:
        fn = schema["function"]
        check(f"{fn['name']} schema has a description",
              bool(fn.get("description")))


def test_time_arithmetic():
    section("GTFS time arithmetic")

    from transit.tools import raptor
    S, C = raptor._secs, raptor.civil

    check("round-trips a normal time", C(S("08:03:31")), "08:03:31")
    # GTFS runs past 24:00 for after-midnight service, and collapsing that to
    # 01:00 would file a Saturday-night trip on Saturday morning.
    check("keeps 24+ notation", C(S("25:23:00")), "25:23:00")
    check("crosses midnight upward", C(S("23:59:00") + 120), "24:01:00")
    check("shifts backwards", C(S("08:03:31") - 180), "08:00:31")
    # A 00:02 departure minus a 5-minute walk produced '-1:57:00', which is
    # not a time and fails schema validation. The lesson moved from _shift()
    # to civil() when RAPTOR replaced the pairwise search — the code went
    # away, the 00:02 departure did not.
    check("clamps at zero rather than going negative",
          C(S("00:02:00") - 300), "00:00:00")


def test_leg_labelling():
    section("journey leg labelling")

    # Same guarantee as before RAPTOR replaced the pairwise search: the model
    # must never be handed bare distances and left to write the labels. It
    # produced a final leg reading "Distillery Loop -> Kensington Market".
    # Only the producer changed, so the test follows it rather than retiring
    # with the old code — a test deleted alongside its subject takes the
    # lesson with it.
    from types import SimpleNamespace

    from transit.tools import raptor

    tt = SimpleNamespace(
        stop_ids=["8128", "8126", "15648", "15462"],
        stop_names=["Spadina/Nassau", "Spadina/King",
                    "King/Spadina", "Distillery Loop"],
        stop_pos=[(43.655, -79.401), (43.645, -79.396),
                  (43.645, -79.395), (43.650, -79.359)],
    )
    legs = [
        raptor.Leg("ride", "510", 0, 1, raptor._secs("08:03:31"),
                   raptor._secs("08:11:47"), "Union Station"),
        raptor.Leg("walk", None, 1, 2, raptor._secs("08:11:47"),
                   raptor._secs("08:13:47")),
        raptor.Leg("ride", "504", 2, 3, raptor._secs("08:15:22"),
                   raptor._secs("08:37:00"), "Distillery Loop"),
    ]
    out = _leg_dicts(tt, legs, "Kensington Market", "Distillery District",
                     257, 200)

    check("walk legs at both ends plus the transfer", len(out), 5)
    check("first leg starts at the user's origin",
          out[0]["from"], "Kensington Market")
    check("last leg ends at the user's destination",
          out[-1]["to"], "Distillery District")
    check("every ride carries a headsign",
          all(l.get("headsign") for l in out if l["mode"] == "transit"))
    check("no leg leaves the traveller unaccounted for",
          all(a["to"] == b["from"] for a, b in zip(out, out[1:])))

    prev, ordered = None, True
    for leg in out:
        if leg["arrive"] < leg["depart"] or (prev and leg["depart"] < prev):
            ordered = False
        prev = leg["arrive"]
    check("every leg is chronological and non-overlapping", ordered)


def test_input_validation():
    section("input validation")

    check("unknown POI category is rejected clearly",
          "Unknown category" in tools.find_pois(43.6, -79.4, "casino"))
    # The model called plan_journey with (0, 0) before geocoding anything.
    bad = tools.plan_journey(0, 0, 0, 0)
    check("placeholder coordinates are rejected",
          "not in the Toronto area" in bad)
    check("rejection says what to do first", "geocode first" in bad)


def test_against_database():
    section("queries against transit.db")
    if not HAS_DB:
        print("  (skipped: transit.db not built — run python scripts/load_gtfs.py)")
        return

    doc = tools.describe_transit_schema()
    check("schema doc includes live calendar rows", "service_id '1'" in doc)
    check("schema doc warns about time padding", "NOT CONSISTENTLY ZERO-PADDED" in doc)

    # Stops are direction-specific platforms; this is what the model kept
    # getting wrong, so the tool must surface it.
    near = json.loads(tools.find_nearby_stops(43.6552136, -79.4022604, 400))
    check("nearby stops are returned", len(near) > 0)
    check("each stop reports its direction", all("direction_ids" in s for s in near))
    check("each stop reports its headsigns", all("serves" in s for s in near))

    # SQL guardrails.
    check("writes are refused",
          "Only SELECT" in tools.query_transit("DROP TABLE routes"))
    check("multiple statements are refused",
          "one statement" in tools.query_transit("SELECT 1; DROP TABLE routes"))
    check("bad SQL returns a readable error, not an exception",
          tools.query_transit("SELECT nope FROM routes").startswith("SQL error"))
    check("the routes table survived all that",
          '"n": 233' in tools.query_transit("SELECT COUNT(*) AS n FROM routes"))


if __name__ == "__main__":
    for fn in (test_registry, test_time_arithmetic, test_leg_labelling,
               test_input_validation, test_against_database):
        fn()
    from _harness import PASSED
    print(f"\n{PASSED['n']} checks passed")
