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

import tools                                      # noqa: E402
from tools.journey import _shift, _with_walks     # noqa: E402

HAS_DB = Path(ROOT / "transit.db").exists()


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

    check("shifts backwards", _shift("08:03:31", -3), "08:00:31")
    check("keeps 24+ notation", _shift("25:23:00", 2), "25:25:00")
    check("crosses midnight upward", _shift("23:59:00", 2), "24:01:00")
    # A 00:02 departure minus a 5-minute walk produced '-1:57:00', which is
    # not a time and fails schema validation.
    check("clamps at zero rather than going negative",
          _shift("00:02:00", -5), "00:00:00")


def test_leg_labelling():
    section("journey leg labelling")

    option = {
        "type": "one_transfer",
        "legs": [
            {"route": "510", "headsign": "South", "from": "Spadina/Nassau",
             "from_stop": "8128", "to": "Spadina/King", "to_stop": "8126",
             "depart": "08:03:31", "arrive": "08:11:47"},
            {"route": "504", "headsign": "East", "from": "King/Spadina",
             "from_stop": "15648", "to": "Distillery Loop", "to_stop": "15462",
             "depart": "08:15:22", "arrive": "08:37:00"},
        ],
        "transfer_walk_m": 73, "transfer_walk_min": 2,
        "walk_to_stop_m": 257, "walk_from_stop_m": 200,
    }
    out = _with_walks(dict(option), "Kensington Market", "Distillery District")
    legs = out["legs"]

    # The model was left to infer walk endpoints and produced a final leg
    # reading "Distillery Loop -> Kensington Market".
    check("walk legs are inserted at both ends and the transfer", len(legs), 5)
    check("first leg starts at the user's origin", legs[0]["from"], "Kensington Market")
    check("last leg ends at the user's destination", legs[-1]["to"], "Distillery District")

    prev = None
    ordered = True
    for leg in legs:
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
        print("  (skipped: transit.db not built — run python load_gtfs.py)")
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
