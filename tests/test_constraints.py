"""Constraint checking: is this itinerary physically and schedulably possible?

    python tests/test_constraints.py

Pydantic already checks shape. These checks are about the world: does that
departure exist, can a person walk that far in that time, is one minute
enough to change vehicles. Schedule-dependent checks skip automatically if
transit.db isn't built.
"""

import os
import sys
from pathlib import Path

from _harness import check, section

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from transit.verify import constraints as C          # noqa: E402
from transit import paths

HAS_DB = paths.TRANSIT_DB.exists()


# Lightweight stand-ins rather than the real Pydantic models. constraints.py
# only reads attributes, so this suite tests constraint logic in isolation and
# runs even where pydantic isn't installed. Coupling a unit test to a library
# it doesn't exercise is how suites become unrunnable.
class _Leg:
    def __init__(self, mode, route, origin, destination, depart, arrive):
        self.mode, self.route = mode, route
        self.origin, self.destination = origin, destination
        self.depart, self.arrive = depart, arrive


class _Itinerary:
    def __init__(self, legs):
        self.legs = legs

    @property
    def transfers(self):
        return max(0, sum(1 for l in self.legs if l.mode != "walk") - 1)


def itin(*legs, summary="test"):
    return _Itinerary(list(legs))


def ride(route, origin, dest, depart, arrive, mode="streetcar"):
    return _Leg(mode, route, origin, dest, depart, arrive)


def walk(origin, dest, depart, arrive):
    return _Leg("walk", None, origin, dest, depart, arrive)


def kinds(violations) -> list[str]:
    return sorted(v.kind for v in violations)


def test_transfers():
    section("transfer feasibility")

    tight = itin(
        ride("510", "Spadina Ave at Nassau St", "Spadina Ave at King St West",
             "08:03:31", "08:11:47"),
        ride("504", "King St West at Spadina Ave East Side", "Distillery Loop",
             "08:13:00", "08:35:00"),
    )
    v = [x for x in C.verify(tight, C.Preferences(min_transfer_min=5))
         if x.kind == "tight_transfer"]
    check("a 1-minute vehicle change is flagged", len(v), 1)
    check("the message says how much is needed", ">= 5 min" in v[0].fix)

    roomy = itin(
        ride("510", "A", "B", "08:03:31", "08:11:47"),
        ride("504", "C", "D", "08:20:00", "08:35:00"),
    )
    v = [x for x in C.verify(roomy, C.Preferences(min_transfer_min=5))
         if x.kind == "tight_transfer"]
    check("an 8-minute change is fine", v, [])

    # Waiting for a vehicle after walking isn't a transfer — flagging it
    # produced four spurious warnings on a comfortable itinerary.
    after_walk = itin(
        walk("Kensington Market", "Spadina Ave at Nassau St", "08:00:00", "08:03:00"),
        ride("510", "Spadina Ave at Nassau St", "B", "08:03:31", "08:11:47"),
    )
    v = [x for x in C.verify(after_walk, C.Preferences()) if x.kind == "tight_transfer"]
    check("walk-then-board is not a tight transfer", v, [])


def test_preferences():
    section("user preferences")

    j = itin(ride("510", "A", "B", "06:30:00", "07:00:00"))
    v = C.verify(j, C.Preferences(earliest_departure="09:00:00"))
    check("departing before the requested time is flagged",
          "too_early" in kinds(v))

    v = C.verify(j, C.Preferences(latest_arrival="06:45:00"))
    check("arriving after the deadline is flagged", "too_late" in kinds(v))

    three = itin(
        ride("510", "A", "B", "08:00:00", "08:10:00"),
        ride("504", "B", "C", "08:20:00", "08:30:00"),
        ride("501", "C", "D", "08:40:00", "08:50:00"),
    )
    v = C.verify(three, C.Preferences(max_transfers=1))
    check("too many transfers is flagged", "too_many_transfers" in kinds(v))

    bus = itin(ride("35", "A", "B", "08:00:00", "08:20:00", mode="bus"))
    v = C.verify(bus, C.Preferences(avoid_modes=["bus"]))
    check("an avoided mode is flagged", "avoided_mode" in kinds(v))

    check("preferences describe themselves for the prompt",
          "at most 1 transfer(s)" in C.Preferences(max_transfers=1).describe())


def test_against_schedule():
    section("checks against the real schedule")
    if not HAS_DB:
        print("  (skipped: transit.db not built)")
        return

    real = itin(ride("510", "Spadina Ave at Nassau St",
                     "Spadina Ave at King St West", "08:03:31", "08:11:47"))
    check("a real departure verifies clean",
          [v for v in C.verify(real, C.Preferences())
           if v.kind == "departure_not_scheduled"], [])

    # 29 seconds off. Looks right, isn't in the feed. This is the failure
    # grounding can't catch: a time that exists somewhere, just not here.
    nudged = itin(ride("510", "Spadina Ave at Nassau St",
                       "Spadina Ave at King St West", "08:04:00", "08:12:00"))
    v = C.verify(nudged, C.Preferences())
    check("a departure 29s off the schedule is caught",
          "departure_not_scheduled" in kinds(v))

    # Right time, wrong route — a copy-paste error across legs.
    swapped = itin(ride("504", "Spadina Ave at Nassau St", "Distillery Loop",
                        "08:03:31", "08:35:00"))
    v = C.verify(swapped, C.Preferences())
    check("a real time on the wrong route is caught",
          "departure_not_scheduled" in kinds(v))

    unknown = itin(ride("999", "Spadina Ave at Nassau St", "B",
                        "08:03:31", "08:11:47"))
    check("a nonexistent route is caught",
          "unknown_route" in kinds(C.verify(unknown, C.Preferences())))

    # 1093m in 2 minutes is 33 km/h.
    sprint = itin(walk("Spadina Ave at Nassau St", "Spadina Ave at King St West",
                       "08:00:00", "08:02:00"))
    v = [x for x in C.verify(sprint, C.Preferences()) if x.kind == "walk_too_fast"]
    check("an impossible walk is caught", len(v), 1)
    check("and it suggests a realistic duration", "min for this walk" in v[0].fix)

    strolling = itin(walk("Spadina Ave at Nassau St", "Spadina Ave at King St West",
                          "08:00:00", "08:15:00"))
    check("a realistic walk passes",
          [x for x in C.verify(strolling, C.Preferences())
           if x.kind == "walk_too_fast"], [])


def test_route_label_resolution():
    section("route labels as humans write them")
    if not HAS_DB:
        print("  (skipped: transit.db not built)")
        return

    conn = C._conn()
    try:
        # Exact matching flagged two correct legs on a real run and triggered
        # a full repair round to fix nothing. "510 Spadina" is short_name +
        # long_name; "504A" is a branch appearing in 990 headsigns but in no
        # route_short_name.
        for label, expected in [
            ("510", "510"),
            ("510 Spadina", "510"),
            ("504A King", "504"),
            ("504A", "504"),
            ("Line 1", "1"),
            ("Line 2 (Bloor - Danforth)", "2"),
        ]:
            check(f"{label!r} resolves to {expected!r}",
                  C.resolve_route(conn, label), expected)

        for label in ("999", "Purple Line", ""):
            check(f"{label!r} is genuinely unknown",
                  C.resolve_route(conn, label), None)

        # Resolution is deliberately permissive; the departure check is what
        # actually validates route + stop + time together.
        check("a real route at a stop it doesn't serve is still caught",
              C._departure_is_scheduled(conn, "510", "Distillery Loop",
                                        "09:01:00"), False)
    finally:
        conn.close()

    real = itin(ride("510 Spadina", "Spadina Ave at Nassau St",
                     "Spadina Ave at King St West", "08:03:31", "08:11:47"))
    check("a correctly-labelled leg raises no violation",
          [v for v in C.verify(real, C.Preferences()) if v.kind == "unknown_route"],
          [])


def test_stop_labels_carrying_their_id():
    section("stop labels as the tools hand them back")

    for label, expected in [
        ("Spadina Ave at Nassau St South Side (8128)",
         ("Spadina Ave at Nassau St South Side", "8128")),
        ("King St West at Spadina Ave East Side (15648)",
         ("King St West at Spadina Ave East Side", "15648")),
        ("Distillery Loop", ("Distillery Loop", None)),
        # Not every parenthesis is an id.
        ("Union Station (Bay St)", ("Union Station (Bay St)", None)),
        ("", ("", None)),
    ]:
        check(f"{label!r} splits correctly", C.split_stop_label(label), expected)

    if not HAS_DB:
        print("  (rest skipped: transit.db not built)")
        return

    conn = C._conn()
    try:
        # Both of these are REAL departures that got flagged on a live run.
        # find_nearby_stops returns "NAME (STOP_ID)", the model copied the
        # label verbatim, and `LIKE '%NAME (8128)%'` matched no stop_name —
        # so two correct legs cost a full repair round. Exactly the failure
        # resolve_route was written for, one level down.
        for route, label, t in (
            ("510", "Spadina Ave at Nassau St South Side (8128)", "08:21:31"),
            ("504", "King St West at Spadina Ave East Side (15648)", "08:39:22"),
        ):
            check(f"{route} at {t} with an id-suffixed label verifies",
                  C._departure_is_scheduled(conn, route, label, t))
            check(f"{route} at {t} verifies without the suffix too",
                  C._departure_is_scheduled(
                      conn, route, C.split_stop_label(label)[0], t))

        # Tolerating the suffix must not blunt the check.
        check("29 seconds off is still caught",
              C._departure_is_scheduled(
                  conn, "510", "Spadina Ave at Nassau St South Side (8128)",
                  "08:22:00"), False)
        check("the wrong route at a real time is still caught",
              C._departure_is_scheduled(
                  conn, "504", "Spadina Ave at Nassau St South Side (8128)",
                  "08:21:31"), False)
        check("a route that doesn't serve the stop is still caught",
              C._departure_is_scheduled(
                  conn, "510", "Distillery Loop (15462)", "08:21:31"), False)
        # A stale id shouldn't condemn an otherwise-correct leg.
        check("a bogus id falls back to the name",
              C._departure_is_scheduled(
                  conn, "510", "Spadina Ave at Nassau St South Side (99999)",
                  "08:21:31"))

        # Silent version of the same bug: an unparsed label returns no
        # coordinates, so the walk-speed check skips instead of firing.
        check("coordinates resolve through the suffix",
              C._stop_coords(conn, "Spadina Ave at Nassau St South Side (8128)")
              is not None)
    finally:
        conn.close()


def test_infeasibility_preflight():
    section("constraints that cannot be satisfied at all")
    if not HAS_DB:
        print("  (skipped: transit.db not built)")
        return

    # Scarborough Town Centre is served by seventeen bus routes and nothing
    # else — Line 3 RT closed in 2023 and isn't in the feed. Asked to avoid
    # buses, the model produced Line 3 from stale training knowledge and wrote
    # a confident 88-minute itinerary. 34 requests to reach a fabrication.
    warnings = C.preflight(C.Preferences(avoid_modes=["bus"]),
                           (43.7761341, -79.2584376))
    check("a bus-only destination is flagged up front", len(warnings), 1)
    check("the warning says the trip is impossible",
          "CANNOT be planned" in warnings[0])
    check("and warns against using a remembered line",
          "Line 3" in warnings[0])

    # Somewhere streetcars reach: no conflict.
    check("a reachable destination raises nothing",
          C.preflight(C.Preferences(avoid_modes=["bus"]),
                      (43.6503278, -79.3592001)), [])
    check("no avoid preference means no check",
          C.preflight(C.Preferences(), (43.7761341, -79.2584376)), [])


def test_reporting():
    section("violation messages")

    j = itin(ride("999", "A", "B", "06:00:00", "06:30:00"))
    v = C.verify(j, C.Preferences(earliest_departure="09:00:00"))
    text = C.report(v)
    check("report names the count", str(len(v)) in text)
    # Every violation must say what to change; "invalid" alone makes the agent
    # guess, and a repair loop is only as good as its error messages.
    check("every violation carries a fix", all(x.fix for x in v))
    check("clean itineraries say so",
          C.report([]), "No constraint violations.")


if __name__ == "__main__":
    for fn in (test_transfers, test_preferences, test_against_schedule,
               test_route_label_resolution, test_stop_labels_carrying_their_id,
               test_infeasibility_preflight,
               test_reporting):
        fn()
    from _harness import PASSED
    print(f"\n{PASSED['n']} checks passed")
