"""RAPTOR: does it find real journeys, and are they actually in the feed?

    python tests/test_raptor.py

A journey planner that returns something plausible is worse than one that
returns nothing, because nobody checks a plausible answer. So the central
test here does not ask "did it find a route" — it takes every leg it found
and looks the exact (route, stop, departure, arrival) up in stop_times. A
leg that isn't in the feed is a leg the traveller can't board.

Needs transit.db, and skips loudly without it.
"""

import sqlite3
import sys
import time
from pathlib import Path

from _harness import check, section

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from transit import paths                                   # noqa: E402
from transit.tools import raptor                            # noqa: E402

HAVE_DB = paths.TRANSIT_DB.exists()

# Kensington Market, Distillery District, Union, Yorkdale, Scarborough TC.
KENSINGTON = (43.6547, -79.4005)
DISTILLERY = (43.6503, -79.3592)
UNION = (43.6453, -79.3806)
YORKDALE = (43.7254, -79.4522)
SCARBOROUGH = (43.7764, -79.2573)

_tt = None


def table():
    global _tt
    if _tt is None:
        _tt = raptor.timetable("1")
    return _tt


def journey(origin, dest, after="08:00:00"):
    tt = table()
    o = [(i, raptor._walk_seconds(m))
         for i, m in tt.nearest(*origin, raptor.ACCESS_RADIUS_M)][:15]
    d = [(i, raptor._walk_seconds(m))
         for i, m in tt.nearest(*dest, raptor.ACCESS_RADIUS_M)][:15]
    return raptor.query(tt, o, d, raptor._secs(after))


def test_clock():
    section("GTFS clock")

    check("parses past midnight", raptor._secs("25:01:00"), 25 * 3600 + 60)
    check("renders past midnight", raptor.civil(25 * 3600 + 60), "25:01:00")
    # Callers subtract the walk to the first stop. Five minutes before 00:02
    # is negative, and '-1:57:00' is not a time.
    check("clamps below zero", raptor.civil(-300), "00:00:00")


def test_patterns_are_not_routes():
    section("a pattern is not a route_id")
    if not HAVE_DB:
        print("  SKIP — no transit.db")
        return

    tt = table()
    check("there are far more patterns than routes", len(tt.patterns) > 500)
    check("every pattern has at least two stops",
          all(len(p) >= 2 for p in tt.patterns))
    check("every pattern has at least one trip",
          all(len(t) > 0 for t in tt.pattern_trips))

    # Merging the 504's branches would let a scan board a trip that never
    # reaches the stop it expects next — wrong, not slow.
    for pattern, trips in zip(tt.patterns, tt.pattern_trips):
        if len(trips) > 1:
            check("trips on a pattern all have its length",
                  all(len(t) == len(pattern) for t in trips))
            break

    # The scan takes the first catchable trip and stops looking, which is
    # only valid if they're in departure order.
    unsorted = [i for i, trips in enumerate(tt.pattern_trips)
                if any(a[0][1] > b[0][1] for a, b in zip(trips, trips[1:]))]
    check("trips within a pattern are sorted by departure", not unsorted)


def test_finds_journeys_that_exist():
    section("journeys")
    if not HAVE_DB:
        print("  SKIP — no transit.db")
        return

    cases = [
        ("Union -> Yorkdale", UNION, YORKDALE),
        ("Kensington -> Distillery", KENSINGTON, DISTILLERY),
        # THE ONE THAT MOTIVATED ALL OF THIS. The pairwise search took 308
        # seconds on this and returned nothing, which the agent reported as
        # "No route found" — a claim about Toronto rather than about our code.
        ("Kensington -> Scarborough Town Centre", KENSINGTON, SCARBOROUGH),
    ]
    for name, origin, dest in cases:
        started = time.perf_counter()
        legs = journey(origin, dest)
        elapsed = time.perf_counter() - started
        check(f"{name}: found a journey", bool(legs))
        # 308s was the number to beat. One second is a generous ceiling that
        # still catches a rewrite that reintroduces the old behaviour.
        check(f"{name}: answered in under a second", elapsed < 1.0,
              True if elapsed < 1.0 else f"took {elapsed:.1f}s")
        if legs:
            check(f"{name}: rides at least one vehicle",
                  any(l.mode == "ride" for l in legs))


def test_every_leg_is_in_the_feed():
    section("the check that matters")
    if not HAVE_DB:
        print("  SKIP — no transit.db")
        return

    tt = table()
    conn = sqlite3.connect(paths.readonly_uri(paths.TRANSIT_DB), uri=True)
    try:
        for name, origin, dest, after in (
            ("Kensington -> Scarborough", KENSINGTON, SCARBOROUGH, "08:00:00"),
            ("Union -> Yorkdale", UNION, YORKDALE, "08:00:00"),
            # After-midnight service, where GTFS hours exceed 24 and every
            # naive time comparison in this project has broken at least once.
            ("late night", KENSINGTON, SCARBOROUGH, "23:30:00"),
        ):
            legs = journey(origin, dest, after)
            if not legs:
                check(f"{name}: found something to verify", False)
                continue

            for leg in legs:
                if leg.mode != "ride":
                    continue
                # Not "does this route serve these stops" — does ONE TRIP
                # leave the first at this exact second and reach the second
                # at that one. A journey assembled from two different trips
                # is a journey nobody can take.
                row = conn.execute(
                    """SELECT 1 FROM stop_times a
                       JOIN stop_times b ON b.trip_id = a.trip_id
                       JOIN trips t  ON t.trip_id = a.trip_id
                       JOIN routes r ON r.route_id = t.route_id
                       WHERE r.route_short_name = ? AND t.service_id = '1'
                         AND a.stop_id = ? AND b.stop_id = ?
                         AND substr('0' || a.departure_time, -8) = ?
                         AND substr('0' || b.arrival_time, -8) = ?
                         AND CAST(b.stop_sequence AS INT)
                             > CAST(a.stop_sequence AS INT)
                       LIMIT 1""",
                    (leg.route, tt.stop_ids[leg.from_stop],
                     tt.stop_ids[leg.to_stop],
                     raptor.civil(leg.depart), raptor.civil(leg.arrive))
                ).fetchone()
                check(f"{name}: {leg.route} {raptor.civil(leg.depart)} is a "
                      f"real trip", bool(row))
    finally:
        conn.close()


def test_journeys_hold_together():
    section("legs chain")
    if not HAVE_DB:
        print("  SKIP — no transit.db")
        return

    tt = table()
    legs = journey(KENSINGTON, SCARBOROUGH)
    check("found a journey", bool(legs))
    if not legs:
        return

    for a, b in zip(legs, legs[1:]):
        check("no teleporting between legs", a.to_stop == b.from_stop)
        # Boarding before the previous vehicle arrives is the classic
        # journey-planner bug, and it always looks fine in a summary.
        check("no boarding before arriving", b.depart >= a.arrive)

    for leg in legs:
        check(f"{leg.mode} leg does not end before it starts",
              leg.arrive >= leg.depart)
        if leg.mode == "walk":
            metres = raptor._metres(*tt.stop_pos[leg.from_stop],
                                    *tt.stop_pos[leg.to_stop])
            check("a transfer walk is within the radius",
                  metres <= raptor.TRANSFER_RADIUS_M + 1)
            check("and takes a believable time",
                  leg.arrive - leg.depart >= raptor.MIN_CHANGE_S)


def test_impossible_journeys_return_nothing():
    section("absence")
    if not HAVE_DB:
        print("  SKIP — no transit.db")
        return

    tt = table()
    # Lake Ontario. Returning None is correct; the tool above is responsible
    # for saying "not found" rather than "does not exist".
    far = [(i, raptor._walk_seconds(m))
           for i, m in tt.nearest(43.55, -79.30, raptor.ACCESS_RADIUS_M)]
    check("nowhere near a stop yields no candidates", not far)

    o = [(i, raptor._walk_seconds(m))
         for i, m in tt.nearest(*KENSINGTON, raptor.ACCESS_RADIUS_M)][:15]
    check("no destination means no journey",
          raptor.query(tt, o, [], raptor._secs("08:00:00")) is None)
    check("no origin means no journey",
          raptor.query(tt, [], o, raptor._secs("08:00:00")) is None)


def test_more_rounds_never_arrive_later():
    section("rounds")
    if not HAVE_DB:
        print("  SKIP — no transit.db")
        return

    tt = table()
    o = [(i, raptor._walk_seconds(m))
         for i, m in tt.nearest(*KENSINGTON, raptor.ACCESS_RADIUS_M)][:15]
    d = [(i, raptor._walk_seconds(m))
         for i, m in tt.nearest(*SCARBOROUGH, raptor.ACCESS_RADIUS_M)][:15]
    after = raptor._secs("08:00:00")

    arrivals = []
    for rounds in (1, 2, 3, 5):
        legs = raptor.query(tt, o, d, after, max_rounds=rounds)
        arrivals.append(legs[-1].arrive if legs else None)

    check("one round can't reach Scarborough from Kensington",
          arrivals[0] is None)
    check("more rounds eventually can", arrivals[-1] is not None)
    # Each extra round only adds options, so the arrival can improve or stay
    # put. Getting worse would mean the search is losing journeys it found.
    found = [a for a in arrivals if a is not None]
    check("adding rounds never makes the answer worse",
          all(b <= a for a, b in zip(found, found[1:])))


if __name__ == "__main__":
    for fn in (test_clock,
               test_patterns_are_not_routes,
               test_finds_journeys_that_exist,
               test_every_leg_is_in_the_feed,
               test_journeys_hold_together,
               test_impossible_journeys_return_nothing,
               test_more_rounds_never_arrive_later):
        fn()
    from _harness import PASSED
    print(f"\n{PASSED['n']} checks passed")
