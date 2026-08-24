"""Does TTC's real-time feed actually join to our database? Find out first.

    pip install gtfs-realtime-bindings requests
    python scripts/probe_rt.py

WHY THIS EXISTS BEFORE ANY FEATURE CODE.

The registry says TTC's GTFS-RT feed pairs with `SurfaceGTFS.zip`. Our
`transit.db` was built from a DIFFERENT file, `TTC Routes and Schedules
Data.zip`. Two feeds from the same agency describing overlapping service are
not obliged to use the same `trip_id`s, and nothing anywhere promises they do.

If they don't match, every real-time lookup returns nothing. And a lookup that
returns nothing renders as "no delays reported" — which is exactly what a
working feed looks like on a good day. That is this project's oldest failure
mode: two different states that render identically. A skipped test and a
passing test. A cache replay and a fast run. An empty join and a punctual bus.

So we measure the join rate before we build anything on top of it, and we
print the rate rather than a verdict, because "31% of trips matched" is a
finding and "real-time works" is a guess.

WHAT IT REPORTS

  * whether each of the three endpoints is reachable and what it returned
  * how many entities are in each feed
  * what fraction of RT trip_id / stop_id / route_id exist in transit.db
  * a handful of unmatched IDs side by side with ours, because the shape of
    the mismatch tells you whether it's a prefix, a different scheme, or a
    genuinely separate universe of identifiers
  * which route types showed up, to confirm the subway really is absent

It also writes the raw protobuf bytes to data/rt_sample/, so the test suite
can exercise the decoder offline forever after. A feed that only exists for
30 seconds cannot be a test fixture; a file can.

This script READS the database and never writes to it.
"""

from __future__ import annotations

import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transit import paths                                    # noqa: E402

FEEDS = {
    "vehicles": "https://bustime.ttc.ca/gtfsrt/vehicles",
    "trips": "https://bustime.ttc.ca/gtfsrt/trips",
    "alerts": "https://bustime.ttc.ca/gtfsrt/alerts",
}

SAMPLE_DIR = paths.DATA / "rt_sample"

# Enough to characterise a mismatch without dumping thousands of ids.
SHOW = 6


def _need(module: str, pip: str):
    try:
        return __import__(module)
    except ImportError:
        sys.exit(f"This probe needs {pip}:\n    pip install {pip}")


def fetch(name: str, url: str, requests) -> bytes | None:
    """Fetch one feed. Returns raw bytes, or None with the reason printed.

    Deliberately does not raise: one dead endpoint should not stop us
    measuring the other two.
    """
    try:
        started = time.perf_counter()
        response = requests.get(url, timeout=30,
                                headers={"User-Agent": "transit-agent/1.0 "
                                                       "(learning project)"})
        seconds = time.perf_counter() - started
    except Exception as exc:                                 # noqa: BLE001
        print(f"  {name:<9} UNREACHABLE  {type(exc).__name__}: {exc}")
        return None

    kind = response.headers.get("Content-Type", "?")
    print(f"  {name:<9} HTTP {response.status_code}  "
          f"{len(response.content):>9,} bytes  {seconds:4.1f}s  {kind}")

    if response.status_code != 200:
        return None
    if not response.content:
        print(f"  {name:<9} empty body — nothing to decode")
        return None
    # A protobuf feed starts with a field tag, never with '<' or '{'. Getting
    # HTML here means a login page or an error page dressed as a 200.
    if response.content[:1] in (b"<", b"{"):
        print(f"  {name:<9} looks like text, not protobuf: "
              f"{response.content[:80]!r}")
        return None
    return response.content


def decode(raw: bytes, gtfs_realtime_pb2):
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(raw)
    return feed


def db_ids(conn: sqlite3.Connection) -> tuple[set, set, set]:
    """Every identifier our database knows, as sets for O(1) membership."""
    trips = {r[0] for r in conn.execute("SELECT trip_id FROM trips")}
    stops = {r[0] for r in conn.execute("SELECT stop_id FROM stops")}
    routes = {r[0] for r in conn.execute("SELECT route_id FROM routes")}
    return trips, stops, routes


def rate(seen: set, known: set, label: str, sample_known: list) -> float:
    """Print a join rate and, if it's poor, show what the two sides look like.

    The samples matter more than the number. 0% with ids that look alike is a
    prefix problem; 0% with ids that look nothing alike is a different feed.
    """
    if not seen:
        print(f"  {label:<9} none present in the feed")
        return -1.0

    hit = seen & known
    pct = 100.0 * len(hit) / len(seen)
    print(f"  {label:<9} {len(hit):>6,} / {len(seen):>6,} matched  "
          f"{pct:5.1f}%")

    if pct < 99.0:
        missing = sorted(seen - known)[:SHOW]
        print(f"      feed says : {missing}")
        print(f"      we have   : {sample_known[:SHOW]}")
    return pct


def collision_check(conn: sqlite3.Connection, per_route: dict) -> float:
    """Of the stop_ids that "matched", how many mean the same physical stop?

    THIS IS THE MEASUREMENT THAT MATTERS, and the first version of this script
    didn't have it. It reported a 59.3% stop_id match rate and that number was
    almost entirely noise: both feeds number stops with small integers, so
    thousands of ids collide by arithmetic rather than by agreement. Feed route
    23 is Dawes Rd; its "matched" stops resolved to Bathurst St.

    A set-membership test answers "is this string present". It cannot answer
    "does this string mean the same thing", and those two questions look
    identical right up until you act on the answer.

    So: for each route the feed reports, intersect its stops with the stops OUR
    database puts on that same route. Agreement survives; coincidence doesn't.
    """
    same = other = 0
    for route_id, feed_stops in per_route.items():
        ours = {r[0] for r in conn.execute(
            "SELECT DISTINCT st.stop_id FROM stop_times st "
            "JOIN trips t ON t.trip_id = st.trip_id WHERE t.route_id = ?",
            (route_id,))}
        if not ours:
            continue
        for stop_id in feed_stops:
            if stop_id in ours:
                same += 1
            elif conn.execute("SELECT 1 FROM stops WHERE stop_id = ?",
                              (stop_id,)).fetchone():
                other += 1                      # exists, but somewhere else

    total = same + other
    if not total:
        print("  no feed stop_id exists in our stops table at all")
        return 0.0
    pct = 100.0 * same / total
    print(f"  {same:>6,} / {total:>6,} of the 'matched' stop_ids are actually "
          f"on that route   {pct:5.1f}%")
    if pct < 50:
        print(f"  {other:,} are number collisions — the same integer naming a "
              f"different stop.")
    return pct


def route_types(conn: sqlite3.Connection, route_ids: set) -> dict:
    """GTFS route_type: 0 tram/streetcar, 1 subway, 3 bus.

    The registry claims this feed is surface-only. Claims get checked.
    """
    names = {"0": "streetcar", "1": "subway", "2": "rail", "3": "bus"}
    counts: dict[str, int] = {}
    for rid in route_ids:
        row = conn.execute(
            "SELECT route_type, route_short_name FROM routes WHERE route_id = ?",
            (rid,)).fetchone()
        if row is None:
            counts["unknown to us"] = counts.get("unknown to us", 0) + 1
            continue
        label = names.get(str(row[0]), f"type {row[0]}")
        counts[label] = counts.get(label, 0) + 1
    return counts


def main() -> None:
    requests = _need("requests", "requests")
    try:
        from google.transit import gtfs_realtime_pb2
    except ImportError:
        sys.exit("This probe needs the protobuf definitions:\n"
                 "    pip install gtfs-realtime-bindings")

    if not paths.TRANSIT_DB.exists():
        sys.exit(f"No database at {paths.TRANSIT_DB}. "
                 f"Run scripts/load_gtfs.py first.")

    print("\nFETCHING  (three endpoints, no API key, ~30s of live data)\n")
    raws = {name: fetch(name, url, requests)
            for name, url in FEEDS.items()}

    if not any(raws.values()):
        sys.exit("\nNothing came back. Check the network, then the URLs.")

    # Save before decoding. If the decode crashes we still want the bytes:
    # this exact moment cannot be fetched again.
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for name, raw in raws.items():
        if raw:
            (SAMPLE_DIR / f"{name}.pb").write_bytes(raw)
    print(f"\n  saved raw feeds to {SAMPLE_DIR}  (offline test fixtures)")

    print("\nDECODING\n")
    feeds = {}
    for name, raw in raws.items():
        if not raw:
            continue
        try:
            feed = decode(raw, gtfs_realtime_pb2)
        except Exception as exc:                             # noqa: BLE001
            print(f"  {name:<9} DECODE FAILED  {type(exc).__name__}: {exc}")
            continue
        feeds[name] = feed
        stamp = feed.header.timestamp
        age = (datetime.now(timezone.utc)
               - datetime.fromtimestamp(stamp, timezone.utc)).total_seconds() \
            if stamp else float("nan")
        print(f"  {name:<9} {len(feed.entity):>6,} entities   "
              f"gtfs-rt v{feed.header.gtfs_realtime_version}   "
              f"published {age:.0f}s ago")

    # ---- gather every identifier the feed mentions ----------------------
    trip_ids: set = set()
    stop_ids: set = set()
    route_ids: set = set()
    delays: list = []
    no_trip_id = 0
    per_route: dict = {}          # route_id -> the stops the feed puts on it

    def entities(name: str):
        """Entities from one feed, or nothing if that feed didn't arrive."""
        feed = feeds.get(name)
        return feed.entity if feed is not None else []

    for entity in entities("trips"):
        if not entity.HasField("trip_update"):
            continue
        update = entity.trip_update
        if update.trip.trip_id:
            trip_ids.add(update.trip.trip_id)
        else:
            # Some agencies identify a trip by route + start time instead.
            # That changes how we'd join, so it's worth counting.
            no_trip_id += 1
        if update.trip.route_id:
            route_ids.add(update.trip.route_id)
        here = per_route.setdefault(update.trip.route_id, set())
        for stu in update.stop_time_update:
            if stu.stop_id:
                stop_ids.add(stu.stop_id)
                here.add(stu.stop_id)
            if stu.HasField("departure") and stu.departure.delay:
                delays.append(stu.departure.delay)

    for entity in entities("vehicles"):
        if not entity.HasField("vehicle"):
            continue
        vehicle = entity.vehicle
        if vehicle.trip.trip_id:
            trip_ids.add(vehicle.trip.trip_id)
        if vehicle.trip.route_id:
            route_ids.add(vehicle.trip.route_id)
        if vehicle.stop_id:
            stop_ids.add(vehicle.stop_id)

    # ---- the question this script exists to answer ----------------------
    conn = sqlite3.connect(paths.readonly_uri(paths.TRANSIT_DB), uri=True)
    known_trips, known_stops, known_routes = db_ids(conn)

    print("\nJOIN RATE against data/transit.db "
          f"({len(known_trips):,} trips, {len(known_stops):,} stops, "
          f"{len(known_routes):,} routes)\n")

    t = rate(trip_ids, known_trips, "trip_id", sorted(known_trips)[:SHOW])
    rate(stop_ids, known_stops, "stop_id", sorted(known_stops)[:SHOW])
    rate(route_ids, known_routes, "route_id", sorted(known_routes)[:SHOW])

    if no_trip_id:
        print(f"  {no_trip_id:,} trip updates carried no trip_id "
              f"(identified by route + start time instead)")

    # A match rate counts strings. This counts agreement.
    print("\nDO THE MATCHED STOPS MEAN THE SAME STOP?\n")
    real = collision_check(conn, per_route)

    print("\nWHAT MOVES IN THIS FEED\n")
    for label, count in sorted(route_types(conn, route_ids).items(),
                               key=lambda kv: -kv[1]):
        print(f"  {label:<14} {count:>4} routes")

    if delays:
        delays.sort()
        late = sum(1 for d in delays if d > 60)
        print(f"\n  {len(delays):,} departure predictions   "
              f"median {delays[len(delays) // 2] / 60:+.1f} min   "
              f"{late:,} running >1 min late")

    # ---- verdict, stated as a decision rather than a mood ---------------
    print("\nREADING\n")
    if real < 50:
        print(f"  Stop identifiers DISAGREE ({real:.0f}% real agreement). Any")
        print("  per-stop prediction would be attached to the wrong stop, and")
        print("  would look exactly like a correct one. Delay estimates are")
        print("  off the table until the stop universes are reconciled.")
        print("  Route-level facts (which routes are running, where vehicles")
        print("  are) need no stop join and remain usable.")
    elif t < 0:
        print("  No trip_ids at all. The join would have to be on route +")
        print("  stop + time, which is weaker and needs its own design.")
    elif t >= 95:
        print("  The feeds share identifiers. Real-time can annotate our")
        print("  existing itineraries directly.")
    elif t >= 20:
        print(f"  Partial overlap ({t:.0f}%). Some trips can be annotated and")
        print("  some cannot, so every prediction needs to say which it is.")
    else:
        print(f"  Effectively no overlap ({t:.0f}%). The RT feed belongs to")
        print("  SurfaceGTFS.zip, not to our database. To use it we would")
        print("  load that feed alongside and join through stop + route +")
        print("  time — or load it as the source of truth for surface routes.")
    print()


if __name__ == "__main__":
    main()
