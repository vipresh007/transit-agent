"""Read-only access to the TTC GTFS feed in SQLite.

Two layers here, and the split matters:

  - `query_transit` hands the model raw SQL. Flexible, and how the project
    started, but the model gets queries wrong in ways that return zero rows
    and read exactly like "no service".
  - `find_nearby_stops` / `find_direct_trips` are purpose-built. They encode
    the facts the model kept getting wrong -- that a stop_id is one
    direction-specific platform, that times aren't zero-padded, that dates
    outside the feed silently match nothing.

The lesson from building both: give the model raw power for open-ended
questions, and a dedicated tool for anything it has demonstrably failed at.
"""

import json
import os
import sqlite3

DB_PATH = "transit.db"

SCHEMA_DOC = """\
SQLite database of the TTC GTFS feed. All columns are TEXT.

TABLES
  routes(route_id, route_short_name, route_long_name, route_type)
      route_short_name is what riders call it: '501', '35', 'LINE 1'.
      route_type: 0=streetcar, 1=subway, 3=bus.

  trips(route_id, service_id, trip_id, trip_headsign, direction_id)
      One row per scheduled vehicle run.

  stop_times(trip_id, arrival_time, departure_time, stop_id, stop_sequence)
      One row per stop on a trip. This table has millions of rows -- always
      filter on stop_id or trip_id, both of which are indexed.

  stops(stop_id, stop_code, stop_name, stop_lat, stop_lon)

  calendar(service_id, monday..sunday, start_date, end_date)
      Each weekday column is '1' or '0'. Dates are 'YYYYMMDD' strings.

  calendar_dates(service_id, date, exception_type)
      Overrides calendar for specific dates.
      exception_type '1' = service ADDED, '2' = service REMOVED.
      Holidays live here. Ignoring this table gives wrong answers on
      statutory holidays even though calendar looks right.

CRITICAL QUIRKS
  * Times can exceed 24:00:00. A departure_time of '25:30:00' means 1:30am
    on the following calendar day, still counted as the previous service day.

  * TIMES ARE NOT CONSISTENTLY ZERO-PADDED. This feed mixes '6:32:37'
    (7 chars) and '06:32:37' (8 chars), so naive string comparison sorts
    '9:15:00' AFTER '25:23:00' and silently gives the wrong "last" vehicle.
    Always normalise first. The cleanest idiom is:
        substr('0' || departure_time, -8)
    which prepends a zero and keeps the last 8 characters, so both '6:32:37'
    and '06:32:37' become '06:32:37'.
    Use the padded form in every ORDER BY, MAX(), MIN(), and range comparison
    on a time column. This applies EQUALLY to earliest and latest questions --
    an unpadded MIN() on this feed returns '10:00:00' when the true first
    departure is 03:49:18.

WORKED EXAMPLE -- first and last departure of a route
  Copy this shape for any "first/last/earliest/latest vehicle" question.
  Substitute the route, direction and service, and change nothing else:

    SELECT MIN(substr('0' || st.departure_time, -8)) AS first_departure,
           MAX(substr('0' || st.departure_time, -8)) AS last_departure
    FROM trips t
    JOIN stop_times st ON st.trip_id = t.trip_id
    JOIN routes r      ON r.route_id = t.route_id
    WHERE r.route_short_name = '501'
      AND t.direction_id = '0'      -- confirm via trip_headsign first
      AND t.service_id  = '1'       -- '1' = weekday in this feed
      AND st.stop_sequence = '1'    -- '1' is the trip's first stop

  One query answers both ends. Do not run separate exploratory queries for
  earliest and latest.

WORKED EXAMPLE -- departures around a time of day
  For "morning", "after 9am", "when I finish work" and similar, you want the
  next few departures AFTER a time. Do NOT use MIN() for this: MIN() returns
  the first vehicle of the service day, typically before 06:00, which is
  never what someone planning a morning trip means.

    SELECT substr('0' || st.departure_time, -8) AS departure,
           t.trip_headsign
    FROM trips t
    JOIN stop_times st ON st.trip_id = t.trip_id
    JOIN routes r      ON r.route_id = t.route_id
    WHERE r.route_short_name = '506'
      AND t.service_id = '1'
      AND st.stop_id = '809'                  -- from find_nearby_stops
      AND substr('0' || st.departure_time, -8) >= '08:00:00'
    ORDER BY departure
    LIMIT 4

  Returning several departures also reveals the frequency, which is what
  actually matters for a transfer: "every 10 minutes" tells the traveller
  more than any single scheduled time.
  * direction_id is '0' or '1' and has NO inherent meaning. It does not map
    to eastbound/westbound. To find which direction is which, look at
    trip_headsign for each direction_id -- the headsign names the terminus
    ('Neville Park' is the east end of the 501; 'Long Branch' is the west).
    Resolve this in ONE query, then move on:
      SELECT direction_id, trip_headsign, COUNT(*) FROM trips
      WHERE route_id = 'X' GROUP BY direction_id, trip_headsign
  * "The last vehicle" is ambiguous. Usually the user means the last trip to
    START from the terminus. Pick the reading that fits, state which one you
    used in your answer, and do not run more queries to resolve the ambiguity.
  * stop_lat/stop_lon are TEXT. CAST to REAL before doing arithmetic.
  * A single named location (e.g. 'DUNDAS STATION') has several stop_ids for
    different platforms and directions. Match with LIKE and expect duplicates.
  * To find whether a service runs on a date, check calendar for the weekday
    AND the date range, THEN apply any calendar_dates exception.
"""


def find_nearby_stops(lat: float, lon: float, radius_m: int = 800) -> str:
    """Transit stops near a coordinate, with the routes that serve them.

    The tool that was missing. Given only geocode + SQL, the agent guessed at
    stop NAMES -- searching for '%Kensington%' in a feed with no such stop --
    and burned its whole budget. Place names and stop names are different
    vocabularies; the only reliable bridge between them is coordinates.

    A capability gap doesn't announce itself as an error. It shows up as the
    model flailing near the thing it can't do.
    """
    if not os.path.exists(DB_PATH):
        return f"{DB_PATH} not found. Run `python load_gtfs.py` first."

    # Equirectangular approximation: fine at city scale and, unlike haversine,
    # expressible in plain SQL so SQLite can use it directly.
    deg_lat = radius_m / 111_320
    deg_lon = radius_m / (111_320 * 0.723)  # cos(43.65 degrees)

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            """
            SELECT s.stop_id, s.stop_name,
                   CAST(ROUND(111320.0 * SQRT(
                       POWER(CAST(s.stop_lat AS REAL) - ?, 2) +
                       POWER((CAST(s.stop_lon AS REAL) - ?) * 0.723, 2)
                   )) AS INT) AS metres,
                   GROUP_CONCAT(DISTINCT r.route_short_name) AS routes,
                   GROUP_CONCAT(DISTINCT t.direction_id)     AS directions,
                   GROUP_CONCAT(DISTINCT t.trip_headsign)    AS headsigns
            FROM stops s
            JOIN stop_times st ON st.stop_id = s.stop_id
            JOIN trips t       ON t.trip_id  = st.trip_id
            JOIN routes r      ON r.route_id = t.route_id
            WHERE CAST(s.stop_lat AS REAL) BETWEEN ? AND ?
              AND CAST(s.stop_lon AS REAL) BETWEEN ? AND ?
            GROUP BY s.stop_id
            ORDER BY metres
            LIMIT 12
            """,
            (lat, lon, lat - deg_lat, lat + deg_lat, lon - deg_lon, lon + deg_lon),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return (
            f"No transit stops within {radius_m}m of ({lat}, {lon}). "
            f"Try a larger radius, once."
        )

    # Directions and headsigns matter as much as distance. A TTC stop_id is
    # ONE PLATFORM, serving one direction: 809 is eastbound College/Augusta,
    # 12338 is the westbound side of the same intersection. Querying
    # direction_id=1 at stop 809 returns nothing, and "no rows" is
    # indistinguishable from "no service" -- so this has to be handed over.
    out = []
    for sid, name, m, routes, directions, headsigns in rows:
        heads = sorted(set((headsigns or "").split(",")))
        out.append({
            "stop_id": sid,
            "stop_name": name,
            "metres": m,
            "routes": (routes or "").split(","),
            "direction_ids": sorted(set((directions or "").split(","))),
            # Headsigns name the terminus, which is how you tell which way a
            # direction_id actually points.
            "serves": heads[:4],
        })
    return json.dumps(out)


def find_direct_trips(
    origin_stop_id: str,
    dest_stop_id: str,
    after_time: str = "08:00:00",
    service_id: str = "1",
) -> str:
    """Scheduled trips serving BOTH stops, origin before destination.

    The core journey query, and the one the agent kept failing to write. It
    needs a self-join on stop_times with a stop_sequence ordering condition --
    fiddly enough that hand-written attempts produced empty results that were
    then misread as "no service".

    An empty result here is INFORMATIVE: it means no single vehicle serves
    both stops, so a transfer is required. That's a real answer, not a
    failure, and the message says so.
    """
    if not os.path.exists(DB_PATH):
        return f"{DB_PATH} not found. Run `python load_gtfs.py` first."

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            """
            SELECT r.route_short_name,
                   t.trip_headsign,
                   substr('0' || a.departure_time, -8) AS depart,
                   substr('0' || b.arrival_time,   -8) AS arrive,
                   (CAST(b.stop_sequence AS INT) - CAST(a.stop_sequence AS INT))
                       AS stops_between
            FROM trips t
            JOIN routes r     ON r.route_id = t.route_id
            JOIN stop_times a ON a.trip_id  = t.trip_id AND a.stop_id = ?
            JOIN stop_times b ON b.trip_id  = t.trip_id AND b.stop_id = ?
            WHERE t.service_id = ?
              AND CAST(a.stop_sequence AS INT) < CAST(b.stop_sequence AS INT)
              AND substr('0' || a.departure_time, -8) >= ?
            ORDER BY depart
            LIMIT 5
            """,
            (origin_stop_id, dest_stop_id, service_id, after_time),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        return f"SQL error: {exc}"
    finally:
        conn.close()

    if not rows:
        return (
            f"No direct trip serves both stop {origin_stop_id} and stop "
            f"{dest_stop_id} (in that order) after {after_time}. This means a "
            f"TRANSFER IS REQUIRED — it does not mean there is no service. "
            f"Find an intermediate stop served by a route from each end, then "
            f"call this tool twice: origin -> interchange, interchange -> dest."
        )

    return json.dumps([
        {"route": route, "headsign": head, "depart": dep,
         "arrive": arr, "stops": n}
        for route, head, dep, arr, n in rows
    ])


def describe_transit_schema() -> str:
    """Return the GTFS schema, its traps, and the live calendar table.

    Built at call time rather than hardcoded. The calendar is only 12 rows,
    and the model kept inventing service_ids and querying dates outside the
    feed's coverage. Facts that small and that load-bearing should be handed
    over, not guessed -- and computing them means they can't go stale when
    the TTC republishes.
    """
    if not os.path.exists(DB_PATH):
        return (
            f"{DB_PATH} not found. Run `python load_gtfs.py` first to download "
            "and load the TTC feed."
        )

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        cal = conn.execute(
            "SELECT service_id, monday, tuesday, wednesday, thursday, friday, "
            "saturday, sunday, start_date, end_date FROM calendar"
        ).fetchall()
        lo, hi = conn.execute(
            "SELECT MIN(start_date), MAX(end_date) FROM calendar"
        ).fetchone()
    finally:
        conn.close()

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    lines = []
    for row in cal:
        sid, *flags, start, end = row
        active = [d for d, f in zip(days, flags) if f == "1"] or ["(none — see calendar_dates)"]
        lines.append(f"    service_id '{sid}': {', '.join(active)}")

    live = f"""
LIVE FEED FACTS (read from your database just now — use these, do not guess)

  Coverage: {lo} to {hi}. Any date outside this range returns nothing.
  Queries against dates from other years are a common and silent mistake.

  The complete calendar table ({len(cal)} rows):
{chr(10).join(lines)}

  So: weekday = service_id '1', Saturday = '2', Sunday = '3'.
  The service_ids with no weekday flags run ONLY on dates listed in
  calendar_dates with exception_type '1'.
"""
    return SCHEMA_DOC + live


def query_transit(sql: str) -> str:
    """Run a read-only SELECT against the transit database."""
    if not os.path.exists(DB_PATH):
        return f"{DB_PATH} not found. Run `python load_gtfs.py` first."

    cleaned = sql.strip().rstrip(";")
    if not cleaned.lower().startswith(("select", "with")):
        return "Only SELECT (or WITH ... SELECT) queries are allowed."

    # One statement at a time keeps the surface small.
    if ";" in cleaned:
        return "Send one statement at a time, without a trailing semicolon."

    # 20, not 50. Every row returned is re-sent to the model on every
    # subsequent turn of the loop, so result size compounds. Twenty rows is
    # enough to see a pattern; if the agent needs more it can ask.
    if "limit" not in cleaned.lower():
        cleaned += " LIMIT 20"

    # Open read-only so a bad query can't damage the database.
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)

    # Abort runaway queries. Without this, one accidental cross join against
    # stop_times hangs the agent until the step cap saves you.
    steps = {"n": 0}

    def guard():
        steps["n"] += 1
        return 1 if steps["n"] > 8000 else 0  # nonzero aborts

    conn.set_progress_handler(guard, 100_000)

    try:
        cursor = conn.execute(cleaned)
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
    except sqlite3.OperationalError as exc:
        # Returned, not raised: the model reads this and rewrites its SQL.
        return f"SQL error: {exc}"
    finally:
        conn.close()

    if not rows:
        return "Query returned no rows."
    return json.dumps([dict(zip(columns, row)) for row in rows])


def check_mode_feasibility(lat: float, lon: float, avoid_modes: str) -> str:
    """Can this place be reached without the modes the traveller avoids?

    Asked for Scarborough Town Centre with "avoid bus", the model produced
    Line 3 RT — a rail line that closed in 2023 and is not in the feed —
    rather than report that the trip is bus-only. One query up front turns an
    impossible request into an honest answer instead of 34 requests spent
    hunting for a route that cannot exist.
    """
    import constraints

    modes = [m.strip() for m in (avoid_modes or "").split(",") if m.strip()]
    if not modes:
        return "No modes to avoid; nothing to check."

    warnings = constraints.preflight(
        constraints.Preferences(avoid_modes=modes), (lat, lon)
    )
    if warnings:
        return warnings[0]
    return (
        f"Reachable without {', '.join(modes)}. Continue planning normally."
    )
