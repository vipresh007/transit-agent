"""
Tools the agent can call.

Every tool here is a plain Python function. Two rules:
  1. It takes JSON-friendly arguments (str, int, float, bool).
  2. It returns a string, or something json.dumps() can handle.

That's it. There is no magic. The "agent" part is just the model
choosing which of these to call and with what arguments.

All three services below are free and need no API key.
"""

import json
import os
import sqlite3

import requests

# Nominatim's usage policy requires a real User-Agent identifying your app.
# Be a good citizen: they run this for free.
HEADERS = {"User-Agent": "transit-agent-learning-project/0.1"}
TIMEOUT = 20


def geocode(place: str) -> str:
    """Turn a place name into coordinates. 'CN Tower, Toronto' -> lat/lon."""
    r = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": place, "format": "json", "limit": 1},
        headers=HEADERS,
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    results = r.json()
    if not results:
        return f"No location found for {place!r}. Try adding a city or country."

    hit = results[0]
    return json.dumps(
        {
            "name": hit.get("display_name"),
            "lat": float(hit["lat"]),
            "lon": float(hit["lon"]),
        }
    )


def get_weather(lat: float, lon: float, date: str = "") -> str:
    """Forecast for a coordinate. date is YYYY-MM-DD; blank means today."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "timezone": "auto",
        "forecast_days": 7,
    }
    if date:
        params["start_date"] = date
        params["end_date"] = date
        params.pop("forecast_days")

    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params=params,
        headers=HEADERS,
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    daily = r.json()["daily"]

    days = [
        {
            "date": d,
            "high_c": hi,
            "low_c": lo,
            "rain_chance_pct": rain,
        }
        for d, hi, lo, rain in zip(
            daily["time"],
            daily["temperature_2m_max"],
            daily["temperature_2m_min"],
            daily["precipitation_probability_max"],
        )
    ]
    return json.dumps(days)


# Overpass speaks its own query language. Each category maps to a LIST of tag
# selectors, because OSM tagging is inconsistent in practice: TTC subway
# stations appear as station=subway, as railway=station + subway=yes, and as
# public_transport=station. Querying only one spelling returns nothing, and
# an agent that gets nothing back tends to retry with different radii forever
# rather than concluding the tag is wrong.
POI_TAGS = {
    "museum": ['[tourism=museum]'],
    "gallery": ['[tourism=gallery]'],
    "attraction": ['[tourism=attraction]'],
    "viewpoint": ['[tourism=viewpoint]'],
    "park": ['[leisure=park]'],
    "cafe": ['[amenity=cafe]'],
    "restaurant": ['[amenity=restaurant]'],
    "bar": ['[amenity=bar]'],
    "subway_station": [
        '[station=subway]',
        '[railway=station][subway=yes]',
        '[railway=station][transport=subway]',
    ],
}


def find_pois(lat: float, lon: float, category: str, radius_m: int = 1500) -> str:
    """Find points of interest near a coordinate. See POI_TAGS for categories."""
    selectors = POI_TAGS.get(category)
    if selectors is None:
        return f"Unknown category {category!r}. Valid options: {', '.join(POI_TAGS)}"

    # Overpass QL: union of every tagging variant, nodes and ways, in radius.
    clauses = "\n      ".join(
        f"node{tag}(around:{radius_m},{lat},{lon});\n"
        f"      way{tag}(around:{radius_m},{lat},{lon});"
        for tag in selectors
    )
    query = f"""
    [out:json][timeout:25];
    (
      {clauses}
    );
    out center 60;
    """

    r = requests.post(
        "https://overpass-api.de/api/interpreter",
        data={"data": query},
        headers=HEADERS,
        timeout=60,
    )
    r.raise_for_status()

    places = {}
    for el in r.json().get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name or name in places:
            continue  # unnamed POIs are noise; union queries return dupes
        places[name] = {
            "name": name,
            "lat": el.get("lat") or el.get("center", {}).get("lat"),
            "lon": el.get("lon") or el.get("center", {}).get("lon"),
            "opening_hours": tags.get("opening_hours"),
        }

    if not places:
        # Tell the agent what to do next. A bare "no results" invites it to
        # retry the same call at a slightly different radius, forever.
        return (
            f"No {category} found within {radius_m}m. Do NOT retry with a "
            f"different radius — the answer will be the same. For transit "
            f"stops, query the stops table via query_transit instead: it has "
            f"stop_lat/stop_lon and is authoritative for the TTC."
        )
    return json.dumps(list(places.values())[:20])


# ---------------------------------------------------------------------------
# Transit tools (stage 2): read-only SQL over the TTC GTFS feed.
#
# Instead of writing a tool per question ("next departure", "does route X
# run on Sunday"), we hand the model the schema and let it write SQL. It will
# get queries wrong. That's the point -- the error goes back into the loop and
# it tries again, which is self-correction with a real feedback signal.
# ---------------------------------------------------------------------------

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


def _direct_leg(conn, origin: str, dest: str, after: str, service_id: str):
    """One scheduled ride from origin to dest, or None."""
    return conn.execute(
        """
        SELECT r.route_short_name, t.trip_headsign,
               substr('0' || a.departure_time, -8),
               substr('0' || b.arrival_time,   -8)
        FROM trips t
        JOIN routes r     ON r.route_id = t.route_id
        JOIN stop_times a ON a.trip_id = t.trip_id AND a.stop_id = ?
        JOIN stop_times b ON b.trip_id = t.trip_id AND b.stop_id = ?
        WHERE t.service_id = ?
          AND CAST(a.stop_sequence AS INT) < CAST(b.stop_sequence AS INT)
          AND substr('0' || a.departure_time, -8) >= ?
        ORDER BY substr('0' || a.departure_time, -8)
        LIMIT 1
        """,
        (origin, dest, service_id, after),
    ).fetchone()


def plan_journey(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    after_time: str = "08:00:00",
    service_id: str = "1",
) -> str:
    """Find a real journey between two coordinates: direct, or one transfer.

    The capstone tool. Everything before it left the agent guessing at the
    hard part -- WHICH stops to use and WHERE to transfer -- and guessing
    produced empty queries it then misread as "no service". Here the search
    happens in SQL, over actual data:

      1. try every nearby-origin / nearby-destination pair for a direct ride
      2. failing that, compute interchanges: stops reachable from the origin
         that sit within 250m of a stop that can reach the destination
      3. look up real times for both legs, requiring the second to depart
         after the first arrives plus walking time

    Note step 2 depends on which origin PLATFORM you start from. From
    College/Augusta the 506 runs east along Carlton and never meets a route
    to the Distillery; from Spadina/Nassau the 510 meets the 504 at King.
    Same neighbourhood, different answer -- which is why this has to search
    over candidate stops rather than pick one.
    """
    if not os.path.exists(DB_PATH):
        return f"{DB_PATH} not found. Run `python load_gtfs.py` first."

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        # Candidate counts are a speed/quality tradeoff, and cutting them was
        # the wrong lever: trimming destinations to 2 dropped Distillery Loop
        # (3rd nearest) and turned a 34-minute answer into a 61-minute one.
        # Keep the search wide and make it fast instead — run optimize_db.py
        # to add the composite indexes this relies on.
        origins = json.loads(find_nearby_stops(origin_lat, origin_lon, 800))[:4]
        dests = json.loads(find_nearby_stops(dest_lat, dest_lon, 800))[:4]
    except (json.JSONDecodeError, TypeError):
        conn.close()
        return "No transit stops near one or both coordinates."

    try:
        # --- 1. direct rides -------------------------------------------------
        direct = []
        for o in origins:
            for d in dests:
                hit = _direct_leg(conn, o["stop_id"], d["stop_id"],
                                  after_time, service_id)
                if hit:
                    route, head, dep, arr = hit
                    direct.append({
                        "type": "direct",
                        "legs": [{
                            "route": route, "headsign": head,
                            "from": o["stop_name"], "from_stop": o["stop_id"],
                            "to": d["stop_name"], "to_stop": d["stop_id"],
                            "depart": dep, "arrive": arr,
                        }],
                        "walk_to_stop_m": o["metres"],
                        "walk_from_stop_m": d["metres"],
                    })
        if direct:
            direct.sort(key=lambda j: j["legs"][0]["arrive"])
            return json.dumps(direct[:3])

        # --- 2. one transfer -------------------------------------------------
        interchange_sql = """
        WITH fwd AS (
          SELECT DISTINCT b.stop_id FROM stop_times a
          JOIN stop_times b ON b.trip_id = a.trip_id
          WHERE a.stop_id = ?
            AND CAST(b.stop_sequence AS INT) > CAST(a.stop_sequence AS INT)),
        bwd AS (
          SELECT DISTINCT a.stop_id FROM stop_times a
          JOIN stop_times b ON b.trip_id = a.trip_id
          WHERE b.stop_id = ?
            AND CAST(a.stop_sequence AS INT) < CAST(b.stop_sequence AS INT))
        SELECT sf.stop_id, sf.stop_name, sw.stop_id, sw.stop_name,
               CAST(ROUND(111320.0*SQRT(
                 POWER(CAST(sf.stop_lat AS REAL)-CAST(sw.stop_lat AS REAL),2) +
                 POWER((CAST(sf.stop_lon AS REAL)-CAST(sw.stop_lon AS REAL))*0.723,2)
               )) AS INT) AS gap
        FROM fwd f JOIN stops sf ON sf.stop_id = f.stop_id,
             bwd w JOIN stops sw ON sw.stop_id = w.stop_id
        WHERE gap < 250
        ORDER BY gap LIMIT 3
        """

        options = []
        seen_shapes = set()
        for o in origins:
            for d in dests:
                for xa, xa_name, xb, xb_name, gap in conn.execute(
                    interchange_sql, (o["stop_id"], d["stop_id"])
                ).fetchall():
                    leg1 = _direct_leg(conn, o["stop_id"], xa,
                                       after_time, service_id)
                    if not leg1:
                        continue
                    # Allow 1 minute per 60m of walking, minimum 2 minutes.
                    walk_min = max(2, round(gap / 60))
                    h, m, s = (int(p) for p in leg1[3].split(":"))
                    ready = h * 3600 + m * 60 + s + walk_min * 60
                    ready_str = f"{ready//3600:02d}:{(ready%3600)//60:02d}:{ready%60:02d}"

                    leg2 = _direct_leg(conn, xb, d["stop_id"],
                                       ready_str, service_id)
                    if not leg2:
                        continue
                    # Two interchanges one block apart on the same pair of
                    # routes are the same journey to a traveller.
                    shape = (leg1[0], leg2[0])
                    if shape in seen_shapes:
                        continue
                    seen_shapes.add(shape)
                    options.append({
                        "type": "one_transfer",
                        "legs": [
                            {"route": leg1[0], "headsign": leg1[1],
                             "from": o["stop_name"], "from_stop": o["stop_id"],
                             "to": xa_name, "to_stop": xa,
                             "depart": leg1[2], "arrive": leg1[3]},
                            {"route": leg2[0], "headsign": leg2[1],
                             "from": xb_name, "from_stop": xb,
                             "to": d["stop_name"], "to_stop": d["stop_id"],
                             "depart": leg2[2], "arrive": leg2[3]},
                        ],
                        "transfer_walk_m": gap,
                        "transfer_walk_min": walk_min,
                        "walk_to_stop_m": o["metres"],
                        "walk_from_stop_m": d["metres"],
                    })
        if not options:
            return (
                "No direct or single-transfer journey found between these "
                "coordinates after {t}. The trip may need two transfers, or "
                "there may be no service at that hour.".format(t=after_time)
            )

        options.sort(key=lambda j: j["legs"][-1]["arrive"])
        return json.dumps(options[:3])
    finally:
        conn.close()


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


# ---------------------------------------------------------------------------
# Schemas: how we describe the functions above to the model.
#
# This is the part people find surprising. The model never sees your code --
# it only sees these descriptions. If a tool gets called wrongly, the fix is
# almost always here, in the wording, not in the function.
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "geocode",
            "description": (
                "Convert a place name or address into latitude/longitude coordinates. "
                "Call this first whenever you need coordinates for another tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "place": {
                        "type": "string",
                        "description": "Place name, ideally with city, e.g. 'Kensington Market, Toronto'",
                    }
                },
                "required": ["place"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get the daily weather forecast for a coordinate, up to 7 days ahead. "
                "Use this to decide between indoor and outdoor activities."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                    "date": {
                        "type": "string",
                        "description": "Optional single date as YYYY-MM-DD. Omit for a 7-day forecast.",
                    },
                },
                "required": ["lat", "lon"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_pois",
            "description": (
                "Find named points of interest near a coordinate, with opening hours "
                "when OpenStreetMap has them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                    "category": {
                        "type": "string",
                        "enum": list(POI_TAGS.keys()),
                    },
                    "radius_m": {
                        "type": "integer",
                        "description": "Search radius in metres. Default 1500.",
                    },
                },
                "required": ["lat", "lon", "category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_nearby_stops",
            "description": (
                "Find TTC stops near a coordinate, with distance and the routes "
                "serving each. ALWAYS use this to go from a place to its stops. "
                "Do NOT search the stops table by place name — stop names are "
                "intersections like 'College St at Augusta Ave', so "
                "neighbourhood and landmark names will never match."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                    "radius_m": {
                        "type": "integer",
                        "description": "Search radius in metres. Default 800.",
                    },
                },
                "required": ["lat", "lon"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_journey",
            "description": (
                "Plan a real journey between two coordinates. Searches nearby "
                "stops at both ends for a direct ride, and failing that "
                "computes single-transfer options with a real interchange and "
                "verified departure/arrival times for each leg. "
                "USE THIS FIRST for any A-to-B question — it replaces manually "
                "picking stops, guessing an interchange, and writing journey "
                "SQL, all of which are error-prone."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "origin_lat": {"type": "number"},
                    "origin_lon": {"type": "number"},
                    "dest_lat": {"type": "number"},
                    "dest_lon": {"type": "number"},
                    "after_time": {
                        "type": "string",
                        "description": "Earliest departure, HH:MM:SS. Default 08:00:00.",
                    },
                    "service_id": {
                        "type": "string",
                        "description": "'1' weekday, '2' Saturday, '3' Sunday.",
                    },
                },
                "required": ["origin_lat", "origin_lon", "dest_lat", "dest_lon"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_direct_trips",
            "description": (
                "Find scheduled trips that serve BOTH stops on the same vehicle, "
                "origin before destination, with real departure and arrival "
                "times. Use this INSTEAD of hand-writing journey SQL. If it "
                "returns no trips, a transfer is required — call it again for "
                "each leg via an interchange stop."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "origin_stop_id": {
                        "type": "string",
                        "description": "stop_id from find_nearby_stops. Platforms "
                        "are direction-specific; pick the one whose 'serves' "
                        "headsigns point the way you're going.",
                    },
                    "dest_stop_id": {"type": "string"},
                    "after_time": {
                        "type": "string",
                        "description": "Earliest departure, HH:MM:SS. Default 08:00:00.",
                    },
                    "service_id": {
                        "type": "string",
                        "description": "'1' weekday, '2' Saturday, '3' Sunday.",
                    },
                },
                "required": ["origin_stop_id", "dest_stop_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_transit_schema",
            "description": (
                "Get the schema of the TTC transit database, including column "
                "names and important quirks. ALWAYS call this before writing "
                "any SQL with query_transit."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_transit",
            "description": (
                "Run a read-only SQL SELECT against the TTC schedule database "
                "(routes, trips, stop_times, stops, calendar, calendar_dates). "
                "Use for real departure times, route lookups, and service days. "
                "Call describe_transit_schema first. If the query errors, read "
                "the message and try a corrected query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A single SELECT statement, no trailing semicolon.",
                    }
                },
                "required": ["sql"],
            },
        },
    },
]

# Name -> function, so the loop can dispatch on whatever the model asks for.
TOOL_FUNCTIONS = {
    "geocode": geocode,
    "get_weather": get_weather,
    "find_pois": find_pois,
    "describe_transit_schema": describe_transit_schema,
    "query_transit": query_transit,
    "find_nearby_stops": find_nearby_stops,
    "find_direct_trips": find_direct_trips,
    "plan_journey": plan_journey,
}

# Tools whose output can constitute a VERIFIED schedule time.
#
# Declared here, next to the tools themselves, and imported by agent.py. It
# lived in agent.py as a hardcoded set and went stale twice — once when
# find_direct_trips was added, again when plan_journey was. A registry that
# sits far away from the thing it registers will always drift; keeping it
# adjacent at least makes the omission visible when you add a tool.
SCHEDULE_TOOLS = {"query_transit", "find_direct_trips", "plan_journey"}
