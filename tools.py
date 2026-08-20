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


# Overpass speaks its own query language. These are the tags we know how to ask for.
POI_TAGS = {
    "museum": '[tourism=museum]',
    "gallery": '[tourism=gallery]',
    "attraction": '[tourism=attraction]',
    "viewpoint": '[tourism=viewpoint]',
    "park": '[leisure=park]',
    "cafe": '[amenity=cafe]',
    "restaurant": '[amenity=restaurant]',
    "bar": '[amenity=bar]',
    "subway_station": '[railway=station][station=subway]',
}


def find_pois(lat: float, lon: float, category: str, radius_m: int = 1500) -> str:
    """Find points of interest near a coordinate. See POI_TAGS for categories."""
    tag = POI_TAGS.get(category)
    if tag is None:
        return f"Unknown category {category!r}. Valid options: {', '.join(POI_TAGS)}"

    # Overpass QL: find nodes+ways with this tag inside a radius, return centers.
    query = f"""
    [out:json][timeout:25];
    (
      node{tag}(around:{radius_m},{lat},{lon});
      way{tag}(around:{radius_m},{lat},{lon});
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

    places = []
    for el in r.json().get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name:
            continue  # unnamed POIs are noise for our purposes
        places.append(
            {
                "name": name,
                "lat": el.get("lat") or el.get("center", {}).get("lat"),
                "lon": el.get("lon") or el.get("center", {}).get("lon"),
                "opening_hours": tags.get("opening_hours"),
            }
        )

    if not places:
        return f"No {category} found within {radius_m}m."
    return json.dumps(places[:20])


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
    on a time column.
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


def describe_transit_schema() -> str:
    """Return the GTFS schema and its known traps."""
    if not os.path.exists(DB_PATH):
        return (
            f"{DB_PATH} not found. Run `python load_gtfs.py` first to download "
            "and load the TTC feed."
        )
    return SCHEMA_DOC


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

    if "limit" not in cleaned.lower():
        cleaned += " LIMIT 50"

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
}
