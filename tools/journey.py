"""Journey planning: direct rides and single-transfer routes with real times.

The capstone tool. Everything else left the model guessing at the hard part
-- which stops, and where to transfer -- and guessing produced empty queries
it then misread as "no service". Here the search happens in SQL over actual
data, and the result is a fully-labelled set of legs so the model transcribes
rather than reconstructs.
"""

import json
import os
import sqlite3

from .transit import DB_PATH, find_nearby_stops

def _shift(t: str, minutes: int) -> str:
    """Shift a GTFS time by minutes, keeping 24+ hour notation.

    Clamped at zero: walking back 5 minutes from a 00:02 departure produced
    '-1:57:00', which is not a time and fails schema validation. Rare, but
    a 00:02 departure is exactly the kind of input nobody tests by hand.
    """
    h, m, s = (int(p) for p in t.split(":"))
    total = max(0, h * 3600 + m * 60 + s + minutes * 60)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def _with_walks(option: dict, origin_name: str, dest_name: str) -> dict:
    """Add fully-labelled walking legs at both ends.

    Returning bare distances made the model invent the labels, and it got
    them backwards: a final leg reading "Distillery Loop -> Kensington
    Market". Anything the model has to reconstruct, it can reconstruct wrong.
    Emitting complete legs turns a reasoning step into transcription.
    """
    legs = option["legs"]
    walk_in = max(1, round(option["walk_to_stop_m"] / 75))
    walk_out = max(1, round(option["walk_from_stop_m"] / 75))

    # Interleave the transfer walk too, so every minute of the journey is an
    # explicit leg and nothing has to be inferred.
    middle = []
    for i, leg in enumerate(legs):
        middle.append({**leg, "mode": "transit"})
        if i == 0 and len(legs) > 1:
            middle.append({
                "mode": "walk", "route": None,
                "from": leg["to"], "to": legs[1]["from"],
                "depart": leg["arrive"],
                "arrive": _shift(leg["arrive"], option["transfer_walk_min"]),
                "metres": option["transfer_walk_m"],
            })

    option["legs"] = [
        {
            "mode": "walk", "route": None,
            "from": origin_name, "to": legs[0]["from"],
            "depart": _shift(legs[0]["depart"], -walk_in),
            "arrive": legs[0]["depart"],
            "metres": option["walk_to_stop_m"],
        },
        *middle,
        {
            "mode": "walk", "route": None,
            "from": legs[-1]["to"], "to": dest_name,
            "depart": legs[-1]["arrive"],
            "arrive": _shift(legs[-1]["arrive"], walk_out),
            "metres": option["walk_from_stop_m"],
        },
    ]
    option["depart"] = option["legs"][0]["depart"]
    option["arrive"] = option["legs"][-1]["arrive"]
    return option


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
    origin_name: str = "origin",
    dest_name: str = "destination",
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

    # Reject placeholder coordinates. The model called this with (0, 0) before
    # geocoding anything -- a wasted request that silently returned "no stops
    # nearby", which reads like a fact about Toronto rather than about the
    # arguments. Say plainly what went wrong and what to do first.
    for label, lat, lon in (
        ("origin", origin_lat, origin_lon),
        ("destination", dest_lat, dest_lon),
    ):
        if (lat, lon) == (0, 0) or not (43.0 < lat < 44.5 and -80.5 < lon < -78.5):
            return (
                f"The {label} coordinates ({lat}, {lon}) are not in the Toronto "
                f"area. Call geocode first to turn the place name into real "
                f"coordinates, then call plan_journey with those."
            )

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
            return json.dumps([_with_walks(o, origin_name, dest_name)
                               for o in direct[:3]])

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
        return json.dumps([_with_walks(o, origin_name, dest_name)
                           for o in options[:3]])
    finally:
        conn.close()
