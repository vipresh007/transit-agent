"""Journey planning: real journeys with any number of transfers.

The capstone tool. Everything else left the model guessing at the hard part
-- which stops, and where to transfer -- and guessing produced empty queries
it then misread as "no service".

The search itself lives in raptor.py. This module turns coordinates into
candidate stops, calls it, and renders the result as fully-labelled legs so
the model transcribes rather than reconstructs.
"""

import json
import os
import sqlite3

from . import raptor
from .transit import DB_PATH


def _leg_dicts(tt, legs, origin_name, dest_name,
               origin_walk_m, dest_walk_m) -> list:
    """RAPTOR legs -> the shape the agent and the verifier already expect.

    Every minute is an explicit leg, including both end walks and every
    transfer walk. Returning bare distances once made the model invent the
    labels and get them backwards — a final leg reading "Distillery Loop ->
    Kensington Market". Anything the model has to reconstruct, it can
    reconstruct wrong; transcription is safer than reasoning.
    """
    out = []
    first, last = legs[0], legs[-1]

    walk_in = max(60, raptor._walk_seconds(origin_walk_m))
    out.append({
        "mode": "walk", "route": None,
        "from": origin_name, "to": tt.stop_names[first.from_stop],
        "from_stop": None, "to_stop": tt.stop_ids[first.from_stop],
        "depart": raptor.civil(first.depart - walk_in),
        "arrive": raptor.civil(first.depart),
        "metres": round(origin_walk_m),
    })

    for leg in legs:
        out.append({
            "mode": "transit" if leg.mode == "ride" else "walk",
            "route": leg.route,
            # The pattern's final stop, which is what the sign on the front
            # of the vehicle says. Cheap here, and it is the one thing a
            # traveller uses to check they boarded the right direction.
            "headsign": leg.headsign if hasattr(leg, "headsign") else None,
            "from": tt.stop_names[leg.from_stop],
            "to": tt.stop_names[leg.to_stop],
            "from_stop": tt.stop_ids[leg.from_stop],
            "to_stop": tt.stop_ids[leg.to_stop],
            "depart": raptor.civil(leg.depart),
            "arrive": raptor.civil(leg.arrive),
            **({} if leg.mode == "ride" else
               {"metres": round(raptor._metres(*tt.stop_pos[leg.from_stop],
                                               *tt.stop_pos[leg.to_stop]))}),
        })

    walk_out = max(60, raptor._walk_seconds(dest_walk_m))
    out.append({
        "mode": "walk", "route": None,
        "from": tt.stop_names[last.to_stop], "to": dest_name,
        "from_stop": tt.stop_ids[last.to_stop], "to_stop": None,
        "depart": raptor.civil(last.arrive),
        "arrive": raptor.civil(last.arrive + walk_out),
        "metres": round(dest_walk_m),
    })
    return out


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
    """Find a real journey between two coordinates, with any number of changes.

    The capstone tool. Everything before it left the agent guessing at the
    hard part -- WHICH stops to use and WHERE to transfer -- and guessing
    produced empty queries it then misread as "no service".

    HOW THIS USED TO WORK, AND WHY IT DOESN'T ANYMORE. The first version
    searched in SQL: every nearby-origin / nearby-destination pair for a
    direct ride, then, failing that, an interchange join to find stops
    reachable from one end that sit near stops reaching the other. Correct,
    and it stopped there — two transfers would have been the cross product of
    two large reachable sets. One transfer already took **308 seconds** on
    Kensington Market to Scarborough Town Centre before returning nothing,
    and the agent reported that as "No route found", which is a claim about
    Toronto rather than about our code.

    It now runs RAPTOR (see raptor.py): rounds instead of joins, so the
    transfer count is a loop counter. The same journey takes **59ms** and
    returns 510 -> Line 2 -> 129. Four transfers are as cheap as one.
    """
    for label, lat, lon in (
        ("origin", origin_lat, origin_lon),
        ("destination", dest_lat, dest_lon),
    ):
        # Reject placeholder coordinates. The model called this with (0, 0)
        # before geocoding anything -- a wasted request that silently
        # returned "no stops nearby", which reads like a fact about Toronto
        # rather than about the arguments.
        if (lat, lon) == (0, 0) or not (43.0 < lat < 44.5 and -80.5 < lon < -78.5):
            return (
                f"The {label} coordinates ({lat}, {lon}) are not in the Toronto "
                f"area. Call geocode first to turn the place name into real "
                f"coordinates, then call plan_journey with those."
            )

    if not os.path.exists(DB_PATH):
        return f"{DB_PATH} not found. Run `python scripts/load_gtfs.py` first."

    try:
        tt = raptor.timetable(service_id)
    except sqlite3.Error as exc:
        return f"Could not read the schedule: {exc}"

    near_origin = tt.nearest(origin_lat, origin_lon, raptor.ACCESS_RADIUS_M)
    near_dest = tt.nearest(dest_lat, dest_lon, raptor.ACCESS_RADIUS_M)
    if not near_origin or not near_dest:
        which = "origin" if not near_origin else "destination"
        return (f"No transit stops within {raptor.ACCESS_RADIUS_M}m of the "
                f"{which}.")

    # Candidate counts are a speed/quality tradeoff, and cutting them was
    # once the wrong lever: trimming destinations to 2 dropped Distillery
    # Loop (3rd nearest) and turned a 34-minute answer into a 61-minute one.
    # RAPTOR is fast enough to keep the search wide.
    origins = [(i, raptor._walk_seconds(m)) for i, m in near_origin[:15]]
    dests = [(i, raptor._walk_seconds(m)) for i, m in near_dest[:15]]
    origin_walk = dict(near_origin)
    dest_walk = dict(near_dest)

    after = raptor._secs(after_time)
    options = []
    seen = set()

    # Three departures, not one. The earliest arrival is the right answer to
    # "when can I get there", but a traveller wants to see that missing it
    # isn't fatal. Each pass starts a minute after the previous boarding.
    for _ in range(3):
        legs = raptor.query(tt, origins, dests, after)
        if not legs:
            break
        rides = [l for l in legs if l.mode == "ride"]
        if not rides:
            break

        shape = tuple((l.route, l.depart) for l in rides)
        if shape not in seen:
            seen.add(shape)
            options.append({
                "type": ("direct" if len(rides) == 1
                         else f"{len(rides) - 1}_transfer"),
                "transfers": len(rides) - 1,
                "legs": _leg_dicts(tt, legs, origin_name, dest_name,
                                   origin_walk.get(legs[0].from_stop, 0),
                                   dest_walk.get(legs[-1].to_stop, 0)),
                "depart": raptor.civil(rides[0].depart),
                "arrive": raptor.civil(rides[-1].arrive),
                "total_min": round((rides[-1].arrive - rides[0].depart) / 60),
            })
        after = rides[0].depart + 60

    if not options:
        # Say which kind of nothing this is. A limit of the search reported
        # as a fact about the city is the failure this tool used to have.
        return (
            f"No journey found departing after {after_time} on service "
            f"{service_id}. The search covered up to four transfers from "
            f"{len(origins)} nearby origin stops, so this is more likely no "
            f"service at that hour than a missing route — but say that it "
            f"wasn't found, not that it doesn't exist."
        )

    return json.dumps(options)
