"""Turning a PlanResult into rows. No Streamlit, no printing, no I/O.

Why this isn't just inside ui.py: ui.py imports streamlit at module scope, so
no test can import it, so nothing in it can be checked. That gap cost a real
crash on first launch — the sidebar called `memory.load().items()`, but load()
returns `(preferences, notes)`. A five-second mistake that only a browser
could find, because the only code that could find it was unreachable.

So the pure functions live here, where the suite can reach them, and ui.py
keeps only what genuinely needs a widget. The rule generalises: if a UI file
can't be imported, everything in it is untested, so put as little in it as
possible.
"""

from __future__ import annotations

import json
import math
import sqlite3

from transit import paths
from transit.tools import geo, memory
from transit.verify import constraints
from transit.verify.gtfstime import to_civil


def leg_rows(itinerary) -> list[dict]:
    """The itinerary as table rows."""
    return [
        {
            "Leave": to_civil(leg.depart),
            "Mode": "walk" if leg.mode == "walk" else f"{leg.mode} {leg.route}",
            "From": leg.origin,
            "To": leg.destination,
            "Mins": leg.duration_min,
        }
        for leg in itinerary.legs
    ]


def remembered_rows() -> list[tuple[str, str, bool]]:
    """Stored memory as (label, value, forgettable) rows.

    memory.load() returns TWO things — enforceable preferences and free-text
    notes — and they behave differently: a preference becomes a hard
    constraint on every future plan, a note is only ever shown to the model.
    Flattening them into one list would let someone "forget" a note expecting
    it to change their journeys.
    """
    preferences, notes = memory.load()
    rows = [(key, str(value), True) for key, value in sorted(preferences.items())]
    rows += [("note", note, False) for note in notes]
    return rows


def badge_values(result) -> dict[str, str]:
    """The three headline numbers, decided once so no front end can disagree."""
    coverage = result.grounding.get("coverage")
    return {
        "Schedule": (f"{len(result.violations)} problem(s)"
                     if result.violations else "verified"),
        # An answer with no retrieved times isn't wrong, it's unfounded — and
        # the two look identical unless something says so out loud.
        "Times": "ESTIMATED" if result.no_schedule_data else "from the feed",
        "Grounding": f"{coverage:.0%}" if coverage is not None else "—",
    }


# ---------------------------------------------------------------------------
# Map geometry
# ---------------------------------------------------------------------------

# Toronto-ish. Used only to reject a geocode that wandered to another
# continent — "Distillery District" alone matches places in several countries,
# and a single bad point stretches the map viewport across an ocean.
TORONTO_BOX = (43.4, 44.0, -79.8, -79.0)          # lat_min, lat_max, lon_min, lon_max

MODE_COLOUR = {
    "subway": (0, 122, 200),
    "streetcar": (208, 45, 45),
    "bus": (110, 110, 120),
    "walk": (140, 140, 150),
}

_point_cache: dict[str, tuple[float, float] | None] = {}


def _in_toronto(lat: float, lon: float) -> bool:
    lat_min, lat_max, lon_min, lon_max = TORONTO_BOX
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def locate(label: str, allow_network: bool = False) -> tuple[float, float] | None:
    """Coordinates for a leg endpoint, or None if we genuinely don't know.

    Two kinds of label arrive here and they resolve differently:

      "Spadina Ave at Nassau St South Side (8128)"   a real GTFS stop
      "Kensington Market"                            a neighbourhood, which
                                                     appears nowhere in the
                                                     feed — stop names are
                                                     intersections

    The database is tried first because it's instant, offline and exact.
    Geocoding is opt-in: it's a network call to a volunteer-run service, and
    a map is not worth hammering Nominatim for.

    Returns None rather than a guess. A fabricated coordinate would draw a
    confident line to the wrong place, which is worse than a gap — the same
    reason the itinerary reports "no route" instead of inventing one.
    """
    label = (label or "").strip()
    if not label:
        return None
    if label in _point_cache:
        return _point_cache[label]

    found = None
    if paths.TRANSIT_DB.exists():
        conn = sqlite3.connect(paths.readonly_uri(paths.TRANSIT_DB), uri=True)
        try:
            found = constraints._stop_coords(conn, label)
        finally:
            conn.close()

    if found is None and "(" in label:
        # Drop a trailing qualifier and try the bare name: "Yorkdale Station
        # (northbound platform)" -> "Yorkdale Station". The two platforms are
        # metres apart, so for a MAP either is right.
        #
        # Deliberately here and NOT in the departure check. Approximating
        # which side of a platform to draw a dot on is cartography;
        # approximating which platform a train leaves from is a claim someone
        # would act on. Same asymmetry as shapes versus times.
        bare = label.split("(")[0].strip()
        if bare and paths.TRANSIT_DB.exists():
            conn = sqlite3.connect(paths.readonly_uri(paths.TRANSIT_DB), uri=True)
            try:
                found = constraints._stop_coords(conn, bare)
            finally:
                conn.close()

    if found is None and allow_network:
        try:
            payload = json.loads(geo.geocode(f"{label}, Toronto"))
            found = (float(payload["lat"]), float(payload["lon"]))
        except Exception:                                   # noqa: BLE001
            found = None

    if found and not _in_toronto(*found):
        # A place name that matched somewhere else entirely. One such point
        # zooms the map out to fit two continents.
        found = None

    _point_cache[label] = found
    return found


def map_layers(itinerary, allow_network: bool = False) -> dict:
    """Points and paths for the journey, ready to hand to a map widget.

    Kept here rather than in ui.py so it can be tested: the interesting part
    is which endpoints resolve and which don't, and that needs no browser.
    """
    points: list[dict] = []
    paths_out: list[dict] = []
    unresolved: list[str] = []
    # Keyed by POSITION, not by label. The same stop arrives written both ways
    # — "Spadina Ave at Nassau St South Side" from one leg and the same name
    # with "(8128)" appended from the next — and de-duplicating on the string
    # drew two identical pins on top of each other. The thing that must be
    # unique is the place, not how it happened to be spelled.
    seen: set[tuple] = set()

    for index, leg in enumerate(itinerary.legs):
        start = locate(leg.origin, allow_network)
        end = locate(leg.destination, allow_network)

        for label, point in ((leg.origin, start), (leg.destination, end)):
            if point is None:
                if label not in unresolved:
                    unresolved.append(label)
                continue
            key = (round(point[0], 5), round(point[1], 5))
            if key in seen:
                continue
            seen.add(key)
            points.append({
                "name": constraints.split_stop_label(label)[0],
                "lat": point[0],
                "lon": point[1],
            })

        if start and end:
            # Real track geometry when we have it, a straight line when we
            # don't. Approximating a LINE is fine — it's cartography. The
            # project refuses to approximate a departure time because that's
            # a claim about the world; where the rails physically run is not
            # something the traveller acts on minute by minute.
            shape = None
            if leg.mode != "walk":
                shape = leg_shape(leg.route, leg.origin, leg.destination)

            paths_out.append({
                "path": shape or [[start[1], start[0]], [end[1], end[0]]],
                "exact": shape is not None,
                "colour": list(MODE_COLOUR.get(leg.mode, MODE_COLOUR["bus"])),
                "mode": leg.mode,
                "label": (f"walk {leg.duration_min} min" if leg.mode == "walk"
                          else f"{leg.route} · {leg.duration_min} min"),
                "dashed": leg.mode == "walk",
                "leg": index + 1,
            })

    return {"points": points, "paths": paths_out, "unresolved": unresolved}


def viewport(points: list[dict]) -> dict | None:
    """Centre and zoom that fit every point, with a little air."""
    if not points:
        return None
    lats = [p["lat"] for p in points]
    lons = [p["lon"] for p in points]
    span = max(max(lats) - min(lats), (max(lons) - min(lons)) * 0.72, 0.005)
    # Rough but stable: each halving of the span buys one zoom level.
    zoom = max(10.0, min(15.0, 13.5 - math.log2(span / 0.02) * 0.9))
    return {
        "latitude": (max(lats) + min(lats)) / 2,
        "longitude": (max(lons) + min(lons)) / 2,
        "zoom": round(zoom, 1),
    }


# ---------------------------------------------------------------------------
# Markup
#
# HTML in a "view" module looks odd until you remember why view.py exists:
# ui.py can't be imported, so nothing in it can be tested. Building the markup
# here means the timeline's structure — which legs get a warning, what a walk
# looks like, whether a route badge appears — is covered by the suite rather
# than only by looking at it.
# ---------------------------------------------------------------------------

HEX = {
    "subway": "#007ac8",
    "streetcar": "#e8483f",
    "bus": "#6e6e78",
    "walk": "#8b949e",
}


def _escape(text) -> str:
    """Stop a stop name from becoming markup.

    Stop names come from a public feed and answers come from a model. Neither
    is hostile here, but both are untrusted input to an HTML string, and
    `unsafe_allow_html` means exactly what it says.
    """
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def timeline_html(itinerary) -> str:
    """The itinerary as a vertical timeline."""
    risky = dict(itinerary.risky_connections())
    out = ['<div class="tl">']

    for index, leg in enumerate(itinerary.legs):
        colour = HEX.get(leg.mode, HEX["bus"])
        walking = leg.mode == "walk"
        badge = ("WALK" if walking
                 else f"{_escape(leg.route)} · {_escape(leg.mode)}")
        out.append(
            f'<div class="leg{" walking" if walking else ""}" '
            f'style="--dot:{colour}">'
            f'<div class="card">'
            f'<div class="time">{_escape(to_civil(leg.depart))}</div>'
            f'<div class="body">'
            f'<span class="route">{badge}</span>'
            f'<div class="where">{_escape(constraints.split_stop_label(leg.origin)[0])}'
            f' → {_escape(constraints.split_stop_label(leg.destination)[0])}</div>'
            f'</div>'
            f'<div class="dur">{leg.duration_min} min</div>'
            f'</div>'
        )
        if index in risky:
            out.append(
                f'<div class="gap">⚠ only {risky[index]} min to make the '
                f'{_escape(itinerary.legs[index + 1].route)}</div>')
        out.append("</div>")

    last = itinerary.legs[-1] if itinerary.legs else None
    if last:
        out.append(
            f'<div class="leg" style="--dot:#3fb950"><div class="card">'
            f'<div class="time">{_escape(to_civil(last.arrive))}</div>'
            f'<div class="body"><span class="route">ARRIVE</span>'
            f'<div class="where">{_escape(constraints.split_stop_label(last.destination)[0])}'
            f'</div></div></div></div>')

    out.append("</div>")
    return "".join(out)


def stats_html(result) -> str:
    """The three headline numbers as cards, colour-coded by what they mean."""
    values = badge_values(result)
    tone = {
        "Schedule": "ok" if not result.violations else "bad",
        "Times": "bad" if result.no_schedule_data else "ok",
        "Grounding": _grounding_tone(result.grounding.get("coverage")),
    }
    cards = "".join(
        f'<div class="stat {tone.get(k, "")}">'
        f'<div class="k">{_escape(k)}</div>'
        f'<div class="v">{_escape(v)}</div></div>'
        for k, v in values.items()
    )
    return f'<div class="stats">{cards}</div>'


def _grounding_tone(coverage) -> str:
    if coverage is None:
        return ""
    # Matches agent.MIN_GROUNDING: below 0.85 the agent itself pushes back,
    # so the badge shouldn't look content when the loop wasn't.
    return "ok" if coverage >= 0.85 else ("warn" if coverage >= 0.6 else "bad")


def hero_html(subtitle: str, chips: list[str]) -> str:
    pills = "".join(f'<span class="chip">{_escape(c)}</span>' for c in chips)
    return (
        '<div class="hero">'
        '<h1>Toronto Transit Agent</h1>'
        f'<p>{_escape(subtitle)}</p>'
        f'<div class="chips">{pills}</div>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# JSON for the web front end
# ---------------------------------------------------------------------------

def result_to_dict(result, allow_network: bool = False) -> dict:
    """A PlanResult as plain JSON for the browser.

    The browser renders; Python decides. Every judgement that could differ
    between front ends — is this verified, are the times real, what colour is
    a 504 — is made once here. The CLI, the old Streamlit page and the web app
    disagreeing about whether an itinerary is trustworthy would be far worse
    than any of them looking plain.
    """
    itinerary = result.itinerary
    payload = {
        "question": result.question,
        "error": result.error,
        "flags": result.flags,
        "stats": badge_values(result),
        "tone": {
            "Schedule": "ok" if not result.violations else "bad",
            "Times": "bad" if result.no_schedule_data else "ok",
            "Grounding": _grounding_tone(result.grounding.get("coverage")),
        },
        "violations": [
            {"kind": v.kind, "detail": v.detail, "fix": v.fix}
            for v in result.violations
        ],
        "grounding": {
            "coverage": result.grounding.get("coverage"),
            "claims": result.grounding.get("claims"),
            "unsupported": result.grounding.get("unsupported", []),
        },
        "steps": result.steps,
        "legs": [],
        "caveats": [],
        "map": {"points": [], "paths": [], "unresolved": []},
        "viewport": None,
        "feasible": False,
        "summary": "",
    }
    if itinerary is None:
        return payload

    payload.update(
        summary=itinerary.summary,
        feasible=itinerary.feasible,
        infeasible_reason=itinerary.infeasible_reason,
        caveats=list(itinerary.caveats),
        total_min=itinerary.total_min,
        transfers=itinerary.transfers,
    )

    risky = dict(itinerary.risky_connections())
    for index, leg in enumerate(itinerary.legs):
        payload["legs"].append({
            "mode": leg.mode,
            "route": leg.route,
            "colour": HEX.get(leg.mode, HEX["bus"]),
            "depart": to_civil(leg.depart),
            "arrive": to_civil(leg.arrive),
            "origin": constraints.split_stop_label(leg.origin)[0],
            "destination": constraints.split_stop_label(leg.destination)[0],
            "minutes": leg.duration_min,
            # Attached to the leg it concerns, not collected into a separate
            # list the front end would have to re-associate.
            "warning": (f"only {risky[index]} min to make the "
                        f"{itinerary.legs[index + 1].route}")
                       if index in risky else None,
        })

    if itinerary.feasible:
        payload["map"] = map_layers(itinerary, allow_network=allow_network)
        payload["viewport"] = viewport(payload["map"]["points"])
    return payload


def has_shapes() -> bool:
    """Is the optional geometry table loaded?"""
    if not paths.TRANSIT_DB.exists():
        return False
    conn = sqlite3.connect(paths.readonly_uri(paths.TRANSIT_DB), uri=True)
    try:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='shapes'"
        ).fetchone() is not None
    finally:
        conn.close()


def leg_shape(route: str, origin: str, destination: str,
              service_id: str = "1") -> list[list[float]] | None:
    """The real track between two stops, as [[lon, lat], ...].

    Sliced by shape_dist_traveled — the distance along the route recorded
    against both the shape points and the stop times. That makes it a range
    query rather than a nearest-point guess, which matters on any route that
    passes near its own path twice: a loop, or a branch that doubles back.
    Snapping by proximity would silently pick the wrong side.

    Returns None when the geometry isn't available, so the caller can fall
    back to a straight line rather than draw nothing. An approximate line is
    honest here in a way an approximate DEPARTURE never would be — this is
    cartography, not a claim about when a streetcar leaves.
    """
    if not route or not has_shapes():
        return None

    name_a, id_a = constraints.split_stop_label(origin)
    name_b, id_b = constraints.split_stop_label(destination)

    conn = sqlite3.connect(paths.readonly_uri(paths.TRANSIT_DB), uri=True)
    try:
        # Resolve the route label the SAME way the schedule verifier does.
        # This matched route_short_name exactly and so found nothing for
        # "510 Spadina" or "504A" — both of which constraints.resolve_route
        # has handled since stage 7. The fix already existed in this codebase
        # and simply wasn't reused, which is its own lesson: a normalisation
        # that only one caller applies is a normalisation the next caller
        # will get wrong.
        resolved = constraints.resolve_route(conn, route)
        if not resolved:
            return None
        route = resolved

        # One trip that serves BOTH stops, in order. Any such trip's shape is
        # the path this leg follows.
        row = conn.execute(
            """
            -- CAST because load_gtfs stores every GTFS column as TEXT, and
            -- shapes.shape_dist_traveled is REAL. Comparing the two compares
            -- a number against a string: SQLite returns nothing, quietly.
            -- Exactly the trap the zero-padded departure times set, one
            -- column over. Numeric-looking TEXT is still TEXT.
            SELECT t.shape_id,
                   CAST(a.shape_dist_traveled AS REAL),
                   CAST(b.shape_dist_traveled AS REAL)
            FROM trips t
            JOIN routes r      ON r.route_id = t.route_id
            JOIN stop_times a  ON a.trip_id = t.trip_id
            JOIN stop_times b  ON b.trip_id = t.trip_id
            JOIN stops sa      ON sa.stop_id = a.stop_id
            JOIN stops sb      ON sb.stop_id = b.stop_id
            WHERE r.route_short_name = ?
              AND t.service_id = ?
              AND (sa.stop_id = ? OR sa.stop_name LIKE ?)
              AND (sb.stop_id = ? OR sb.stop_name LIKE ?)
              AND CAST(b.stop_sequence AS INTEGER) > CAST(a.stop_sequence AS INTEGER)
              AND t.shape_id IS NOT NULL AND t.shape_id != ''
              -- The first stop of a trip has an empty distance, not a zero.
              AND a.shape_dist_traveled != '' AND b.shape_dist_traveled != ''
            LIMIT 1
            """,
            (route, service_id,
             id_a or "", f"%{name_a}%",
             id_b or "", f"%{name_b}%"),
        ).fetchone()
        if not row:
            return None

        shape_id, start, end = row
        if start is None or end is None:
            return None

        points = conn.execute(
            """
            SELECT shape_pt_lon, shape_pt_lat FROM shapes
            WHERE shape_id = ? AND shape_dist_traveled BETWEEN ? AND ?
            ORDER BY shape_dist_traveled
            """,
            (shape_id, min(start, end), max(start, end)),
        ).fetchall()
    finally:
        conn.close()

    # Two points is what a straight line already gives us; below that the
    # slice found nothing useful.
    return [[lon, lat] for lon, lat in points] if len(points) > 2 else None
