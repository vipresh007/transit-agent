"""Is this itinerary actually possible?

Stage 3's Pydantic validators check SHAPE: times parse, arrivals follow
departures, legs don't overlap. All necessary, none sufficient. A perfectly
schema-valid itinerary can still claim:

  - a departure that isn't in the schedule
  - a route that doesn't serve the stop it's boarding at
  - a 2-minute transfer between platforms 900m apart
  - a walk covering 2km in 3 minutes
  - travel on a day the route doesn't run

Type validity is free and catches formatting. Semantic validity needs domain
knowledge checked against real data, and it's where the remaining errors live.

Every violation carries a `fix` string. That matters: handing the agent
"tight_transfer at leg 2" makes it guess, while "the gap is 1 min and you need
5; find a later departure for leg 3 or an earlier arrival for leg 2" tells it
what to change. The quality of a repair loop is mostly the quality of its
error messages.
"""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass, field
from transit import paths

DB_PATH = paths.TRANSIT_DB

# Walking faster than this is not walking. 5 km/h is a brisk pace; 7 allows
# for imprecise stop coordinates without waving through a claimed sprint.
MAX_WALK_KMH = 7.0

# Below this, you are relying on both vehicles being exactly on time.
DEFAULT_MIN_TRANSFER_MIN = 5


@dataclass
class Preferences:
    """User constraints. Stage 8 will persist these; for now they're per-run."""

    earliest_departure: str | None = None   # "09:00:00"
    latest_arrival: str | None = None       # "23:00:00"
    min_transfer_min: int = DEFAULT_MIN_TRANSFER_MIN
    max_transfers: int | None = None
    avoid_modes: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> Preferences:
        return cls(
            earliest_departure=os.getenv("PREF_EARLIEST") or None,
            latest_arrival=os.getenv("PREF_LATEST") or None,
            min_transfer_min=int(os.getenv("PREF_MIN_TRANSFER", DEFAULT_MIN_TRANSFER_MIN)),
            max_transfers=(int(os.getenv("PREF_MAX_TRANSFERS"))
                           if os.getenv("PREF_MAX_TRANSFERS") else None),
            avoid_modes=[m.strip() for m in
                         (os.getenv("PREF_AVOID") or "").split(",") if m.strip()],
        )

    def describe(self) -> str:
        bits = []
        if self.earliest_departure:
            bits.append(f"leave no earlier than {self.earliest_departure}")
        if self.latest_arrival:
            bits.append(f"arrive by {self.latest_arrival}")
        bits.append(f"allow >= {self.min_transfer_min} min per transfer")
        if self.max_transfers is not None:
            bits.append(f"at most {self.max_transfers} transfer(s)")
        if self.avoid_modes:
            bits.append(f"avoid {', '.join(self.avoid_modes)}")
        return "; ".join(bits)


@dataclass
class Violation:
    kind: str
    leg: int | None
    detail: str
    fix: str

    def __str__(self) -> str:
        where = f"leg {self.leg + 1}" if self.leg is not None else "itinerary"
        return f"[{self.kind}] {where}: {self.detail}  -> {self.fix}"


def _seconds(t: str) -> int:
    h, m, s = (int(p) for p in t.split(":"))
    return h * 3600 + m * 60 + s


def _conn():
    return sqlite3.connect(paths.readonly_uri(DB_PATH), uri=True)


# ---------------------------------------------------------------------------
# Checks against the real schedule
# ---------------------------------------------------------------------------

def resolve_route(conn, label: str) -> str | None:
    """Map however the answer named a route onto a real route_short_name.

    Exact matching produced two false positives on a correct itinerary:

      "510 Spadina"  short_name '510' + long_name 'Spadina' — how riders say it
      "504A King"    504A is a BRANCH, appearing in 990 headsigns but in no
                     route_short_name; the route is 504

    Both triggered an expensive repair round to fix nothing. A checker that
    demands the exact internal identifier will keep firing on correct answers
    written in human vocabulary, and every false positive costs a full agent
    run. Resolve the label instead of rejecting it.

    Deliberately permissive: "510 Bloor" resolves to 510 even though 510 is
    Spadina. This check only catches invented routes like "999"; whether the
    route serves that stop at that time is _departure_is_scheduled's job, and
    it validates route + stop + time together. Two loose overlapping checks
    beat one strict check that fires on correct answers.
    """
    label = (label or "").strip()
    if not label:
        return None

    def exists(name):
        return conn.execute(
            "SELECT 1 FROM routes WHERE route_short_name = ? LIMIT 1", (name,)
        ).fetchone() is not None

    if exists(label):
        return label

    first = label.split()[0]
    if exists(first):
        return first

    # Branch letters: 504A -> 504
    base = re.match(r"^(\d+)", first)
    if base and exists(base.group(1)):
        return base.group(1)

    # "510 Spadina" == short_name + ' ' + long_name, and "Line 1" is how the
    # subway is universally named (short_name '1', long_name
    # "Line 1 (Yonge-University)").
    row = conn.execute(
        """
        SELECT route_short_name FROM routes
        WHERE lower(route_short_name || ' ' || route_long_name) = lower(?)
           OR lower(route_long_name) = lower(?)
           OR lower(route_long_name) LIKE lower(?) || ' (%'
        LIMIT 1
        """,
        (label, label, label),
    ).fetchone()
    return row[0] if row else None


# find_nearby_stops returns "NAME (STOP_ID)", and the model copies that label
# into the itinerary — reasonably, since it disambiguates the platform. But it
# does not copy it the same way twice. Across real runs it produced:
#
#     Spadina Ave at Nassau St South Side
#     Spadina Ave at Nassau St South Side (8128)
#     Spadina Ave at Nassau St South Side (stop 8128)
#     Stop 8128 (Spadina Ave at Nassau St South Side)
#
# All four mean one platform. A parser that knows one of them silently fails
# on the rest: two whole journeys resolved to ZERO map points, and — far worse
# — the schedule verifier reported false violations on correct legs, because
# `LIKE '%Name (stop 8128)%'` matches no stop_name either.
#
# Fifth time in this project that a checker has failed on a label written a
# way it didn't expect. Parse the shapes the model ACTUALLY produces, not the
# one shape the tool emits.
STOP_ID_TRAILING = re.compile(r"\s*\(\s*(?:stop\s*)?(\d+)\s*\)\s*$", re.I)
STOP_ID_LEADING = re.compile(r"^\s*stop\s*#?\s*(\d+)\s*[:\-–]?\s*\((.+)\)\s*$", re.I)
STOP_ID_BARE = re.compile(r"^\s*stop\s*#?\s*(\d+)\s*[:\-–]\s*(.+)$", re.I)


def split_stop_label(label: str) -> tuple[str, str | None]:
    """Any of the ways a stop gets written -> (name, stop_id or None).

    The id is a gift when present: matching on stop_id pins the exact
    platform, which is stronger than a LIKE on a name shared by both
    directions of a street.
    """
    text = (label or "").strip()
    if not text:
        return "", None

    # "Stop 8128 (Spadina Ave at Nassau St South Side)"
    match = STOP_ID_LEADING.match(text)
    if match:
        return match.group(2).strip(), match.group(1)

    # "Stop 8128 - Spadina Ave at Nassau St"
    match = STOP_ID_BARE.match(text)
    if match:
        return match.group(2).strip(), match.group(1)

    # "... (8128)" or "... (stop 8128)"
    match = STOP_ID_TRAILING.search(text)
    if match:
        return text[:match.start()].strip(), match.group(1)

    return text, None


def _departure_is_scheduled(conn, route: str, stop_name: str, depart: str,
                            service_id: str = "1") -> bool:
    """Does a trip on this route really leave this stop at this time?

    The strongest single check available. Grounding catches times invented from
    nothing; this catches a time that IS real but belongs to a different stop
    or route — a copy-paste error the model makes when juggling several legs,
    and one that looks completely plausible in the output.
    """
    name, stop_id = split_stop_label(stop_name)

    if stop_id:
        # Exact platform. Also much faster than a leading-wildcard LIKE.
        row = conn.execute(
            """
            SELECT 1
            FROM trips t
            JOIN routes r      ON r.route_id = t.route_id
            JOIN stop_times st ON st.trip_id = t.trip_id
            WHERE r.route_short_name = ?
              AND t.service_id = ?
              AND st.stop_id = ?
              AND substr('0' || st.departure_time, -8) = ?
            LIMIT 1
            """,
            (route, service_id, stop_id, depart),
        ).fetchone()
        if row is not None:
            return True
        # An id that resolves to nothing is more likely a stale label than a
        # fabricated departure, so fall through to the name check rather than
        # reporting a violation on the strength of the parenthesis alone.

    return conn.execute(
        """
        SELECT 1
        FROM trips t
        JOIN routes r      ON r.route_id = t.route_id
        JOIN stop_times st ON st.trip_id = t.trip_id
        JOIN stops s       ON s.stop_id  = st.stop_id
        WHERE r.route_short_name = ?
          AND t.service_id = ?
          AND s.stop_name LIKE ?
          AND substr('0' || st.departure_time, -8) = ?
        LIMIT 1
        """,
        (route, service_id, f"%{name}%", depart),
    ).fetchone() is not None


def _stop_coords(conn, stop_name: str):
    """Coordinates for a stop label, which may carry a '(stop_id)' suffix.

    Same trap as the departure check, quieter consequences: an unresolved
    label returns None, the walk-speed check silently skips, and an impossible
    2-minute sprint across a kilometre goes unflagged. A checker that can't
    parse its input doesn't report a problem — it reports nothing.
    """
    name, stop_id = split_stop_label(stop_name)
    if stop_id:
        row = conn.execute(
            "SELECT CAST(stop_lat AS REAL), CAST(stop_lon AS REAL) FROM stops "
            "WHERE stop_id = ? LIMIT 1", (stop_id,)
        ).fetchone()
        if row:
            return row
    return conn.execute(
        "SELECT CAST(stop_lat AS REAL), CAST(stop_lon AS REAL) FROM stops "
        "WHERE stop_name LIKE ? LIMIT 1", (f"%{name}%",)
    ).fetchone()


def _metres(a, b) -> float:
    dlat = (a[0] - b[0]) * 111_320
    dlon = (a[1] - b[1]) * 111_320 * 0.723
    return (dlat ** 2 + dlon ** 2) ** 0.5


# ---------------------------------------------------------------------------

def verify(itinerary, prefs: Preferences | None = None,
           service_id: str = "1") -> list[Violation]:
    """Check an Itinerary against the schedule and the user's preferences."""
    prefs = prefs or Preferences()
    legs = itinerary.legs
    out: list[Violation] = []

    if not os.path.exists(DB_PATH):
        return out   # nothing to verify against; silence beats false alarms

    conn = _conn()
    try:
        for i, leg in enumerate(legs):
            if leg.mode == "walk":
                _check_walk(conn, out, i, leg)
                continue

            resolved = resolve_route(conn, leg.route) if leg.route else None
            if leg.route and resolved is None:
                out.append(Violation(
                    "unknown_route", i,
                    f"route {leg.route!r} is not in the TTC feed",
                    "Look the route up with query_transit and use its real "
                    "route_short_name, or drop this leg.",
                ))
                continue

            if resolved and not _departure_is_scheduled(
                conn, resolved, leg.origin, leg.depart, service_id
            ):
                out.append(Violation(
                    "departure_not_scheduled", i,
                    f"no {leg.route} departs {leg.origin!r} at {leg.depart} "
                    f"on service {service_id}",
                    "Re-run plan_journey or find_direct_trips for this leg and "
                    "use a departure that exists. Do not adjust the time by hand.",
                ))

        _check_transfers(out, legs, prefs)
        _check_preferences(out, itinerary, legs, prefs)
    finally:
        conn.close()

    return out


def _check_walk(conn, out, i, leg) -> None:
    a = _stop_coords(conn, leg.origin)
    b = _stop_coords(conn, leg.destination)
    if not a or not b:
        return   # endpoints aren't transit stops (a neighbourhood, a landmark)

    minutes = max(1, (_seconds(leg.arrive) - _seconds(leg.depart)) // 60)
    metres = _metres(a, b)
    kmh = (metres / 1000) / (minutes / 60)

    if kmh > MAX_WALK_KMH:
        out.append(Violation(
            "walk_too_fast", i,
            f"{metres:.0f}m in {minutes} min is {kmh:.1f} km/h",
            f"Allow about {metres / 80:.0f} min for this walk (80 m/min) and "
            f"shift the following legs later, or pick closer stops.",
        ))


def _check_transfers(out, legs, prefs: Preferences) -> None:
    for i, (a, b) in enumerate(zip(legs, legs[1:])):
        if a.mode == "walk" or b.mode == "walk":
            continue   # waiting isn't a transfer; only vehicle-to-vehicle counts
        gap = (_seconds(b.depart) - _seconds(a.arrive)) // 60
        if gap < prefs.min_transfer_min:
            out.append(Violation(
                "tight_transfer", i + 1,
                f"only {gap} min between arriving on the {a.route} and "
                f"departing on the {b.route}",
                f"You need >= {prefs.min_transfer_min} min. Find a later "
                f"{b.route} departure from {b.origin!r}.",
            ))


def _check_preferences(out, itinerary, legs, prefs: Preferences) -> None:
    if prefs.earliest_departure and legs:
        if _seconds(legs[0].depart) < _seconds(prefs.earliest_departure):
            out.append(Violation(
                "too_early", 0,
                f"departs {legs[0].depart}, earlier than the requested "
                f"{prefs.earliest_departure}",
                f"Re-plan with after_time={prefs.earliest_departure}.",
            ))

    if prefs.latest_arrival and legs:
        if _seconds(legs[-1].arrive) > _seconds(prefs.latest_arrival):
            out.append(Violation(
                "too_late", len(legs) - 1,
                f"arrives {legs[-1].arrive}, later than the requested "
                f"{prefs.latest_arrival}",
                "Find an earlier departure or a faster route.",
            ))

    if prefs.max_transfers is not None and itinerary.transfers > prefs.max_transfers:
        out.append(Violation(
            "too_many_transfers", None,
            f"{itinerary.transfers} transfers, more than the {prefs.max_transfers} "
            f"requested",
            "Look for a direct route, even if it takes longer.",
        ))

    for i, leg in enumerate(legs):
        if leg.mode in prefs.avoid_modes:
            out.append(Violation(
                "avoided_mode", i,
                f"uses {leg.mode}, which the traveller asked to avoid",
                f"Find an alternative that isn't a {leg.mode}.",
            ))


def preflight(prefs: Preferences, *coords, radius_m: int = 900) -> list[str]:
    """Warn BEFORE planning if a preference makes the trip impossible.

    A real run asked for Kensington Market -> Scarborough Town Centre with a
    standing "avoid bus" preference. Scarborough Centre is served by seventeen
    bus routes and nothing else — Line 3 RT closed in 2023 and isn't in the
    feed. Rather than report the conflict, the model produced Line 3 from
    stale training knowledge and wrote a confident 88-minute itinerary.

    Verification caught it afterwards, but only after ~34 requests. Checking
    up front costs one SQL query and lets the agent say "this needs a bus"
    instead of hunting for a route that cannot exist.
    """
    if not prefs.avoid_modes or not os.path.exists(DB_PATH):
        return []

    type_of = {"0": "streetcar", "1": "subway", "3": "bus"}
    avoided = {m.strip().lower() for m in prefs.avoid_modes}
    warnings = []

    conn = _conn()
    try:
        for lat, lon in coords:
            dlat = radius_m / 111_320
            dlon = radius_m / (111_320 * 0.723)
            rows = conn.execute(
                """
                SELECT DISTINCT r.route_short_name, r.route_type
                FROM stops s
                JOIN stop_times st ON st.stop_id = s.stop_id
                JOIN trips t       ON t.trip_id  = st.trip_id
                JOIN routes r      ON r.route_id = t.route_id
                WHERE CAST(s.stop_lat AS REAL) BETWEEN ? AND ?
                  AND CAST(s.stop_lon AS REAL) BETWEEN ? AND ?
                """,
                (lat - dlat, lat + dlat, lon - dlon, lon + dlon),
            ).fetchall()
            if not rows:
                continue

            modes = {type_of.get(rt, rt) for _name, rt in rows}
            usable = modes - avoided
            if not usable:
                warnings.append(
                    f"Every route within {radius_m}m of ({lat:.4f}, {lon:.4f}) "
                    f"is {'/'.join(sorted(modes))}, which the traveller asked "
                    f"to avoid. This journey CANNOT be planned without "
                    f"{'/'.join(sorted(modes))}. Say so plainly and let them "
                    f"decide — do not look for another route, and do not use "
                    f"a rail line from memory: Line 3 RT closed in 2023 and "
                    f"is not in this feed."
                )
    finally:
        conn.close()

    return warnings


def report(violations: list[Violation]) -> str:
    if not violations:
        return "No constraint violations."
    lines = [f"{len(violations)} constraint violation(s):"]
    lines += [f"  {v}" for v in violations]
    return "\n".join(lines)
