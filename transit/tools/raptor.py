"""RAPTOR: journeys with any number of transfers, in one pass.

WHY THIS REPLACED THE PAIRWISE SEARCH.

`plan_journey` used to try each nearby origin stop against each nearby
destination stop, in SQL: 16 pairs for a direct ride, and for one transfer,
16 pairs each running two large self-joins to find stops reachable from one
end that sit near stops reaching the other.

That works, and it does not extend. Two transfers would be the cross product
of two of those reachable sets, and one transfer already took **308 seconds**
on Kensington Market to Scarborough Town Centre — because the further apart
the endpoints are, the more stops each end reaches, and both sides of the
intersection grow together. The honest failure message it produced ("this
probably needs two transfers") was true and useless.

RAPTOR (Delling, Pajor & Werneck) turns the transfer count into a LOOP
COUNTER instead of a join depth:

    round 1  = everywhere you can reach on one vehicle
    round 2  = everywhere reachable from all of those, on one more vehicle
    round k  = ... k vehicles

Each round scans every route at most once, so k transfers costs k passes
rather than a k-way product. Three transfers is three rounds.

THE CENTRAL DEFINITION: a "route" here is a PATTERN — a unique ordered stop
sequence — not a GTFS route_id. The 504 has branches and short turns, and
merging them would let the algorithm board a trip that never visits the stop
it thinks comes next. TTC service 1 has 233 route_ids and 1,115 patterns.

WHAT IT OPTIMISES. Earliest arrival, then fewest transfers among equal
arrivals. It does not model fares, crowding, or a preference for staying put.

WHAT IT DOES NOT APPROXIMATE. Every departure and arrival comes from
stop_times. The walking legs at each end and between stops are straight-line
distance over an assumed speed, which is the same approximation the rest of
this project already makes for footpaths — and it is applied to DURATION, not
to a claimed departure. A journey this returns can be checked leg by leg
against the timetable, and constraints.py does exactly that.
"""

from __future__ import annotations

import collections
import math
import sqlite3
import time
from dataclasses import dataclass

from transit import paths

INF = 1 << 30

# Straight-line metres per second for walking, plus the multiplier that turns
# crow-flies into something like a street route. 1.4 m/s is a normal walking
# pace; 1.35 is the usual detour factor for a gridded city.
WALK_SPEED = 1.4
DETOUR = 1.35

# How far a traveller will walk to change vehicles, and to start or finish.
TRANSFER_RADIUS_M = 400
ACCESS_RADIUS_M = 800

# Minimum time to get off one vehicle and onto another at a different stop is
# covered by the footpath; this is the extra for the same stop or platform.
MIN_CHANGE_S = 60

_CACHE: dict = {}


def _secs(text: str) -> int:
    """GTFS clock to seconds. Hours run past 24 for after-midnight service."""
    h, m, s = text.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def civil(seconds: int) -> str:
    """Back to a GTFS-shaped clock string, keeping 24+ hours as GTFS does.

    CLAMPED AT ZERO. Callers subtract the walk to the first stop, and five
    minutes before a 00:02 departure is negative — which rendered as
    '-1:57:00', not a time, and failed schema validation downstream. The old
    `_shift()` helper learned this the same way; the lesson had to move here
    when RAPTOR replaced it, because deleting code deletes what it knew.
    """
    seconds = max(0, seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


@dataclass
class Timetable:
    """Everything RAPTOR needs, indexed by integer rather than by string.

    String keys cost a hash on every one of the millions of comparisons a
    query makes. Stops become indices once, here, and become names again only
    when a journey is handed back.
    """
    stop_ids: list            # index -> GTFS stop_id
    stop_names: list          # index -> human name
    stop_pos: list            # index -> (lat, lon)
    index: dict               # GTFS stop_id -> index
    patterns: list            # pattern -> list of stop indices
    pattern_routes: list      # pattern -> route_short_name
    pattern_heads: list       # pattern -> headsign (its final stop's name)
    pattern_trips: list       # pattern -> list of trips; trip = [(arr, dep), ...]
    stop_patterns: list       # stop -> list of (pattern, position)
    footpaths: list           # stop -> list of (stop, seconds)
    service_id: str

    def nearest(self, lat: float, lon: float, radius_m: int) -> list:
        """Stops within radius, as (index, metres). Linear scan: 9,307 stops
        is nothing next to the query that follows, and a spatial index here
        would be optimising the wrong end."""
        out = []
        for i, (slat, slon) in enumerate(self.stop_pos):
            d = _metres(lat, lon, slat, slon)
            if d <= radius_m:
                out.append((i, d))
        out.sort(key=lambda p: p[1])
        return out


def _metres(lat1, lon1, lat2, lon2) -> float:
    """Equirectangular distance. Over a city it is within a metre of haversine
    and costs a fraction as much, and this runs 87 million times during a
    footpath build."""
    x = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    y = lat2 - lat1
    return 111320.0 * math.sqrt(x * x + y * y)


def _walk_seconds(metres: float) -> int:
    return max(MIN_CHANGE_S, int(metres * DETOUR / WALK_SPEED))


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------

def build(service_id: str = "1", db_path=None) -> Timetable:
    """Read one service day out of SQLite into RAPTOR's structures.

    Roughly 4 seconds and ~200MB for TTC weekday service. Cached per
    service_id, because doing this per tool call would dwarf the query.
    """
    db_path = db_path or paths.TRANSIT_DB
    conn = sqlite3.connect(paths.readonly_uri(db_path), uri=True)
    try:
        stop_ids, stop_names, stop_pos, index = [], [], [], {}
        for sid, name, lat, lon in conn.execute(
                "SELECT stop_id, stop_name, CAST(stop_lat AS REAL), "
                "CAST(stop_lon AS REAL) FROM stops"):
            if lat is None or lon is None:
                continue
            index[sid] = len(stop_ids)
            stop_ids.append(sid)
            stop_names.append(name)
            stop_pos.append((lat, lon))

        # One ordered pass, grouped in Python. Sorting in SQL and grouping
        # here beats 38,000 separate queries by two orders of magnitude.
        rows = conn.execute(
            """SELECT st.trip_id, st.stop_id, st.arrival_time,
                      st.departure_time, r.route_short_name
               FROM stop_times st
               JOIN trips t  ON t.trip_id = st.trip_id
               JOIN routes r ON r.route_id = t.route_id
               WHERE t.service_id = ?
               ORDER BY st.trip_id, CAST(st.stop_sequence AS INT)""",
            (service_id,)).fetchall()
    finally:
        conn.close()

    trips: dict = collections.OrderedDict()
    route_of: dict = {}
    for trip_id, stop_id, arr, dep, route in rows:
        slot = index.get(stop_id)
        if slot is None:
            continue
        trips.setdefault(trip_id, []).append((slot, _secs(arr), _secs(dep)))
        route_of[trip_id] = route

    # A pattern is a unique ordered stop sequence. Merging the 504's branches
    # into one route would let a scan board a trip that never reaches the
    # stop it expects next — a wrong answer, not a slow one.
    by_pattern: dict = {}
    for trip_id, stops in trips.items():
        if len(stops) < 2:
            continue
        key = tuple(s for s, _, _ in stops)
        by_pattern.setdefault(key, []).append(
            (trip_id, [(a, d) for _, a, d in stops]))

    patterns, pattern_trips, pattern_routes, pattern_heads = [], [], [], []
    stop_patterns: list = [[] for _ in stop_ids]
    for key, entries in by_pattern.items():
        # Sorted by departure from the first stop, so the scan can take the
        # first catchable trip and stop looking.
        entries.sort(key=lambda e: e[1][0][1])
        p = len(patterns)
        patterns.append(list(key))
        pattern_trips.append([times for _, times in entries])
        pattern_routes.append(route_of.get(entries[0][0], "?"))
        # The pattern's last stop is what the sign on the front of the
        # vehicle says, and it's the one thing a traveller uses to check
        # they boarded the right direction.
        pattern_heads.append(stop_names[key[-1]])
        for position, stop in enumerate(key):
            stop_patterns[stop].append((p, position))

    footpaths = _footpaths(stop_pos, stop_patterns)

    return Timetable(stop_ids, stop_names, stop_pos, index, patterns,
                     pattern_routes, pattern_heads, pattern_trips,
                     stop_patterns, footpaths, service_id)


def _footpaths(stop_pos: list, stop_patterns: list) -> list:
    """Walkable links between stops, via a grid rather than all pairs.

    9,307 stops is 87 million pairs. Bucketing by a ~400m grid cell and
    comparing only the nine neighbouring cells turns that into a few hundred
    thousand — the difference between a minute and a fraction of a second.
    """
    cell = TRANSFER_RADIUS_M / 111320.0
    grid: dict = collections.defaultdict(list)
    for i, (lat, lon) in enumerate(stop_pos):
        grid[(int(lat / cell), int(lon / cell))].append(i)

    out: list = [[] for _ in stop_pos]
    for i, (lat, lon) in enumerate(stop_pos):
        if not stop_patterns[i]:
            continue                      # nothing calls here; nowhere to walk to
        gx, gy = int(lat / cell), int(lon / cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((gx + dx, gy + dy), ()):
                    if j == i or not stop_patterns[j]:
                        continue
                    d = _metres(lat, lon, *stop_pos[j])
                    if d <= TRANSFER_RADIUS_M:
                        out[i].append((j, _walk_seconds(d)))
    return out


def timetable(service_id: str = "1") -> Timetable:
    """Cached build. First call pays ~4s; later calls are free."""
    if service_id not in _CACHE:
        _CACHE[service_id] = build(service_id)
    return _CACHE[service_id]


# --------------------------------------------------------------------------
# the query
# --------------------------------------------------------------------------

@dataclass
class Leg:
    mode: str                 # "ride" or "walk"
    route: str | None
    from_stop: int
    to_stop: int
    depart: int
    arrive: int
    headsign: str | None = None


def query(tt: Timetable, origins: list, destinations: list,
          after: int, max_rounds: int = 5) -> list | None:
    """Earliest arrival from any origin to any destination.

    `origins` and `destinations` are (stop_index, walk_seconds) — the walk to
    the first stop and from the last. Returns legs, or None.

    max_rounds is the transfer ceiling plus one. Five means four transfers,
    which covers anything in Toronto; the old tool managed one.
    """
    n = len(tt.stop_ids)
    best = [INF] * n              # earliest arrival at each stop, any round
    prev = [INF] * n              # earliest arrival as of the PREVIOUS round
    parent: list = [None] * n     # how we got here, for reconstruction
    target = INF
    target_stop = None

    walk_out = dict(destinations)

    marked = set()
    for stop, walk in origins:
        arrival = after + walk
        if arrival < best[stop]:
            best[stop] = prev[stop] = arrival
            parent[stop] = ("start", walk)
            marked.add(stop)

    for _ in range(max_rounds):
        if not marked:
            break

        # Every pattern touched by a marked stop, remembering the EARLIEST
        # position we could board it — scanning a pattern from further along
        # than necessary loses journeys.
        queue: dict = {}
        for stop in marked:
            for pattern, position in tt.stop_patterns[stop]:
                if pattern not in queue or position < queue[pattern]:
                    queue[pattern] = position
        marked = set()

        for pattern, start in queue.items():
            stops = tt.patterns[pattern]
            trips = tt.pattern_trips[pattern]
            trip = None            # times of the trip currently boarded
            boarded_at = None

            for position in range(start, len(stops)):
                stop = stops[position]

                if trip is not None:
                    arrival = trip[position][0]
                    # Pruning against the best known target as well as the
                    # stop is what keeps RAPTOR near-linear: a path that
                    # cannot beat the current answer is abandoned at once.
                    if arrival < min(best[stop], target):
                        best[stop] = arrival
                        parent[stop] = ("ride", pattern, boarded_at,
                                        trip[boarded_at][1], arrival)
                        marked.add(stop)
                        if stop in walk_out:
                            finish = arrival + walk_out[stop]
                            if finish < target:
                                target, target_stop = finish, stop

                # Can an earlier trip on this pattern be caught here? Uses
                # last round's arrival, not this round's: boarding on the
                # strength of a journey found in the same round would count
                # one vehicle as two.
                ready = prev[stop]
                if ready == INF:
                    continue
                ready += MIN_CHANGE_S if parent[stop] and \
                    parent[stop][0] == "ride" else 0
                if trip is None or ready <= trip[position][1]:
                    caught = _earliest_trip(trips, position, ready)
                    if caught is not None and (trip is None
                                               or caught[position][1] < trip[position][1]):
                        trip, boarded_at = caught, position

        # Footpaths. A separate relaxation because walking is not a vehicle:
        # it must not consume a round, or a two-block walk would cost the
        # same as a transfer.
        for stop in list(marked):
            arrival = best[stop]
            for other, walk in tt.footpaths[stop]:
                candidate = arrival + walk
                if candidate < min(best[other], target):
                    best[other] = candidate
                    parent[other] = ("walk", stop, arrival, candidate)
                    marked.add(other)
                    if other in walk_out:
                        finish = candidate + walk_out[other]
                        if finish < target:
                            target, target_stop = finish, other

        prev = best[:]

    if target_stop is None:
        return None
    return _rebuild(tt, parent, target_stop)


def _earliest_trip(trips: list, position: int, ready: int):
    """First trip on this pattern departing `position` at or after `ready`.

    Trips are sorted by departure from the pattern's first stop, and TTC
    vehicles on one pattern do not overtake each other, so that order holds
    at every position and a binary search is valid.
    """
    lo, hi = 0, len(trips)
    while lo < hi:
        mid = (lo + hi) // 2
        if trips[mid][position][1] < ready:
            lo = mid + 1
        else:
            hi = mid
    return trips[lo] if lo < len(trips) else None


def _rebuild(tt: Timetable, parent: list, stop: int) -> list:
    """Walk the parent pointers back to the start and turn them into legs."""
    legs: list = []
    guard = 0
    while stop is not None and guard < 64:
        guard += 1
        step = parent[stop]
        if step is None or step[0] == "start":
            break
        if step[0] == "ride":
            _, pattern, boarded_at, depart, arrive = step
            board = tt.patterns[pattern][boarded_at]
            legs.append(Leg("ride", tt.pattern_routes[pattern],
                            board, stop, depart, arrive,
                            tt.pattern_heads[pattern]))
            stop = board
        else:
            _, origin, depart, arrive = step
            legs.append(Leg("walk", None, origin, stop, depart, arrive))
            stop = origin
    legs.reverse()
    return legs
