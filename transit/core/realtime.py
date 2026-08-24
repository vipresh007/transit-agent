"""Live TTC vehicle positions. Route-level only, and deliberately so.

WHAT THIS MODULE WILL NEVER DO

It will never tell you a departure time. Not "your bus is 6 minutes late", not
"expected 12:08". `scripts/probe_rt.py` measured why, and the number is worth
keeping in front of you:

    stop_id join rate ............................ 59.3%
    of those, actually the same physical stop ..... 1.1%

Both feeds number stops with small integers, so thousands collide by
arithmetic. Feed route 23 is Dawes Rd; its "matched" stops resolved to
Bathurst St, three kilometres away. A per-stop prediction built on that join
would be live, precise, confident and wrong — and it would look exactly like a
correct one. Silence is a better failure than a plausible lie.

WHAT SURVIVES

Route identity. The route number IS the route_id in both feeds; every vehicle
carrying a route resolved to one we know. And a position is a latitude and a
longitude, which need no join at all. So:

    "23 streetcars are on the 504 right now, here is where they are"

That claim needs route + coordinates and nothing else. It cannot be wrong
about a time because it never states one. This is the same asymmetry the
project already applies to map geometry versus departure times: approximate
the picture, never the claim.

DESIGN NOTES

  * Failure is always None, never an exception. A dead feed must degrade to a
    map without vehicles, not a broken itinerary. Real-time is a garnish.
  * Results cache for TTL seconds. The page polls, and hammering a public
    feed for the same 30-second snapshot is rude and pointless.
  * The decoder is fed bytes, not a URL, so tests run against the saved
    data/rt_sample/*.pb files with no network.
"""

from __future__ import annotations

import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path

VEHICLES_URL = "https://bustime.ttc.ca/gtfsrt/vehicles"

# The feed republishes roughly every 30s; 20 keeps it fresh without polling
# faster than the data changes.
TTL = 20.0
TIMEOUT = 8.0

# Toronto, generously. A vehicle reporting a position in the Atlantic is a
# decoder bug or a garbage row, and either way it must not reach the map.
BOX = (43.4, 44.0, -79.8, -79.0)

_lock = threading.Lock()
_cached: tuple[float, list] | None = None


@dataclass(frozen=True)
class Vehicle:
    """One vehicle, right now. Note the absence of any time field."""
    route_id: str
    lat: float
    lon: float
    bearing: float | None = None
    label: str | None = None


# --------------------------------------------------------------------------
# protobuf wire format
#
# Depending on `gtfs-realtime-bindings` would mean the web app dies with an
# ImportError when someone clones this repo and doesn't read the README. The
# wire format is four cases wide and the field numbers are a published, frozen
# part of the GTFS-RT spec, so decoding the handful we need is cheaper than
# the dependency. We read four fields and ignore everything else.
# --------------------------------------------------------------------------

def _varint(buf: bytes, i: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = buf[i]
        i += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, i
        shift += 7


def _fields(buf: bytes):
    """Yield (field_number, payload) for one protobuf message."""
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        number, wire = key >> 3, key & 7
        if wire == 0:                                   # varint
            value, i = _varint(buf, i)
        elif wire == 2:                                 # length-delimited
            size, i = _varint(buf, i)
            value, i = buf[i:i + size], i + size
        elif wire == 5:                                 # 32-bit
            value, i = buf[i:i + 4], i + 4
        elif wire == 1:                                 # 64-bit
            value, i = buf[i:i + 8], i + 8
        else:                                           # 3 and 4 are removed
            raise ValueError(f"unsupported wire type {wire}")
        yield number, value


def _get(buf: bytes, number: int) -> list:
    return [value for found, value in _fields(buf) if found == number]


def _one(buf: bytes, number: int):
    values = _get(buf, number)
    return values[0] if values else None


def _f32(raw) -> float | None:
    return struct.unpack("<f", raw)[0] if raw else None


def decode(raw: bytes) -> list:
    """Decode a VehiclePositions feed. Never raises on a malformed entity.

    Field numbers from the GTFS-RT spec:
      FeedMessage.entity = 2, FeedEntity.vehicle = 4,
      VehiclePosition.trip = 1 / .position = 2 / .vehicle = 8,
      TripDescriptor.route_id = 5,
      Position.latitude = 1 / .longitude = 2 / .bearing = 3,
      VehicleDescriptor.label = 2
    """
    out: list = []
    try:
        entities = _get(raw, 2)
    except Exception:                                        # noqa: BLE001
        # The OUTER scan can fail too — a truncated download ends mid-varint,
        # and `buf[i]` on a short buffer raises IndexError. The first version
        # of this guarded only the per-entity loop, so a corrupt header took
        # the whole page down. One byte of garbage found it.
        return out

    for entity in entities:
        try:
            for vehicle in _get(entity, 4):
                trip = _one(vehicle, 1)
                position = _one(vehicle, 2)
                if position is None:
                    continue                    # no coordinates, nothing to draw

                route = _one(trip, 5) if trip else None
                if not route:
                    # ~38% of the feed. A vehicle with no route can't be tied
                    # to a leg, so it is dropped rather than drawn as a
                    # mystery dot.
                    continue

                lat = _f32(_one(position, 1))
                lon = _f32(_one(position, 2))
                if lat is None or lon is None:
                    continue
                if not (BOX[0] <= lat <= BOX[1] and BOX[2] <= lon <= BOX[3]):
                    continue

                descriptor = _one(vehicle, 8)
                label = _one(descriptor, 2) if descriptor else None

                out.append(Vehicle(
                    route_id=route.decode("utf-8", "replace"),
                    lat=round(lat, 5),
                    lon=round(lon, 5),
                    bearing=_f32(_one(position, 3)),
                    label=label.decode("utf-8", "replace") if label else None,
                ))
        except Exception:                                # noqa: BLE001
            # One corrupt entity must not cost us the other 1,310.
            continue
    return out


def decode_file(path: Path) -> list:
    """Decode a saved feed. This is how the tests reach real data offline."""
    return decode(Path(path).read_bytes())


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------

def fetch(url: str = VEHICLES_URL, force: bool = False) -> list | None:
    """Live vehicles, or None if the feed is unavailable for any reason.

    None and [] mean different things and the caller must be able to tell
    them apart: None is "we don't know", [] is "nothing is running". Collapsing
    those two is how you end up showing an empty map during an outage and
    calling it a quiet night.
    """
    global _cached

    with _lock:
        if _cached and not force and time.monotonic() - _cached[0] < TTL:
            return _cached[1]

    try:
        import requests
    except ImportError:
        return None

    try:
        response = requests.get(
            url, timeout=TIMEOUT,
            headers={"User-Agent": "transit-agent/1.0 (learning project)"})
        if response.status_code != 200 or not response.content:
            return None
        if response.content[:1] in (b"<", b"{"):
            return None                     # an error page wearing a 200
        vehicles = decode(response.content)
    except Exception:                                    # noqa: BLE001
        return None

    with _lock:
        _cached = (time.monotonic(), vehicles)
    return vehicles


def for_routes(routes, url: str = VEHICLES_URL) -> dict | None:
    """Vehicles grouped by route, limited to the routes asked for.

    `routes` holds whatever the itinerary called them — "504", "504A",
    "510 Spadina". Resolved through constraints.resolve_route so the feed's
    plain route_id matches, because this project has already learned six
    times that a checker demanding one spelling fires on correct answers
    written another way.
    """
    vehicles = fetch(url)
    if vehicles is None:
        return None

    import sqlite3

    from transit import paths
    from transit.verify import constraints

    # route_id -> the label the itinerary used, so the browser can group them.
    wanted: dict = {}
    if paths.TRANSIT_DB.exists():
        conn = sqlite3.connect(paths.readonly_uri(paths.TRANSIT_DB), uri=True)
        try:
            for label in routes:
                if not label:
                    continue
                # resolve_route returns a route_short_name. The feed speaks
                # route_id. Those are the same string for TTC surface routes
                # today, and assuming that is precisely the mistake that made
                # the stop join look 59% correct — so go through the table.
                short = constraints.resolve_route(conn, label)
                if short is None:
                    continue
                for (route_id,) in conn.execute(
                        "SELECT route_id FROM routes WHERE route_short_name = ?",
                        (short,)):
                    wanted[str(route_id)] = label
        finally:
            conn.close()
    else:
        wanted = {str(r): r for r in routes if r}

    grouped: dict = {}
    for vehicle in vehicles:
        label = wanted.get(vehicle.route_id)
        if label is None:
            continue
        grouped.setdefault(label, []).append(
            {"lat": vehicle.lat, "lon": vehicle.lon,
             "bearing": vehicle.bearing, "label": vehicle.label})
    return grouped
