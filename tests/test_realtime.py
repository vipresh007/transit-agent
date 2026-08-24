"""Real-time vehicles, tested against real bytes with no network.

    python tests/test_realtime.py

WHY THIS SUITE EXISTS IN THIS SHAPE.

A GTFS-RT feed lives for about thirty seconds. It cannot be a fixture, so
scripts/probe_rt.py saves the raw protobuf to data/rt_sample/ and everything
here runs against those files forever after. If they're missing the suite
SKIPS rather than passes — a real-time decoder that "passes" because it was
never handed any data is the exact failure this project keeps rediscovering.

The most important tests here are the ones that assert what we DON'T do.
The probe measured that the feed's stop ids agree with our database only 1.1%
of the time — 59.3% of them collide by number while naming a different stop
entirely. So a per-stop arrival prediction would be live, precise, and wrong.
Two tests below exist purely to make sure nobody wires that back up later
without meaning to, because it's the kind of thing that looks like an
improvement in a diff.
"""

import sys
from pathlib import Path

from _harness import check, section

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from transit.core import realtime                           # noqa: E402

SAMPLE = ROOT / "data" / "rt_sample" / "vehicles.pb"
HAVE_SAMPLE = SAMPLE.exists()


def test_decodes_saved_feed():
    section("decoding real bytes")

    if not HAVE_SAMPLE:
        print("  SKIP — no data/rt_sample/vehicles.pb "
              "(run scripts/probe_rt.py once to save one)")
        return

    vehicles = realtime.decode_file(SAMPLE)
    check("decoded some vehicles", len(vehicles) > 100)
    check("every vehicle names a route",
          all(v.route_id for v in vehicles))
    check("every vehicle has coordinates",
          all(v.lat and v.lon for v in vehicles))
    # A float misread as the wrong width lands in the Atlantic, and a bounding
    # box catches that instantly where a "did it parse" check would not.
    check("every position is inside Toronto",
          all(realtime.BOX[0] <= v.lat <= realtime.BOX[1]
              and realtime.BOX[2] <= v.lon <= realtime.BOX[3]
              for v in vehicles))
    check("bearings, where present, are compass degrees",
          all(0 <= v.bearing <= 360 for v in vehicles if v.bearing is not None))


def test_a_vehicle_can_never_state_a_time():
    section("the line we refuse to cross")

    # 1.1%. That number is why there is no departure field here, and this test
    # is the tripwire if someone adds one.
    fields = set(realtime.Vehicle.__dataclass_fields__)
    check("Vehicle carries only route and position",
          fields == {"route_id", "lat", "lon", "bearing", "label"})
    for banned in ("depart", "arrive", "delay", "eta", "predicted", "stop_id"):
        check(f"Vehicle has no {banned!r} field", banned not in fields)

    source = (ROOT / "transit" / "core" / "realtime.py").read_text("utf-8")
    # StopTimeUpdate is field 2 of TripUpdate; if it ever appears here,
    # somebody started reading per-stop predictions.
    check("the module never decodes stop_time_update",
          "stop_time_update" not in source)


def test_garbage_never_raises():
    section("a bad feed degrades, it does not crash")

    for junk in (b"", b"\x00", b"not protobuf at all", b"\xff" * 64,
                 bytes(range(256))):
        try:
            out = realtime.decode(junk)
            check(f"decode({junk[:12]!r}...) returned a list",
                  isinstance(out, list))
        except Exception as exc:                             # noqa: BLE001
            check(f"decode({junk[:12]!r}...) did not raise",
                  False, f"raised {type(exc).__name__}: {exc}")

    if HAVE_SAMPLE:
        # Truncation is the realistic corruption: a connection dropped
        # mid-download gives you a valid prefix of a valid message.
        raw = SAMPLE.read_bytes()
        for cut in (10, len(raw) // 3, len(raw) - 5):
            try:
                out = realtime.decode(raw[:cut])
                check(f"truncated at {cut:,} bytes still returns a list",
                      isinstance(out, list))
            except Exception as exc:                         # noqa: BLE001
                check(f"truncated at {cut:,} bytes did not raise",
                      False, f"raised {type(exc).__name__}")


def test_unavailable_is_not_the_same_as_empty():
    section("None vs []")

    # The distinction the whole feature rests on. If a dead feed returned []
    # the map would say "no vehicles running" during an outage, which is a
    # confident false statement rather than a missing one.
    saved = realtime._cached
    try:
        realtime._cached = None
        got = realtime.fetch(url="http://127.0.0.1:9/definitely-not-listening")
        check("an unreachable feed returns None, not []", got is None)

        realtime._cached = (__import__("time").monotonic(), [])
        check("a genuinely empty feed returns []", realtime.fetch() == [])
    finally:
        realtime._cached = saved


def test_grouping_resolves_human_route_names():
    section("route labels")

    if not HAVE_SAMPLE:
        print("  SKIP — no saved sample")
        return
    from transit import paths
    if not paths.TRANSIT_DB.exists():
        print("  SKIP — no transit.db")
        return

    import time as _time
    saved = realtime._cached
    try:
        # Serve the saved feed through the cache so no network is touched.
        realtime._cached = (_time.monotonic(), realtime.decode_file(SAMPLE))

        grouped = realtime.for_routes(["504 King", "510 Spadina"])
        check("both human-written labels resolved",
              set(grouped) == {"504 King", "510 Spadina"}, True)
        check("504 King found vehicles", len(grouped.get("504 King", [])) > 0)
        check("labels come back as the itinerary wrote them",
              "504 King" in grouped)

        # The subway is not in this feed. Returning nothing is correct; the
        # UI is responsible for saying so rather than showing a blank map.
        check("Line 1 yields no vehicles",
              not realtime.for_routes(["Line 1"]).get("Line 1"))
        check("an invented route yields nothing",
              not realtime.for_routes(["999"]).get("999"))
    finally:
        realtime._cached = saved


def test_view_declares_which_routes_can_be_live():
    section("live_routes")

    from types import SimpleNamespace

    from transit.pipeline import view

    # live_routes reads two attributes, so it is tested with two attributes.
    # Reaching for the real pydantic Itinerary here would make this suite skip
    # wherever pydantic isn't installed — and a skipped test that prints
    # nothing looks exactly like a passing one, which is the oldest lesson in
    # this repo.
    leg = lambda mode, route: SimpleNamespace(mode=mode, route=route)  # noqa: E731
    itinerary = SimpleNamespace(legs=[
        leg("walk", None),
        leg("subway", "Line 1"),
        leg("streetcar", "504"),
        leg("bus", "29"),
    ])
    live = view.live_routes(itinerary)

    check("streetcar and bus are offered", live == ["29", "504"], True)
    check("the subway is excluded", "Line 1" not in live)
    check("walking legs contribute nothing", None not in live)
    check("no itinerary yields no routes", view.live_routes(None) == [])


if __name__ == "__main__":
    for fn in (test_decodes_saved_feed,
               test_a_vehicle_can_never_state_a_time,
               test_garbage_never_raises,
               test_unavailable_is_not_the_same_as_empty,
               test_grouping_resolves_human_route_names,
               test_view_declares_which_routes_can_be_live):
        fn()
    from _harness import PASSED
    print(f"\n{PASSED['n']} checks passed")
