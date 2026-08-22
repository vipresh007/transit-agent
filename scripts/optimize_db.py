"""
Add the journey-planner indexes to an existing transit.db.

load_gtfs.py builds these now, but rebuilding costs a 33MB download and
several minutes. This adds them in place, in about a minute.

    python scripts/optimize_db.py

Why they matter: plan_journey self-joins stop_times to find every stop
downstream of a given stop on the same trip. Over 4.2M rows with only
single-column indexes, SQLite scans; with a composite index covering
(trip_id, stop_sequence, stop_id) it seeks. That's the difference between
a 17-second journey search and a 2-second one.
"""

# Run either way: `python scripts/optimize_db.py` or `python -m scripts.load_gtfs`.
# The first puts scripts/ on sys.path rather than the repo root, so `transit`
# would not be importable without this.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import os
import sqlite3
import time
from transit import paths

DB = paths.TRANSIT_DB

INDEXES = [
    ("ix_st_trip_seq",
     "CREATE INDEX IF NOT EXISTS ix_st_trip_seq "
     "ON stop_times(trip_id, stop_sequence, stop_id)"),
    ("ix_st_stop_trip",
     "CREATE INDEX IF NOT EXISTS ix_st_stop_trip "
     "ON stop_times(stop_id, trip_id, stop_sequence)"),
    ("ix_trips_service_route",
     "CREATE INDEX IF NOT EXISTS ix_trips_service_route "
     "ON trips(service_id, route_id, trip_id)"),
]


def main() -> None:
    if not os.path.exists(DB):
        raise SystemExit(f"{DB} not found. Run `python scripts/load_gtfs.py` first.")

    before = os.path.getsize(DB) >> 20
    conn = sqlite3.connect(DB)
    existing = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'")}

    for name, sql in INDEXES:
        if name in existing:
            print(f"  {name}: already present")
            continue
        print(f"  {name}: building...", end="", flush=True)
        t0 = time.time()
        conn.execute(sql)
        print(f" {time.time() - t0:.0f}s")

    print("  ANALYZE...", end="", flush=True)
    t0 = time.time()
    conn.execute("ANALYZE")
    conn.commit()
    print(f" {time.time() - t0:.0f}s")
    conn.close()

    after = os.path.getsize(DB) >> 20
    print(f"\nDone. {DB}: {before}MB -> {after}MB")
    print("Try:  python -c \"import tools,time;t=time.time();"
          "tools.plan_journey(43.6552,-79.4023,43.6503,-79.3592);"
          "print(f'{time.time()-t:.1f}s')\"")


if __name__ == "__main__":
    main()
