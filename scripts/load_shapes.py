"""Load shapes.txt — the real track geometry — into transit.db.

    python scripts/load_shapes.py

WHY SEPARATELY. load_gtfs.py deliberately skips shapes: none of the agent's
reasoning needs it. Routes, trips, stop_times, stops and calendar answer every
scheduling question, and shapes.txt is 17 MB of pure cartography. It's loaded
here, on demand, by the one thing that wants it — the map.

WHAT IT BUYS. A leg drawn as a straight line between two stops cuts diagonally
across blocks; the 510 appears to fly over Chinatown rather than run down
Spadina. With shapes the route follows the actual street.

THE PART THAT MAKES IT EXACT. Both shapes.txt and stop_times.txt carry
`shape_dist_traveled` — distance along the route to that point. So slicing the
geometry for one leg is a range query rather than a nearest-point guess:

    take the shape points between the distance at stop A
    and the distance at stop B

Guessing by proximity would have been wrong exactly where it matters — a route
that passes near its own path twice, like a loop or a there-and-back branch.
"""

# Run either way: `python scripts/load_shapes.py` or `python -m scripts.load_shapes`.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import csv
import io
import sqlite3
import sys
import zipfile

from transit import paths

BATCH = 50_000


def _load() -> None:
    if not paths.GTFS_ZIP.exists():
        sys.exit(f"{paths.GTFS_ZIP} not found. Run scripts/load_gtfs.py first.")
    if not paths.TRANSIT_DB.exists():
        sys.exit(f"{paths.TRANSIT_DB} not found. Run scripts/load_gtfs.py first.")

    conn = sqlite3.connect(paths.TRANSIT_DB)
    conn.execute("DROP TABLE IF EXISTS shapes")
    conn.execute("""
        CREATE TABLE shapes (
            shape_id TEXT,
            shape_pt_lat REAL,
            shape_pt_lon REAL,
            shape_pt_sequence INTEGER,
            shape_dist_traveled REAL
        )
    """)

    rows, total = [], 0
    with zipfile.ZipFile(paths.GTFS_ZIP) as archive:
        with archive.open("shapes.txt") as raw:
            # utf-8-sig: GTFS files very often carry a BOM, which would end up
            # glued to the first column name. Classic silent GTFS bug.
            reader = csv.DictReader(io.TextIOWrapper(raw, "utf-8-sig"))
            for row in reader:
                rows.append((
                    row["shape_id"],
                    float(row["shape_pt_lat"]),
                    float(row["shape_pt_lon"]),
                    int(row["shape_pt_sequence"]),
                    float(row.get("shape_dist_traveled") or 0.0),
                ))
                if len(rows) >= BATCH:
                    conn.executemany("INSERT INTO shapes VALUES (?,?,?,?,?)", rows)
                    total += len(rows)
                    rows.clear()
                    print(f"  {total:,} points…", end="\r", file=sys.stderr)

    if rows:
        conn.executemany("INSERT INTO shapes VALUES (?,?,?,?,?)", rows)
        total += len(rows)

    # The map slices by (shape_id, distance), so index exactly that.
    conn.execute("CREATE INDEX ix_shapes_id_dist "
                 "ON shapes(shape_id, shape_dist_traveled)")
    conn.commit()
    conn.close()
    print(f"\nLoaded {total:,} shape points into {paths.TRANSIT_DB}")


def main() -> None:
    try:
        _load()
    except KeyError as exc:
        sys.exit(f"shapes.txt is missing a column this expects: {exc}")


if __name__ == "__main__":
    main()
