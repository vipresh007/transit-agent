"""
Stage 2: load the TTC GTFS feed into SQLite.

GTFS is just a zip of CSVs with a documented schema. Every transit agency
that publishes open data publishes this same shape, so what you learn here
transfers to ~6000 other feeds worldwide.

    python load_gtfs.py

Downloads ~35 MB, produces transit.db (~1 GB, mostly stop_times). Takes a
few minutes. Both are gitignored.
"""

import csv
import io
import os
import sqlite3
import sys
import zipfile

import requests

CKAN = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/package_show"
DATASET = "ttc-routes-and-schedules"
DB_PATH = "transit.db"
ZIP_PATH = os.path.join("data", "ttc_gtfs.zip")

# The files we care about. GTFS has more, but these five answer almost
# every scheduling question you'd want to ask.
#
# The relationships, which are the actual thing to learn:
#   routes    one route ("501 Queen")
#     -> trips        one vehicle run along that route on a given service day
#          -> stop_times   one row per stop on that trip, with arrival time
#               -> stops        where that stop physically is
#   calendar / calendar_dates    which days a service_id actually runs
WANTED = [
    "stops.txt",
    "routes.txt",
    "trips.txt",
    "stop_times.txt",
    "calendar.txt",
    "calendar_dates.txt",
]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_stop_times_stop ON stop_times(stop_id)",
    "CREATE INDEX IF NOT EXISTS ix_stop_times_trip ON stop_times(trip_id)",
    # Composite covering indexes for the journey planner's self-join on
    # stop_times (find all stops downstream of X on the same trip). With only
    # the single-column indexes above, each interchange search costs ~0.5s and
    # a full journey search took 17 seconds.
    "CREATE INDEX IF NOT EXISTS ix_st_trip_seq ON stop_times(trip_id, stop_sequence, stop_id)",
    "CREATE INDEX IF NOT EXISTS ix_st_stop_trip ON stop_times(stop_id, trip_id, stop_sequence)",
    "CREATE INDEX IF NOT EXISTS ix_trips_route ON trips(route_id)",
    "CREATE INDEX IF NOT EXISTS ix_trips_service ON trips(service_id)",
    "CREATE INDEX IF NOT EXISTS ix_stops_name ON stops(stop_name)",
]


def resolve_download_url() -> str:
    """Ask the City's data portal where the current zip lives.

    Hardcoding the URL would work until the TTC republishes (roughly every
    six weeks) and the resource ID changes. Resolving it costs one request.
    """
    r = requests.get(CKAN, params={"id": DATASET}, timeout=30)
    r.raise_for_status()
    for resource in r.json()["result"]["resources"]:
        if resource.get("format", "").upper() == "ZIP":
            return resource["url"]
    raise RuntimeError("No ZIP resource found in the TTC dataset")


def download(url: str) -> None:
    os.makedirs("data", exist_ok=True)
    if os.path.exists(ZIP_PATH):
        print(f"Using cached {ZIP_PATH} (delete it to re-download)")
        return

    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(ZIP_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r  {done >> 20} / {total >> 20} MB ({pct:.0f}%)",
                          end="", flush=True)
    print()


def load_table(conn: sqlite3.Connection, zf: zipfile.ZipFile, filename: str) -> int:
    table = filename.removesuffix(".txt")

    with zf.open(filename) as raw:
        # utf-8-sig: GTFS files very often carry a BOM, which would otherwise
        # end up glued to your first column name. Classic silent GTFS bug.
        reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig"))
        header = next(reader)
        cols = [c.strip() for c in header]

        conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute(
            f"CREATE TABLE {table} ({', '.join(f'{c} TEXT' for c in cols)})"
        )
        placeholders = ",".join("?" * len(cols))
        insert = f"INSERT INTO {table} VALUES ({placeholders})"

        count = 0
        batch = []
        for row in reader:
            # Tolerate ragged rows rather than crashing on one bad line.
            if len(row) != len(cols):
                row = (row + [""] * len(cols))[: len(cols)]
            batch.append(row)
            if len(batch) >= 50_000:
                conn.executemany(insert, batch)
                count += len(batch)
                batch = []
                print(f"\r  {table}: {count:,} rows", end="", flush=True)
        if batch:
            conn.executemany(insert, batch)
            count += len(batch)

    conn.commit()
    print(f"\r  {table}: {count:,} rows")
    return count


def main() -> None:
    download(resolve_download_url())

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    # Bulk-load settings. Safe here because we can always rebuild from the zip.
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")

    with zipfile.ZipFile(ZIP_PATH) as zf:
        available = set(zf.namelist())
        print(f"\nFiles in feed: {len(available)}")
        for filename in WANTED:
            if filename not in available:
                print(f"  {filename}: MISSING from feed, skipping")
                continue
            load_table(conn, zf, filename)

    print("\nBuilding indexes (this is the slow part)...")
    for stmt in INDEXES:
        conn.execute(stmt)
    conn.commit()

    print("\nSanity checks:")
    checks = [
        ("routes", "SELECT COUNT(*) FROM routes"),
        ("stops", "SELECT COUNT(*) FROM stops"),
        ("trips", "SELECT COUNT(*) FROM trips"),
        ("stop_times", "SELECT COUNT(*) FROM stop_times"),
        ("times past 24:00 (next-day service)",
         "SELECT COUNT(*) FROM stop_times WHERE departure_time >= '24:00:00'"),
    ]
    for label, sql in checks:
        print(f"  {label}: {conn.execute(sql).fetchone()[0]:,}")

    print("\nA few routes:")
    for rid, short, long_ in conn.execute(
        "SELECT route_id, route_short_name, route_long_name FROM routes LIMIT 5"
    ):
        print(f"  {short:>4}  {long_}")

    conn.close()
    size_mb = os.path.getsize(DB_PATH) >> 20
    print(f"\nWrote {DB_PATH} ({size_mb} MB)")


if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        sys.exit(1)
