"""
Smoke test: check the tools work before involving the model.

Debugging an agent means separating two questions:
  1. Do my tools actually work?
  2. Is the model calling them correctly?

This script answers (1). It uses no API key and burns no quota. If something
breaks later, run this first — it tells you which half to look at.

    python tests/smoke_test.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import tools


def check(label, fn):
    print(f"\n[{label}]")
    try:
        result = fn()
    except Exception as exc:
        print(f"  FAIL  {type(exc).__name__}: {exc}")
        return None
    preview = str(result)
    print(f"  ok    {preview[:200]}{'...' if len(preview) > 200 else ''}")
    return result


def main():
    raw = check("geocode", lambda: tools.geocode("CN Tower, Toronto"))
    if raw is None:
        print("\nGeocoding failed — check your internet connection.")
        sys.exit(1)

    loc = json.loads(raw)
    lat, lon = loc["lat"], loc["lon"]

    check("weather", lambda: tools.get_weather(lat, lon))
    check("find_pois/museum", lambda: tools.find_pois(lat, lon, "museum", 2000))
    check("find_pois/subway", lambda: tools.find_pois(lat, lon, "subway_station", 1000))

    # Tools must fail gracefully. The agent loop hands these strings back to
    # the model so it can correct itself, so a readable message matters.
    check("bad category (should return a message, not raise)",
          lambda: tools.find_pois(lat, lon, "casino"))
    check("nonsense place (should return a message, not raise)",
          lambda: tools.geocode("asdfghjkl qwertyuiop"))

    print("\nDone. If everything above says 'ok', your tools are fine —")
    print("any remaining problem is in the model's tool selection.")


if __name__ == "__main__":
    main()
