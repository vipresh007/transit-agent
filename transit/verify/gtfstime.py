"""GTFS clock arithmetic. No dependencies, so anything can use it.

This lived in schemas.py, which imports pydantic — so `to_civil`, a pure
string function, dragged a validation library into every consumer. The UI
helpers needed it, could not be imported without pydantic, and therefore
could not be tested in an environment that lacked it. That is how a
`.items()` on a tuple reached a browser.

Split out because the dependency was accidental, not because the code is
special: converting "25:23:00" to "1:23 AM (next day)" has nothing to do with
schemas.
"""

import re

# GTFS times run past midnight: '25:30:00' is 1:30am on the next calendar day,
# still part of the previous service day. Allow hours 0-47, and allow the
# unpadded single-digit form this feed actually uses.
GTFS_TIME = re.compile(r"^([0-9]|[0-3][0-9]|4[0-7]):[0-5][0-9]:[0-5][0-9]$")


def to_seconds(t: str) -> int:
    """Convert a GTFS time to seconds since midnight of the service day."""
    h, m, s = (int(p) for p in t.split(":"))
    return h * 3600 + m * 60 + s


def to_civil(t: str) -> str:
    """Human-readable form. '25:23:00' -> '1:23 AM (next day)'."""
    h, m, _ = (int(p) for p in t.split(":"))
    nextday = h >= 24
    h %= 24
    suffix = "AM" if h < 12 else "PM"
    display = h % 12 or 12
    return f"{display}:{m:02d} {suffix}" + (" (next day)" if nextday else "")
