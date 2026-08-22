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

from transit.tools import memory
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
