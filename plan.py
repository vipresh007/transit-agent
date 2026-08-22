"""One journey, fully checked: typed itinerary, constraint
verification against the real timetable, repair loop, memory.

    python plan.py "how do I get from Kensington Market to the Distillery District?"

A three-line launcher. The code lives in `transit.pipeline.plan`; this file exists
so the commands in the README keep working after the reorganisation, and so
`python plan.py` still does the obvious thing from a fresh clone.
"""

from transit.pipeline.plan import main

if __name__ == "__main__":
    main()
