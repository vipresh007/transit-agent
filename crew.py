"""Several INDEPENDENT questions researched in parallel, then
merged. Costs ~3-4x the requests; skips schedule verification.

    python crew.py "plan me a Saturday in Toronto: morning, lunch, afternoon"

A three-line launcher. The code lives in `transit.pipeline.crew`; this file exists
so the commands in the README keep working after the reorganisation, and so
`python crew.py` still does the obvious thing from a fresh clone.
"""

from transit.pipeline.crew import main

if __name__ == "__main__":
    main()
