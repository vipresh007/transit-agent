"""The raw loop. Prose answer, no schema, no schedule checks —
use it to test whether the TOOLS work.

    python agent.py "when is the last 501 from Yonge?"

A three-line launcher. The code lives in `transit.core.agent`; this file exists
so the commands in the README keep working after the reorganisation, and so
`python agent.py` still does the obvious thing from a fresh clone.
"""

from transit.core.agent import main

if __name__ == "__main__":
    main()
