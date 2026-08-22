"""The same crew as a LangGraph state graph, with checkpointing
and approval pauses. Stage 10's comparison.

    python graph.py "plan me a Saturday in Toronto"

A three-line launcher. The code lives in `transit.pipeline.graph`; this file exists
so the commands in the README keep working after the reorganisation, and so
`python graph.py` still does the obvious thing from a fresh clone.
"""

from transit.pipeline.graph import main

if __name__ == "__main__":
    main()
