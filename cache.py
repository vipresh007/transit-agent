"""Response cache: replay identical model requests instead of re-sending them.

The dominant cost of iterating on an agent is the wait. A 12-step run takes
minutes and burns quota, and most code changes -- a parser, a validator, a
render function -- don't change what we send to the model at all. Replaying
those makes the loop instant.

Two hard-won rules baked in here:

  1. THE KEY COVERS THE WHOLE REQUEST. It originally hashed model, messages
     and a `use_tools` boolean, so editing a tool's description left the cache
     replaying answers produced by the OLD tools. A cache keyed on a
     convenient subset of the input will serve answers to a question you no
     longer asked.

  2. IT ONLY WORKS IF RUNS ARE DETERMINISTIC. One differing response changes
     the message history for every later step, so a single miss poisons the
     rest. Hence TEMPERATURE=0 and a pinned PROVIDER -- mid-run failover
     changes the model, which changes the key.

Off by default. A cache that's silently on will eventually convince you a bug
is fixed when you're reading a replay.
"""

import hashlib
import json
import os
from pathlib import Path

from openai.types.chat import ChatCompletion

from tools import TOOL_SCHEMAS

ENABLED = os.getenv("CACHE") == "1"
CACHE_DIR = Path(os.getenv("CACHE_DIR", ".cache"))
STATS = {"hits": 0, "misses": 0}

# The tool schemas are part of every request, so they belong in the key.
TOOLS_FINGERPRINT = hashlib.sha256(
    json.dumps(TOOL_SCHEMAS, sort_keys=True).encode()
).hexdigest()[:12]


def key_for(model: str, messages: list, use_tools: bool, temperature: float) -> str:
    blob = json.dumps(
        {
            "model": model,
            "messages": messages,
            "tools": TOOLS_FINGERPRINT if use_tools else None,
            "temperature": temperature,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def read(key: str):
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        return ChatCompletion.model_validate_json(path.read_text("utf-8"))
    except Exception:
        # A corrupt or stale-schema entry should never break a run.
        path.unlink(missing_ok=True)
        return None


def write(key: str, response) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    try:
        (CACHE_DIR / f"{key}.json").write_text(
            response.model_dump_json(), encoding="utf-8"
        )
    except Exception:
        pass  # caching is an optimisation; never let it break the run


def summary() -> str:
    h, m = STATS["hits"], STATS["misses"]
    return f"cache {h} hit / {m} miss" if (h or m) else ""
