"""Stage 8: what the agent remembers between sessions.

Two kinds of memory, and conflating them is the classic mistake:

  STANDING PREFERENCES  durable facts about the traveller. "I'd rather walk
                        than take a bus." True next week too.

  TRIP CONSTRAINTS      true for one journey. "I need to be there by 3pm."
                        Remembering it means every future trip inherits a 3pm
                        deadline nobody asked for.

An agent that can't tell them apart accumulates a pile of stale constraints
and slowly becomes useless in a way that's hard to debug — the answers get
worse and nothing in the current conversation explains why. So `scope` is a
required field, not an optional one, and the tool description spends most of
its words on the distinction.

Storage is SQLite next to the other databases. Preferences use a fixed key
vocabulary so they map cleanly onto constraints.Preferences; anything else
goes in `notes` as free text the agent can read but nothing enforces.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from transit import paths

DB_PATH = paths.MEMORY_DB

# The keys that map onto constraints.Preferences. Free-form facts go to notes.
# A fixed vocabulary is what lets stored memory become enforced constraints
# rather than just more prompt text the model may or may not honour.
PREFERENCE_KEYS = {
    "earliest_departure": "Earliest acceptable departure, HH:MM:SS",
    "latest_arrival": "Latest acceptable arrival, HH:MM:SS",
    "min_transfer_min": "Minimum minutes to change vehicles (integer)",
    "max_transfers": "Maximum transfers willing to make (integer)",
    "avoid_modes": "Comma-separated modes to avoid: bus, streetcar, subway",
}


def _conn(readonly: bool = False):
    if readonly and not os.path.exists(DB_PATH):
        return None
    conn = sqlite3.connect(
        paths.readonly_uri(DB_PATH) if readonly else DB_PATH, uri=readonly
    )
    if not readonly:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS preferences (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                reason TEXT,
                created TEXT,
                updated TEXT
            );
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                created TEXT
            );
        """)
    return conn


# ---------------------------------------------------------------------------
# Tools the agent can call
# ---------------------------------------------------------------------------

def remember(key: str, value: str, scope: str = "trip",
             reason: str = "") -> str:
    """Store something about the traveller. Only `scope='standing'` persists."""
    scope = (scope or "trip").strip().lower()
    if scope not in {"standing", "trip"}:
        return (f"scope must be 'standing' or 'trip', not {scope!r}. "
                f"Use 'standing' only for durable preferences.")

    if scope == "trip":
        # Deliberately a no-op with an explanation. The agent proposing to
        # remember a one-off is the common case, and refusing loudly teaches
        # the distinction better than silently discarding it.
        return (
            f"Not stored. {key!r} was marked as a one-off for this trip, and "
            f"one-off constraints must not persist — a 3pm deadline from today "
            f"would silently apply to every future journey. It still applies "
            f"to THIS conversation; just don't save it."
        )

    key = key.strip().lower()
    if key not in PREFERENCE_KEYS:
        # Unknown key: keep it as a note rather than rejecting. Free-form
        # facts ("I'm travelling with a stroller") are worth remembering even
        # though nothing can enforce them.
        return add_note(f"{key}: {value}" + (f" ({reason})" if reason else ""))

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    conn = _conn()
    try:
        existing = conn.execute(
            "SELECT value FROM preferences WHERE key = ?", (key,)
        ).fetchone()
        conn.execute(
            """
            INSERT INTO preferences (key, value, reason, created, updated)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                reason = excluded.reason,
                updated = excluded.updated
            """,
            (key, str(value).strip(), reason, now, now),
        )
        conn.commit()
    finally:
        conn.close()

    if existing and existing[0] != str(value).strip():
        return (f"Updated standing preference {key}: {existing[0]!r} -> "
                f"{value!r}. The newer statement wins.")
    return f"Saved standing preference {key} = {value!r}."


def add_note(text: str) -> str:
    conn = _conn()
    try:
        conn.execute("INSERT INTO notes (text, created) VALUES (?, ?)",
                     (text.strip(), time.strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
    finally:
        conn.close()
    return f"Noted: {text.strip()}"


def recall() -> str:
    """Everything remembered about the traveller."""
    prefs, notes = load()
    if not prefs and not notes:
        return ("Nothing remembered yet. Standing preferences can be saved "
                "with remember(scope='standing').")
    lines = []
    if prefs:
        lines.append("Standing preferences:")
        lines += [f"  {k} = {v}" for k, v in sorted(prefs.items())]
    if notes:
        lines.append("Notes:")
        lines += [f"  {n}" for n in notes]
    return "\n".join(lines)


def forget(key: str) -> str:
    conn = _conn()
    try:
        n = conn.execute("DELETE FROM preferences WHERE key = ?",
                         (key.strip().lower(),)).rowcount
        conn.commit()
    finally:
        conn.close()
    return (f"Forgot {key!r}." if n else
            f"Nothing stored under {key!r}. Current memory:\n{recall()}")


# ---------------------------------------------------------------------------
# Reading memory back into constraints
# ---------------------------------------------------------------------------

def load() -> tuple[dict, list[str]]:
    conn = _conn(readonly=True)
    if conn is None:
        return {}, []
    try:
        prefs = dict(conn.execute("SELECT key, value FROM preferences"))
        notes = [r[0] for r in conn.execute(
            "SELECT text FROM notes ORDER BY id DESC LIMIT 10")]
    except sqlite3.OperationalError:
        return {}, []
    finally:
        conn.close()
    return prefs, notes


def apply_to(prefs_obj):
    """Overlay remembered preferences onto a Preferences object.

    Precedence: environment/CLI beats memory. Something the traveller said
    just now must override something they said last week — otherwise stored
    memory becomes impossible to escape without editing a database, which is
    the point at which people start distrusting the whole feature.
    """
    stored, _notes = load()
    if not stored:
        return prefs_obj, []

    applied = []
    for key, raw in stored.items():
        # Only fill values the caller didn't set.
        current = getattr(prefs_obj, key, None)
        explicitly_set = (
            current is not None
            and not (key == "min_transfer_min" and current == 5)
            and current != []
        )
        if explicitly_set:
            continue

        try:
            if key in ("min_transfer_min", "max_transfers"):
                setattr(prefs_obj, key, int(raw))
            elif key == "avoid_modes":
                setattr(prefs_obj, key,
                        [m.strip() for m in raw.split(",") if m.strip()])
            else:
                setattr(prefs_obj, key, raw)
            applied.append(f"{key}={raw}")
        except (ValueError, TypeError):
            continue   # a corrupt stored value must not break a run

    return prefs_obj, applied


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "recall_preferences",
            "description": (
                "Read what is remembered about this traveller from previous "
                "conversations. Call this at the START of any planning task."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "Save something about the traveller. "
                "CRITICAL — scope decides whether it persists:\n"
                "  'standing' = true of this person generally. "
                "\"I'd rather walk than take a bus\", \"I never travel before "
                "9am\". Saved forever.\n"
                "  'trip' = true only for this journey. \"I need to be there "
                "by 3pm\", \"today I'm in a hurry\". NOT saved.\n"
                "When unsure, use 'trip'. A one-off saved as standing silently "
                "constrains every future journey and is very hard to notice."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "One of: " + ", ".join(PREFERENCE_KEYS)
                        + ". Anything else is kept as a free-form note.",
                    },
                    "value": {"type": "string"},
                    "scope": {"type": "string", "enum": ["standing", "trip"]},
                    "reason": {
                        "type": "string",
                        "description": "What the traveller said, in their words.",
                    },
                },
                "required": ["key", "value", "scope"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget_preference",
            "description": "Delete a stored standing preference by key.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "recall_preferences": lambda: recall(),
    "remember": remember,
    "forget_preference": forget,
}
