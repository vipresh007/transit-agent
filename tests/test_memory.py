"""Persistent memory: what should survive a session, and what must not.

    python tests/test_memory.py

The failure this exists to prevent is quiet and cumulative. "I need to be
there by 3pm" saved as a standing preference means every future journey
inherits a 3pm deadline. Nothing in next month's conversation explains the
odd answer, and the traveller has no idea a database is constraining them.
"""

import os
import sys
import tempfile
from pathlib import Path

from _harness import check, section

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Point at a scratch database BEFORE importing, so tests never touch real
# memory. A test suite that mutates the user's saved preferences would be a
# spectacular own goal.
_tmp = tempfile.mkdtemp()
os.environ["MEMORY_DB"] = str(Path(_tmp) / "memory.db")

from transit.verify import constraints  # noqa: E402
from transit.tools import memory       # noqa: E402


def reset():
    if os.path.exists(memory.DB_PATH):
        os.remove(memory.DB_PATH)


def test_scope_is_load_bearing():
    section("standing preferences vs one-off constraints")
    reset()

    r = memory.remember("latest_arrival", "15:00:00", scope="trip",
                        reason="needs to be there by 3 today")
    check("a one-off is refused", "Not stored" in r)
    check("and the refusal explains why", "future journey" in r)
    check("nothing persisted", memory.load()[0], {})

    r = memory.remember("avoid_modes", "bus", scope="standing",
                        reason="would rather walk than take a bus")
    check("a durable preference is saved", "Saved standing preference" in r)
    check("and reads back", memory.load()[0], {"avoid_modes": "bus"})

    check("an invalid scope is rejected",
          "must be 'standing' or 'trip'" in memory.remember("x", "y", scope="forever"))


def test_conflicts_and_updates():
    section("newer statements win")
    reset()

    memory.remember("min_transfer_min", "10", scope="standing")
    r = memory.remember("min_transfer_min", "12", scope="standing")
    check("an update reports the change", "'10' -> '12'" in r)
    check("and the new value sticks", memory.load()[0]["min_transfer_min"], "12")

    check("forgetting works", "Forgot" in memory.forget("min_transfer_min"))
    check("and it's gone", "min_transfer_min" not in memory.load()[0])
    check("forgetting something absent is not an error",
          "Nothing stored" in memory.forget("min_transfer_min"))


def test_free_form_notes():
    section("facts that aren't enforceable preferences")
    reset()

    r = memory.remember("travelling_with", "a stroller", scope="standing")
    check("an unknown key becomes a note", "Noted" in r)
    check("notes are recalled", "stroller" in memory.recall())
    # A note must not silently become a constraint — nothing can enforce
    # "travelling with a stroller", and pretending otherwise would be worse
    # than not storing it.
    check("but not a preference", memory.load()[0], {})


def test_precedence():
    section("what the traveller says now beats what they said before")
    reset()
    memory.remember("min_transfer_min", "12", scope="standing")
    memory.remember("avoid_modes", "bus", scope="standing")

    p, applied = memory.apply_to(constraints.Preferences())
    check("memory fills unset values", p.min_transfer_min, 12)
    check("and list values too", p.avoid_modes, ["bus"])
    check("what was applied is reported", sorted(applied),
          ["avoid_modes=bus", "min_transfer_min=12"])

    # The important direction: an explicit setting must survive.
    p2, applied2 = memory.apply_to(
        constraints.Preferences(min_transfer_min=3, avoid_modes=["subway"]))
    check("an explicit value is not overwritten", p2.min_transfer_min, 3)
    check("nor an explicit list", p2.avoid_modes, ["subway"])
    check("and memory reports applying nothing", applied2, [])


def test_robustness():
    section("memory must never break a run")
    reset()

    check("reading empty memory is fine", memory.load(), ({}, []))
    check("recall says so", "Nothing remembered yet" in memory.recall())

    # A corrupt stored value would otherwise raise inside apply_to and take
    # the whole run down. Memory is an enhancement; it must fail quietly.
    conn = memory._conn()
    conn.execute("INSERT INTO preferences (key, value) VALUES (?, ?)",
                 ("min_transfer_min", "not-a-number"))
    conn.commit()
    conn.close()

    p, applied = memory.apply_to(constraints.Preferences())
    check("a corrupt value is skipped, not raised", p.min_transfer_min, 5)
    check("and isn't reported as applied", applied, [])


def test_tool_surface():
    section("what the agent sees")

    names = {s["function"]["name"] for s in memory.TOOL_SCHEMAS}
    check("three memory tools are exposed", names,
          {"recall_preferences", "remember", "forget_preference"})
    check("every tool has a function", set(memory.TOOL_FUNCTIONS), names)

    remember_schema = next(s for s in memory.TOOL_SCHEMAS
                           if s["function"]["name"] == "remember")
    desc = remember_schema["function"]["description"]
    # The scope distinction is the whole point; the description has to teach
    # it, because the model decides scope and nothing downstream can fix a
    # wrong choice.
    check("the description explains standing vs trip", "standing" in desc and "trip" in desc)
    check("it gives concrete examples", "3pm" in desc)
    check("and says what to do when unsure", "unsure" in desc)
    check("scope is required",
          "scope" in remember_schema["function"]["parameters"]["required"])


if __name__ == "__main__":
    for fn in (test_scope_is_load_bearing, test_conflicts_and_updates,
               test_free_form_notes, test_precedence, test_robustness,
               test_tool_surface):
        fn()
    from _harness import PASSED
    print(f"\n{PASSED['n']} checks passed")
