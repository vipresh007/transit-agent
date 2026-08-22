"""Per-thread containers that behave like a plain dict and list.

Stage 9 runs several agents at once, and that breaks an assumption every
earlier stage relied on: that there is exactly one run in flight, so its state
can live in a module-level global.

`LAST_RUN` and `trace.EVENTS` were globals. Two concurrent agents would
interleave their flags and tool-call events into the same objects — no error,
no crash, just a trace that mixes two conversations and a `times_retrieved`
count that belongs to neither. The nastiest kind of concurrency bug: silent,
and only visible if you already suspect it.

The fix is to make the containers thread-local while keeping their interface
identical, so `LAST_RUN["steps"] = 3` and `EVENTS.append(...)` keep working
everywhere and no call site has to change. Global mutable state is fine right
up until you need two of something.
"""

from __future__ import annotations

import threading
from collections.abc import MutableMapping, MutableSequence


class ThreadLocalDict(MutableMapping):
    """A dict whose contents are private to each thread."""

    def __init__(self, factory):
        self._factory = factory
        self._local = threading.local()

    @property
    def _data(self) -> dict:
        if not hasattr(self._local, "data"):
            self._local.data = self._factory()
        return self._local.data

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return repr(self._data)

    def reset(self) -> None:
        self._local.data = self._factory()

    def snapshot(self) -> dict:
        """A plain copy, safe to hand to another thread or serialise."""
        return {
            k: (sorted(v) if isinstance(v, set) else v)
            for k, v in self._data.items()
        }


class ThreadLocalList(MutableSequence):
    """A list whose contents are private to each thread."""

    def __init__(self):
        self._local = threading.local()

    @property
    def _data(self) -> list:
        if not hasattr(self._local, "data"):
            self._local.data = []
        return self._local.data

    def __getitem__(self, i):
        return self._data[i]

    def __setitem__(self, i, value):
        self._data[i] = value

    def __delitem__(self, i):
        del self._data[i]

    def __len__(self):
        return len(self._data)

    def insert(self, i, value):
        self._data.insert(i, value)

    def __repr__(self):
        return repr(self._data)

    def clear(self) -> None:
        self._local.data = []

    def snapshot(self) -> list:
        return list(self._data)
