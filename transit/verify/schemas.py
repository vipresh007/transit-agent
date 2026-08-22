"""
Stage 3: typed output.

Prose answers are unverifiable. "The last streetcar is at 1:23am" can only be
checked by substring matching, which is how we ended up with false negatives
over a curly apostrophe. A typed object can be checked by field.

The schema does three jobs at once:
  1. Documents what a good answer contains (the model sees the JSON schema).
  2. Rejects malformed answers, producing an error the agent can retry against.
  3. Makes constraint checking possible -- which is all stage 7 is.

Validators here are deliberately strict. A validator that accepts anything
provides no signal, and signal is the entire point.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

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


class Leg(BaseModel):
    """One continuous movement: a single vehicle ride, or a walk."""

    mode: Literal["subway", "streetcar", "bus", "walk"]
    route: str | None = Field(
        None, description="Route short name like '501' or 'Line 1'. Null for walking."
    )
    origin: str = Field(description="Stop or place name where this leg starts")
    destination: str = Field(description="Stop or place name where this leg ends")
    depart: str = Field(description="GTFS time, HH:MM:SS, may exceed 24:00:00")
    arrive: str = Field(description="GTFS time, HH:MM:SS, may exceed 24:00:00")

    @field_validator("depart", "arrive")
    @classmethod
    def valid_gtfs_time(cls, v: str) -> str:
        if not GTFS_TIME.match(v):
            raise ValueError(
                f"{v!r} is not a valid GTFS time. Use HH:MM:SS, hours 0-47 "
                f"(e.g. '25:23:00' for 1:23am next day). Not '1:23 AM'."
            )
        return v

    @model_validator(mode="after")
    def arrive_after_depart(self) -> Leg:
        if to_seconds(self.arrive) < to_seconds(self.depart):
            raise ValueError(
                f"leg arrives ({self.arrive}) before it departs ({self.depart}). "
                f"If it crosses midnight, use 24+ hour notation."
            )
        return self

    @model_validator(mode="after")
    def transit_legs_need_a_route(self) -> Leg:
        if self.mode != "walk" and not self.route:
            raise ValueError(f"{self.mode} leg must name a route")
        return self

    @property
    def duration_min(self) -> int:
        return (to_seconds(self.arrive) - to_seconds(self.depart)) // 60

    def __str__(self) -> str:
        label = f"{self.mode} {self.route}" if self.route else "walk"
        return (
            f"{to_civil(self.depart)}  {label}: {self.origin} -> "
            f"{self.destination}  ({self.duration_min} min)"
        )


class Itinerary(BaseModel):
    """A journey, OR a reasoned statement that no journey exists.

    `legs` was min_length=1, which made "this trip is impossible" impossible
    to express. Asked for a bus-free route to a bus-only destination, the
    model emitted a zero-minute walk from Scarborough Town Centre to itself
    and wrote in the caveats: "the single walk leg is a placeholder to satisfy
    schema requirements."

    It told us plainly that the schema forced it to fabricate. A schema that
    can only represent success will get success-shaped output regardless of
    what happened — so "no route" is now a first-class outcome.
    """

    summary: str = Field(description="One sentence describing the trip")
    feasible: bool = Field(
        True,
        description="False if no journey satisfies the constraints. Then legs "
        "MUST be empty and infeasible_reason MUST explain why.",
    )
    infeasible_reason: str | None = Field(
        None, description="Why no journey exists. Required when feasible=False."
    )
    legs: list[Leg] = Field(default_factory=list)
    caveats: list[str] = Field(
        default_factory=list,
        description="Anything unverified: holiday service, real-time delays, "
        "assumptions made. Empty list if genuinely none.",
    )

    @model_validator(mode="after")
    def feasible_or_explained(self) -> Itinerary:
        if self.feasible and not self.legs:
            raise ValueError(
                "a feasible itinerary needs at least one leg; if no journey "
                "exists set feasible=false and give infeasible_reason"
            )
        if not self.feasible:
            if self.legs:
                raise ValueError(
                    "an infeasible itinerary must have NO legs — do not invent "
                    "a placeholder leg to satisfy the schema"
                )
            if not (self.infeasible_reason or "").strip():
                raise ValueError("feasible=false requires infeasible_reason")
        return self

    @model_validator(mode="after")
    def legs_are_chronological(self) -> Itinerary:
        for a, b in zip(self.legs, self.legs[1:]):
            if to_seconds(b.depart) < to_seconds(a.arrive):
                raise ValueError(
                    f"leg departing {b.depart} starts before the previous leg "
                    f"arrives at {a.arrive} — legs must be in time order and "
                    f"must not overlap"
                )
        return self

    @property
    def total_min(self) -> int:
        if not self.legs:
            return 0
        return (to_seconds(self.legs[-1].arrive) - to_seconds(self.legs[0].depart)) // 60

    @property
    def transfers(self) -> int:
        return max(0, sum(1 for leg in self.legs if leg.mode != "walk") - 1)

    def connection_gaps(self) -> list[tuple[int, int]]:
        """Minutes between arriving on one leg and departing on the next.

        Not a validator -- a tight connection is legal, just risky. Stage 7
        turns these into constraints the agent has to satisfy.
        """
        return [
            (i, (to_seconds(b.depart) - to_seconds(a.arrive)) // 60)
            for i, (a, b) in enumerate(zip(self.legs, self.legs[1:]))
        ]

    def risky_connections(self) -> list[tuple[int, int]]:
        """Gaps that a traveller could actually miss.

        A gap only matters between two TRANSIT legs — that's where you're
        racing a departure. Walk->transit gaps are just waiting, and
        transit->walk gaps are meaningless, so flagging those produced four
        warnings on a perfectly comfortable itinerary and trained the reader
        to ignore all of them.
        """
        risky = []
        for i, (a, b) in enumerate(zip(self.legs, self.legs[1:])):
            if a.mode == "walk" or b.mode == "walk":
                continue
            gap = (to_seconds(b.depart) - to_seconds(a.arrive)) // 60
            if gap < 5:
                risky.append((i, gap))
        return risky

    def render(self) -> str:
        if not self.feasible:
            lines = [self.summary, "",
                     f"  NO JOURNEY FOUND: {self.infeasible_reason}"]
            if self.caveats:
                lines += ["", "  Details:"] + [f"    - {c}" for c in self.caveats]
            return "\n".join(lines)

        lines = [self.summary, ""]
        for leg in self.legs:
            lines.append(f"  {leg}")
        lines.append("")
        lines.append(
            f"  Total {self.total_min} min, {self.transfers} transfer(s)"
        )
        for i, gap in self.risky_connections():
            lines.append(
                f"  ! tight transfer after leg {i + 1}: {gap} min to make "
                f"the {self.legs[i + 1].route}"
            )
        if self.caveats:
            lines.append("")
            lines.append("  Caveats:")
            lines.extend(f"    - {c}" for c in self.caveats)
        return "\n".join(lines)
