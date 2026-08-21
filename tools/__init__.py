"""Tools the agent can call.

Every tool is a plain Python function taking JSON-friendly arguments and
returning a string. There is no magic: the "agent" part is just the model
choosing which of these to call and with what arguments.

Layout:
    geo.py       geocoding, weather, points of interest (external APIs)
    transit.py   GTFS schema access and purpose-built stop/trip lookups
    journey.py   end-to-end journey planning with transfers
    guides.py    hybrid search over the Wikivoyage travel guides
    registry.py  schemas + dispatch table

Import from the package, not the submodules:
    from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS, SCHEDULE_TOOLS
"""

from .geo import find_pois, geocode, get_weather
from .guides import guides_status, search_guides
from .journey import plan_journey
from .registry import SCHEDULE_TOOLS, TOOL_FUNCTIONS, TOOL_SCHEMAS
from .transit import (
    DB_PATH,
    describe_transit_schema,
    find_direct_trips,
    find_nearby_stops,
    query_transit,
)

__all__ = [
    "DB_PATH",
    "SCHEDULE_TOOLS",
    "TOOL_FUNCTIONS",
    "TOOL_SCHEMAS",
    "describe_transit_schema",
    "find_direct_trips",
    "find_nearby_stops",
    "find_pois",
    "geocode",
    "guides_status",
    "get_weather",
    "plan_journey",
    "query_transit",
    "search_guides",
]
